"""PiFold-compatible dataset built from static structures.

PiFold is a static-structure inverse-folding model, so by default it is trained and evaluated
on each protein's deposited PDB entry rather than on a frame pulled out of the MD trajectory.
For ATLAS, `lib/relaxed_pdb.py` downloads the RCSB entry, picks the chain the dataset id
names, and crops it to the residues the trajectory covers -- using the trajectory's own residue
list as the reference sequence. For mdCATH, the structure is the CATH domain file itself
(`relaxed_pdb.load_cath_structures`, read from the extracted `mdcath_pdbs.tar.gz` in
`cath_dir`), used whole: no download and no cropping.

Pass ``structure_source='frame'`` (`--structure_source frame`) to feed it one random frame per
protein's trajectory instead (`lib/traj_frames.py`), with splits, filtering and featurization
otherwise untouched -- so "deposited structure vs. a single MD frame" is a one-flag ablation.
A frame needs no download and no cropping (the trajectory's residues *are* the reference), and
the frame is picked deterministically from ``(frame_seed, protein id)``.

(`comparisons/DynamicMPNN` deliberately still trains on MD conformer ensembles -- that
ensemble input is the thing being benchmarked.)

ATLAS and mdCATH are trained and evaluated separately (`--data_name ATLAS` / `MDCATH`),
matching scripts/train_residue_classifier.py's `--ds-name`. Splits come from
`lib/dataset_splits.py` and there are three of them: ATLAS holds out `cross_val` fold
`--val_fold` for validation and fold `--test_fold` for test; mdCATH uses its topology
split's train/validation/test rows as-is. 'valid' and 'test' are disjoint, and neither is
trained on.

Produces items shaped like API.cath_dataset.CATH: dicts with 'title', 'seq', 'N', 'CA', 'C',
'O' (and 'category'/'score' for test), so it plugs directly into API.featurizer.featurize_GTrans.
"""
import os
import sys

import torch.utils.data as data

from .utils import cached_property

# lib/relaxed_pdb.py + lib/dataset_splits.py are copied in flat next to PiFold's entry point
# (see ../README.md and ../run_pifold.sh) -- fall back to walking up to a lib/ directory for
# local/dev runs from inside the source tree.
try:
    import dataset_splits
    import relaxed_pdb
    import traj_frames
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import dataset_splits
    import relaxed_pdb
    import traj_frames

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

# `relaxed_pdb` stacks backbone atoms in scripts/load_dynamics.py's BACKBONE_ATOMS order;
# PiFold wants them as named arrays.
ATOM_INDEX = {atom: i for i, atom in enumerate(relaxed_pdb.BACKBONE_ATOMS)}


class RelaxedStructures(data.Dataset):
    """Static structures for one of the shared datasets.

    `path` is the directory the split index csv and trajectory store were staged into
    (`--data_root`); downloaded RCSB entries are cached under `pdb_cache` inside it.

    `structure_source` selects what a "structure" is: ``'relaxed'`` (the deposited RCSB entry,
    the default and what the class is named for) or ``'frame'`` (one frame of the protein's MD
    trajectory). `frame_seed` seeds the per-protein frame draw and `frame_index` pins one index
    for every protein instead; both are ignored for ``'relaxed'``.
    """

    def __init__(self, path='./', mode='train', max_length=dataset_splits.MAX_LENGTH, data=None,
                 ds_name='atlas', index_csv=None, traj_h5=None, pdb_cache=None, cath_dir=None,
                 val_fold=dataset_splits.DEFAULT_VAL_FOLD,
                 test_fold=dataset_splits.DEFAULT_TEST_FOLD,
                 structure_source='relaxed', frame_seed=dataset_splits.SEED, frame_index=None):
        self.path = path
        self.mode = mode
        self.max_length = max_length
        self.ds_name = ds_name.lower()
        self.index_csv = index_csv or os.path.join(path, dataset_splits.default_index_csv(self.ds_name))
        # With structure_source='relaxed' the trajectory store is read *only* for each
        # protein's reference residue sequence, and traj_h5='' featurizes whole deposited
        # chains uncropped; with 'frame' it is the coordinate source and is required.
        self.traj_h5 = os.path.join(path, dataset_splits.default_h5(self.ds_name)) if traj_h5 is None else traj_h5
        self.pdb_cache = pdb_cache or os.path.join(path, 'pdb_cache')
        # mdCATH relaxed structures: the extracted CATH domain archive.
        self.cath_dir = cath_dir or os.path.join(path, relaxed_pdb.DEFAULT_CATH_DIR)
        self.val_fold = val_fold
        self.test_fold = test_fold
        if structure_source not in ('relaxed', 'frame'):
            raise ValueError("structure_source must be 'relaxed' or 'frame', got {!r}".format(
                structure_source))
        self.structure_source = structure_source
        self.frame_seed = frame_seed
        self.frame_index = frame_index
        if data is None:
            self.data = self.cache_data[mode]
        else:
            self.data = data

    @cached_property
    def cache_data(self):
        if not os.path.exists(self.index_csv):
            raise FileNotFoundError("no such file: {} !!!".format(self.index_csv))

        train_ids, val_ids, test_ids = dataset_splits.get_splits(
            self.ds_name, self.index_csv, val_fold=self.val_fold, test_fold=self.test_fold)
        split_ids = {'train': train_ids, 'valid': val_ids, 'test': test_ids}
        all_split_ids = train_ids + val_ids + test_ids

        if self.structure_source == 'frame':
            if not self.traj_h5:
                raise ValueError("structure_source='frame' needs a trajectory store (traj_h5)")
            if not os.path.exists(self.traj_h5):
                raise FileNotFoundError("no such file: {} !!!".format(self.traj_h5))
            # No cropping and no download: a frame already spans exactly the simulated
            # residues (for mdCATH, exactly the domain).
            records, failures = traj_frames.load_frame_structures(
                all_split_ids, self.traj_h5, seed=self.frame_seed, frame_index=self.frame_index)
            if failures:
                print('{}: {} protein ids had no usable trajectory frame'.format(
                    self.ds_name, len(failures)))
        elif self.ds_name == 'mdcath':
            # The CATH domain files, whole -- no RCSB download, no cropping.
            records, failures = relaxed_pdb.load_cath_structures(all_split_ids, self.cath_dir)
            if failures:
                print('{}: {} protein ids had no usable CATH domain file'.format(
                    self.ds_name, len(failures)))
        else:
            reference_seqs = {}
            if self.traj_h5:
                if not os.path.exists(self.traj_h5):
                    raise FileNotFoundError("no such file: {} !!!".format(self.traj_h5))
                reference_seqs = relaxed_pdb.reference_seqs_from_h5(self.traj_h5, all_split_ids)

            records, failures = relaxed_pdb.load_relaxed_structures(
                all_split_ids, cache_dir=self.pdb_cache, reference_seqs=reference_seqs)
            if failures:
                print('{}: {} protein ids had no usable deposited structure'.format(self.ds_name, len(failures)))

        alphabet_set = set(ALPHABET)
        data_dict = {'train': [], 'valid': [], 'test': []}
        # Every reason a protein in the split index does *not* make it into the dataset, kept
        # so the run can report exactly which ids it scored -- MapDiff and DynamicMPNN drop
        # proteins for their own reasons (DSSP failures, too few conformers), and the test
        # sets only line up if each model says out loud what it dropped.
        self.dropped = {'no_structure': [], 'nonstandard_residue': [], 'too_long': []}

        for mode, protein_ids in split_ids.items():
            for protein_id in protein_ids:
                record = records.get(protein_id)
                if record is None:
                    self.dropped['no_structure'].append(protein_id)
                    continue

                seq = record['seq']
                if set(seq).difference(alphabet_set):
                    self.dropped['nonstandard_residue'].append(protein_id)
                    continue
                if len(seq) > self.max_length:
                    self.dropped['too_long'].append(protein_id)
                    continue

                coords = record['coords']
                entry = {
                    'title': protein_id,
                    'seq': seq,
                    'CA': coords[:, ATOM_INDEX['CA']].astype('float32'),
                    'C': coords[:, ATOM_INDEX['C']].astype('float32'),
                    'O': coords[:, ATOM_INDEX['O']].astype('float32'),
                    'N': coords[:, ATOM_INDEX['N']].astype('float32'),
                    'category': self.ds_name.upper(),
                }
                if mode == 'test':
                    entry['score'] = 100.0
                data_dict[mode].append(entry)

        self.dropped = {reason: sorted(set(ids)) for reason, ids in self.dropped.items()}

        print('{}: {} train / {} valid / {} test {} structures loaded '
              '(val fold {}, test fold {})'.format(
                  self.ds_name, len(data_dict['train']), len(data_dict['valid']),
                  len(data_dict['test']), self.structure_source, self.val_fold, self.test_fold))
        for reason, ids in self.dropped.items():
            if ids:
                print('{}: dropped {} protein(s) -- {}: {}'.format(self.ds_name, len(ids), reason, ids))
        return data_dict

    def split_ids(self, mode=None):
        """The protein ids actually in one split, in dataset order -- logged per run so the
        held-out sets of all four models can be diffed after the fact."""
        return [entry['title'] for entry in self.cache_data[mode or self.mode]]

    def change_mode(self, mode):
        self.mode = mode
        self.data = self.cache_data[mode]

    def __len__(self):
        return len(self.data)

    def get_item(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.data[index]
