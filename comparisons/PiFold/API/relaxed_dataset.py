"""PiFold-compatible dataset built from *relaxed* (deposited) PDB structures.

PiFold is a static-structure inverse-folding model, so it is trained and evaluated on each
protein's deposited PDB entry rather than on a frame pulled out of the MD trajectory (which
is what this dataset used to do). `lib/relaxed_pdb.py` downloads the RCSB entry, picks the
chain the dataset id names, and crops it to the residues the trajectory covers -- using the
trajectory's own residue list as the reference sequence -- so PiFold sees the same residues
as every other model in the comparison. For mdCATH, that cropping is also what reduces a
deposited chain to the single CATH *domain* the `domain` id names (`12asA00` -> entry `12as`,
chain `A`, domain 02).

(`comparisons/DynamicMPNN` deliberately still trains on MD conformer ensembles -- that
ensemble input is the thing being benchmarked -- and uses relaxed structures only as its
scRMSD reference.)

ATLAS and mdCATH are trained and evaluated separately (`--data_name ATLAS` / `MDCATH`),
matching scripts/train_residue_classifier.py's `--ds-name`. Splits come from
`lib/dataset_splits.py`: ATLAS holds out one `cross_val` fold, mdCATH trains on its
topology split's train+test rows and evaluates on `validation` ONLY. Neither dataset has a
further held-out set, so the held-out split is reused as both 'valid' and 'test'.

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
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import dataset_splits
    import relaxed_pdb

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

# `relaxed_pdb` stacks backbone atoms in scripts/load_dynamics.py's BACKBONE_ATOMS order;
# PiFold wants them as named arrays.
ATOM_INDEX = {atom: i for i, atom in enumerate(relaxed_pdb.BACKBONE_ATOMS)}


class RelaxedStructures(data.Dataset):
    """Relaxed (deposited) structures for one of the shared datasets.

    `path` is the directory the split index csv and trajectory store were staged into
    (`--data_root`); downloaded RCSB entries are cached under `pdb_cache` inside it.
    """

    def __init__(self, path='./', mode='train', max_length=dataset_splits.MAX_LENGTH, data=None,
                 ds_name='atlas', index_csv=None, traj_h5=None, pdb_cache=None,
                 val_fold=dataset_splits.DEFAULT_VAL_FOLD):
        self.path = path
        self.mode = mode
        self.max_length = max_length
        self.ds_name = ds_name.lower()
        self.index_csv = index_csv or os.path.join(path, dataset_splits.default_index_csv(self.ds_name))
        # The trajectory store is read *only* for each protein's reference residue sequence.
        # Pass traj_h5='' to featurize whole deposited chains uncropped.
        self.traj_h5 = os.path.join(path, dataset_splits.default_h5(self.ds_name)) if traj_h5 is None else traj_h5
        self.pdb_cache = pdb_cache or os.path.join(path, 'pdb_cache')
        self.val_fold = val_fold
        if data is None:
            self.data = self.cache_data[mode]
        else:
            self.data = data

    @cached_property
    def cache_data(self):
        if not os.path.exists(self.index_csv):
            raise FileNotFoundError("no such file: {} !!!".format(self.index_csv))

        train_ids, val_ids = dataset_splits.get_splits(self.ds_name, self.index_csv, val_fold=self.val_fold)
        # Neither dataset has a further held-out set -- reuse the held-out split as 'test'.
        split_ids = {'train': train_ids, 'valid': val_ids, 'test': val_ids}

        reference_seqs = {}
        if self.traj_h5:
            if not os.path.exists(self.traj_h5):
                raise FileNotFoundError("no such file: {} !!!".format(self.traj_h5))
            reference_seqs = relaxed_pdb.reference_seqs_from_h5(self.traj_h5, train_ids + val_ids)

        records, failures = relaxed_pdb.load_relaxed_structures(
            train_ids + val_ids, cache_dir=self.pdb_cache, reference_seqs=reference_seqs)
        if failures:
            print('{}: {} protein ids had no usable deposited structure'.format(self.ds_name, len(failures)))

        alphabet_set = set(ALPHABET)
        data_dict = {'train': [], 'valid': [], 'test': []}
        # Every reason a protein in the split index does *not* make it into the dataset, kept
        # so the run can report exactly which ids it scored -- MapDiff and DynamicMPNN drop
        # proteins for their own reasons (DSSP failures, too few conformers), and the held-out
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

        # 'valid' and 'test' are the same proteins, so each drop is counted twice there --
        # de-duplicate before reporting.
        self.dropped = {reason: sorted(set(ids)) for reason, ids in self.dropped.items()}

        print('{}: {} train / {} valid / {} test structures loaded (fold {} held out)'.format(
            self.ds_name, len(data_dict['train']), len(data_dict['valid']), len(data_dict['test']),
            self.val_fold))
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
