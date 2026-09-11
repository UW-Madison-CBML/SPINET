import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.utils.data as data
from Bio.PDB.Polypeptide import protein_letters_3to1

from .utils import cached_property

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
# Must match scripts/load_dynamics.py's BACKBONE_ATOMS order (the order `coordinates` is
# stored in for every group in atlas_data.h5).
BACKBONE_ATOMS = ('CA', 'N', 'C', 'O')
REQUIRED_DATASETS = ('coordinates', 'residues')

# protein_letters_3to1 only recognizes the 20 canonical residue names; map a
# few common variants seen in crystal/MD structures onto their canonical letter.
_THREE_TO_ONE_EXTRA = {'MSE': 'M', 'SEC': 'C', 'PYL': 'K', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H'}


def _res_to_one(resname):
    resname = resname.upper()
    if resname in _THREE_TO_ONE_EXTRA:
        return _THREE_TO_ONE_EXTRA[resname]
    return protein_letters_3to1.get(resname)


def _get_atlas_splits(cross_val_csv, val_fold=0):
    """(train_pdbs, val_pdbs) -- the same partition scripts/train_residue_classifier.py and
    comparisons/{gvp-pytorch,DynamicMPNN}/train.py use (fold 0 of the `cross_val` column
    held out). `val_pdbs` doubles as the test split too -- ATLAS only has one held-out fold.
    """
    index = pd.read_csv(cross_val_csv)
    val_mask = index["cross_val"] == val_fold
    val_pdbs = index.loc[val_mask, "pdb"].tolist()
    train_pdbs = index.loc[~val_mask, "pdb"].tolist()
    return train_pdbs, val_pdbs


def _index_h5_groups(h5_file, required=REQUIRED_DATASETS):
    """Map top-level pdb_code -> the nested hdf5 group path actually holding the datasets."""
    lookup = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and all(ds in obj for ds in required):
            lookup.setdefault(name.split("/")[0], name)

    h5_file.visititems(visit)
    return lookup


def _extract_frame(h5_file, group_name, frame_idx):
    """One frame of one protein's trajectory -> 1-letter sequence + backbone coordinates."""
    raw_residues = h5_file[group_name + "/residues"][:]
    seq = ''.join((_res_to_one(r.decode().strip()[:3]) or 'X') for r in raw_residues)

    frame = np.asarray(h5_file[group_name + "/coordinates"][:, frame_idx])  # (R, 4, 3), BACKBONE_ATOMS order
    coords = {atom: frame[:, i].astype(np.float32) for i, atom in enumerate(BACKBONE_ATOMS)}
    return seq, coords


class ATLAS(data.Dataset):
    """PiFold-compatible dataset built directly from the shared ATLAS hdf5 store
    (`atlas_data.h5`) and cross-validation index (`atlas_cross_val_index.csv`)

    Fold 0 of the `cross_val` column is held out as the 'valid' split, and reused as
    'test' too, since ATLAS only has one held-out fold. Each protein's representative
    frame is its `random_indices` column entry (the same frame
    scripts/train_residue_classifier.py / comparisons/gvp-pytorch/train.py pick).

    Expects `path` to contain `atlas_data.h5` and `atlas_cross_val_index.csv` (flat
    names, overridable via `h5_name`/`csv_name`).

    Produces items shaped like API.cath_dataset.CATH: dicts with
    'title', 'seq', 'N', 'CA', 'C', 'O' (and 'category'/'score' for test),
    so it plugs directly into API.featurizer.featurize_GTrans.
    """

    def __init__(self, path='./', mode='train', max_length=500, data=None,
                 h5_name='atlas_data.h5', csv_name='atlas_cross_val_index.csv', val_fold=0):
        self.path = path
        self.mode = mode
        self.max_length = max_length
        self.h5_name = h5_name
        self.csv_name = csv_name
        self.val_fold = val_fold
        if data is None:
            self.data = self.cache_data[mode]
        else:
            self.data = data

    @cached_property
    def cache_data(self):
        h5_path = os.path.join(self.path, self.h5_name)
        csv_path = os.path.join(self.path, self.csv_name)
        if not os.path.exists(h5_path):
            raise FileNotFoundError("no such file: {} !!!".format(h5_path))
        if not os.path.exists(csv_path):
            raise FileNotFoundError("no such file: {} !!!".format(csv_path))

        index_df = pd.read_csv(csv_path)
        train_pdbs, val_pdbs = _get_atlas_splits(csv_path, val_fold=self.val_fold)
        # ATLAS only has one held-out fold -- reuse it as both 'valid' and 'test'.
        split_pdbs = {'train': train_pdbs, 'valid': val_pdbs, 'test': val_pdbs}

        alphabet_set = set(ALPHABET)
        data_dict = {'train': [], 'valid': [], 'test': []}

        with h5py.File(h5_path, 'r') as h5_file:
            group_lookup = _index_h5_groups(h5_file)

            for mode, pdb_ids in split_pdbs.items():
                for pdb_id in tqdm(pdb_ids, desc='loading ATLAS/{}'.format(mode)):
                    group_name = group_lookup.get(pdb_id)
                    if group_name is None:
                        continue

                    frame_idx = int(index_df.loc[index_df["pdb"] == pdb_id, "random_indices"].iloc[0])
                    seq, coords = _extract_frame(h5_file, group_name, frame_idx)

                    bad_chars = set(seq).difference(alphabet_set)
                    if len(bad_chars) > 0:
                        continue
                    if len(seq) > self.max_length:
                        continue

                    entry = {
                        'title': pdb_id,
                        'seq': seq,
                        'CA': coords['CA'],
                        'C': coords['C'],
                        'O': coords['O'],
                        'N': coords['N'],
                        'category': 'ATLAS',
                    }
                    if mode == 'test':
                        entry['score'] = 100.0

                    data_dict[mode].append(entry)

        return data_dict

    def change_mode(self, mode):
        self.mode = mode
        self.data = self.cache_data[mode]

    def __len__(self):
        return len(self.data)

    def get_item(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.data[index]
