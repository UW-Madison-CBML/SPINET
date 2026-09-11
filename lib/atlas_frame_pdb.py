"""Shared ATLAS-hdf5 helpers for baselines that expect a single representative structure
per protein (comparisons/MapDiff), split the exact same way as
scripts/train_residue_classifier.py, comparisons/gvp-pytorch/train.py, and
comparisons/DynamicMPNN/train.py: fold 0 of atlas_cross_val_index.csv's `cross_val` column
held out (used as validation -- and reused as test too, for models that want a separate
test split, since ATLAS only has one held-out fold), folds 1-4 for training. The
representative frame per protein is `random_indices`, the same column
scripts/train_residue_classifier.py and comparisons/gvp-pytorch/train.py pick their single
frame from.
"""
import h5py
import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import protein_letters_3to1

# Must match scripts/load_dynamics.py's BACKBONE_ATOMS order (the order `coordinates` is
# stored in for every group in atlas_data.h5).
BACKBONE_ATOMS = ("CA", "N", "C", "O")
REQUIRED_DATASETS = ("coordinates", "residues")

# protein_letters_3to1 only recognizes the 20 canonical residue names; map a few common
# variants seen in crystal/MD structures onto their canonical letter (mirrors the mapping
# comparisons/PiFold/atlas_dataset.py used against raw PDB files).
_THREE_TO_ONE_EXTRA = {'MSE': 'M', 'SEC': 'C', 'PYL': 'K', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H'}


def res_to_one(resname):
    resname = resname.upper()
    if resname in _THREE_TO_ONE_EXTRA:
        return _THREE_TO_ONE_EXTRA[resname]
    return protein_letters_3to1.get(resname)


def get_atlas_splits(cross_val_csv, val_fold=0):
    """(train_pdbs, val_pdbs), same partition scripts/train_residue_classifier.py and
    comparisons/{gvp-pytorch,DynamicMPNN}/train.py use. `val_pdbs` doubles as the test
    split for callers that need one -- ATLAS only has one held-out fold."""
    index = pd.read_csv(cross_val_csv)
    val_mask = index["cross_val"] == val_fold
    val_pdbs = index.loc[val_mask, "pdb"].tolist()
    train_pdbs = index.loc[~val_mask, "pdb"].tolist()
    return train_pdbs, val_pdbs


def get_frame_index(index_df, pdb_id):
    """The pre-selected representative frame for one protein (the `random_indices` column
    comparisons/gvp-pytorch/train.py and scripts/train_residue_classifier.py also key off)."""
    return int(index_df.loc[index_df["pdb"] == pdb_id, "random_indices"].iloc[0])


def index_h5_groups(h5_file, required=REQUIRED_DATASETS):
    """Map top-level pdb_code (matches atlas_cross_val_index.csv's `pdb` column, and
    `atlas_data.h5`'s own top-level keys) -> the nested hdf5 group path actually holding
    the datasets (e.g. "1ab2_A/temp_1/1ab2_A_R1")."""
    lookup = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and all(ds in obj for ds in required):
            lookup.setdefault(name.split("/")[0], name)

    h5_file.visititems(visit)
    return lookup


def extract_frame(h5_file, group_name, frame_idx):
    """One frame of one protein's trajectory -> backbone coordinates + sequence.

    Returns a dict with 'seq' (1-letter string), 'residues' (list of 3-letter codes), and
    'CA'/'N'/'C'/'O' ([R, 3] float32 arrays) -- the same shape
    comparisons/PiFold/atlas_dataset.py's `parse_pdb_backbone` used to produce from a raw
    PDB file.
    """
    raw_residues = h5_file[group_name + "/residues"][:]
    residues = [r.decode().strip()[:3].upper() for r in raw_residues]
    seq = ''.join(res_to_one(r) or 'X' for r in residues)

    frame = np.asarray(h5_file[group_name + "/coordinates"][:, frame_idx])  # (R, 4, 3), BACKBONE_ATOMS order
    coords = {atom: frame[:, i].astype(np.float32) for i, atom in enumerate(BACKBONE_ATOMS)}

    return {'seq': seq, 'residues': residues, **coords}


def write_frame_pdb(out_path, frame, chain_id="A"):
    """Write one frame's backbone as a minimal single-model, single-chain PDB file, for
    featurizers that need an on-disk .pdb rather than the raw arrays `extract_frame`
    already returns (e.g. comparisons/MapDiff, which runs DSSP on it).

    frame: a dict as returned by extract_frame() -- needs 'residues' and per-atom
    'CA'/'N'/'C'/'O' [R, 3] arrays.
    """
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.PDBIO import PDBIO
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

    structure = Structure("atlas_frame")
    model = Model(0)
    structure.add(model)
    chain = Chain(chain_id)
    model.add(chain)

    serial = 1
    for res_i, resname in enumerate(frame['residues']):
        residue = Residue((' ', res_i + 1, ' '), resname, '')
        for atom_name in ("N", "CA", "C", "O"):
            coord = frame[atom_name][res_i].astype(float)
            residue.add(Atom(atom_name, coord, 0.0, 1.0, ' ', f' {atom_name:<3s}', serial, element=atom_name[0]))
            serial += 1
        chain.add(residue)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path))
