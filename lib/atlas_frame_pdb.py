"""Pull a single representative ATLAS frame out of the shared hdf5 store and write it as a
minimal PDB.

Shared by every comparison model that needs a *structure file* rather than raw tensors --
currently comparisons/MapDiff/{data/generate_graph_atlas.py,eval_atlas.py}, whose upstream
featurizer (`data.generate_graph_cath.pdb2graph`) parses PDBs with Biopython and shells out
to DSSP. Rather than fork that featurizer, we round-trip one frame through a temporary PDB.

The splits and the per-protein representative frame come from the same
`atlas_cross_val_index.csv` (`cross_val` / `random_indices` columns) that
scripts/train_residue_classifier.py, comparisons/gvp-pytorch/train.py and
comparisons/DynamicMPNN/train.py use, so every model is trained/evaluated on identical
proteins and frames.

hdf5 layout (see scripts/load_dynamics.py): each trajectory group holds
`coordinates` of shape (num_res, T, num_atoms, 3) with the atom axis in `BACKBONE_ATOMS`
order (CA, N, C, O), and `residues`, a length-num_res array of 3-letter residue-name bytes.
"""
import numpy as np
import pandas as pd

# `load_dynamics` (scripts/) is imported flat, the same convention the rest of the repo's
# CHTC entry points rely on -- HTCondor transfers it in next to this file.
from load_dynamics import BACKBONE_ATOMS

# Force-field/protonation-state variants ATLAS inherits from its MD setup; PDB parsers and
# DSSP only understand the canonical names. Mirrors
# `lib.residue_classifier_dataset.ResidueClassifierDataset.RESIDUE_ALIASES`.
RESIDUE_ALIASES = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS",
                   "HIE": "HIS", "HIP": "HIS", "ASH": "ASP", "GLH": "GLU",
                   "LYN": "LYS", "CYM": "CYS", "CYX": "CYS", "MSE": "MET"}

# Written out in the conventional backbone order rather than the hdf5's CA-first storage
# order, so the temporary PDB looks like an ordinary deposited structure to Biopython/DSSP.
PDB_ATOM_ORDER = ("N", "CA", "C", "O")

# Every trajectory group carries these; used to tell trajectory groups apart from whatever
# else lives in the store. Kept in sync with
# `lib.residue_classifier_dataset.ResidueClassifierDataset.REQUIRED_DATASETS`.
REQUIRED_DATASETS = ("coordinates", "dihedrals", "spinet_features", "frame_maps", "residues")


def get_atlas_splits(cross_val_csv, val_fold=0):
    """(train_pdbs, val_pdbs) from the shared cross-validation index.

    `val_fold` is held out; every other fold trains. Returns plain lists of pdb codes.
    """
    index_df = cross_val_csv if isinstance(cross_val_csv, pd.DataFrame) else pd.read_csv(cross_val_csv)
    val_mask = index_df["cross_val"] == val_fold
    return index_df.loc[~val_mask, "pdb"].tolist(), index_df.loc[val_mask, "pdb"].tolist()


def index_h5_groups(h5_file):
    """Map pdb code -> hdf5 group path for every trajectory group in the store."""
    lookup = {}

    def visit(name, obj):
        if hasattr(obj, "keys") and all(ds in obj for ds in REQUIRED_DATASETS):
            lookup.setdefault(name.split("/")[0], name)

    h5_file.visititems(visit)
    return lookup


def get_frame_index(index_df, pdb_id):
    """The protein's representative frame (`random_indices`), i.e. the same single frame
    scripts/train_residue_classifier.py and comparisons/gvp-pytorch/train.py evaluate on."""
    rows = index_df.loc[index_df["pdb"] == pdb_id, "random_indices"]
    if rows.empty:
        raise KeyError(f"{pdb_id} has no row in the ATLAS cross-validation index")
    return int(rows.iloc[0])


def extract_frame(h5_file, group_name, frame_idx):
    """One frame of one trajectory, ready for `write_frame_pdb`.

    Returns {'coords': (num_res, num_atoms, 3) float array in `BACKBONE_ATOMS` order,
             'residues': list of canonical 3-letter residue names}.
    """
    coords = np.asarray(h5_file[group_name + "/coordinates"][:, frame_idx], dtype=np.float64)
    residues = [RESIDUE_ALIASES.get(name, name)
                for name in (r.decode().strip()[:3].upper()
                             for r in h5_file[group_name + "/residues"][:])]
    return {"coords": coords, "residues": residues}


def write_frame_pdb(path, frame, chain_id="A"):
    """Write `extract_frame`'s output as a single-model, single-chain PDB.

    Residues are renumbered 1..N contiguously: ATLAS trajectories are already gap-free, and
    a contiguous numbering keeps DSSP's residue keys lined up with the graph's node order
    (see `data.generate_graph_cath.get_struc2ndRes`, which matches Biopython residues to
    DSSP keys by (chain, resseq)).
    """
    coords = frame["coords"]
    residues = frame["residues"]
    if coords.shape[0] != len(residues):
        raise ValueError(f"{coords.shape[0]} residue coordinate rows vs {len(residues)} residue names")

    atom_slots = [(name, BACKBONE_ATOMS.index(name)) for name in PDB_ATOM_ORDER]

    lines = []
    serial = 1
    for res_i, res_name in enumerate(residues, start=1):
        for atom_name, atom_idx in atom_slots:
            x, y, z = coords[res_i - 1, atom_idx]
            if not np.isfinite((x, y, z)).all():
                continue
            # Columns per the PDB v3.3 ATOM record spec. Atom names start in column 14 for
            # the 1-2 character element symbols we emit here (N/C/O), never column 13.
            lines.append(
                f"ATOM  {serial:5d} {atom_name:<4s} {res_name:>3s} {chain_id}{res_i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {atom_name[0]:>2s}"
            )
            serial += 1
    lines.append(f"TER   {serial:5d}      {residues[-1]:>3s} {chain_id}{len(residues):4d}")
    lines.append("END")

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path
