#!/usr/bin/env python3
"""Export mdCATH trajectories from the shards to files that PyMOL can open.

For each trajectory the script writes two files into --out-dir:
  <traj_id>.pdb  All frames as PDB models. Each model holds the backbone atoms
                 N, CA, C, O. PDB model k+1 is trajectory frame k (0-based).
                 The script superposes every frame on frame_i with the CA atoms.
  <traj_id>.pml  A PyMOL script. It loads the PDB file as a movie and makes two
                 extra objects for the two most different frames.

The script reads the shard name, frame_i and frame_j from the CSV that
mdcath_320_test_max_rmsd.py writes.

Examples
--------
# The 5 trajectories with the largest RMSD:
python mdcath_export_for_pymol.py --csv ../mdcath_320_test_max_rmsd.csv --top 5

# Trajectories that you name:
python mdcath_export_for_pymol.py --csv ../mdcath_320_test_max_rmsd.csv \
  --traj-ids 1awcA00_320_0 1awcA00_320_3

# Then view one:
conda activate pymol
pymol ../pymol_export/1awcA00_320_0.pml
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Column positions of each backbone atom in the 18-column feature array.
ATOM_COLS = {"N": slice(15, 18), "CA": slice(9, 12), "C": slice(6, 9), "O": slice(12, 15)}

PML = """# PyMOL script for {traj_id}
# Largest CA RMSD = {rmsd:.2f} A between frame {i} and frame {j} (0-based).
load {pdb}, traj
dss traj
create frame_{i}, traj, {si}, 1
create frame_{j}, traj, {sj}, 1
dss frame_{i}
dss frame_{j}
hide everything
show cartoon
set cartoon_trace_atoms, 0
color grey70, traj
color marine, frame_{i}
color firebrick, frame_{j}
disable traj
orient
# Type "enable traj" and press play to see the whole trajectory.
# PyMOL state k+1 is trajectory frame k.
"""


def kabsch_rotation(mobile_ca, target_ca):
    """Return the rotation R so that (mobile - centroid) @ R fits the target."""
    a = mobile_ca - mobile_ca.mean(0)
    b = target_ca - target_ca.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def export(row, shard_dir, out_dir):
    traj_id, i, j = row["traj_id"], int(row["frame_i"]), int(row["frame_j"])
    with h5py.File(shard_dir / row["shard"], "r") as f:
        g = f["trajectories"][traj_id]
        feats = g["features"][:].astype(np.float64)
        resname = [r.decode() for r in g["residue_name"][:]]
        resid = g["residue_id"][:]
    # xyz has shape (n_frames, n_residues, 4 atoms, 3).
    xyz = np.stack([feats[:, :, c] for c in ATOM_COLS.values()], axis=2)
    target_ca = xyz[i, :, 1]
    lines = []
    for k in range(xyz.shape[0]):
        ca = xyz[k, :, 1]
        rot = kabsch_rotation(ca, target_ca)
        fitted = (xyz[k] - ca.mean(0)) @ rot       # centroid of frame_i CA goes to the origin
        lines.append(f"MODEL     {k + 1:4d}")
        serial = 1
        for r in range(fitted.shape[0]):
            for a, name in enumerate(ATOM_COLS):
                x, y, z = fitted[r, a]
                lines.append(f"ATOM  {serial:5d}  {name:<3s} {resname[r]:>3s} A{resid[r]:4d}    "
                             f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]:>2s}")
                serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    pdb = out_dir / f"{traj_id}.pdb"
    pdb.write_text("\n".join(lines) + "\n")
    (out_dir / f"{traj_id}.pml").write_text(PML.format(
        traj_id=traj_id, rmsd=float(row["max_rmsd_A"]), i=i, j=j, si=i + 1, sj=j + 1,
        pdb=pdb.resolve()))
    print(f"{traj_id}: {xyz.shape[0]} frames, {xyz.shape[1]} residues, "
          f"max RMSD {float(row['max_rmsd_A']):.2f} A between frames {i} and {j} -> {pdb}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--shards", type=Path,
                   default=Path("/mnt/researchdrive/CHTC/sheaf_protein_dynamics/mdcath"))
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: pymol_export next to the CSV")
    p.add_argument("--traj-ids", nargs="*", default=[])
    p.add_argument("--top", type=int, default=0, help="export the first N rows of the CSV")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    pick = pd.concat([df.head(args.top), df[df["traj_id"].isin(args.traj_ids)]])
    pick = pick.drop_duplicates("traj_id")
    unknown = set(args.traj_ids) - set(df["traj_id"])
    if unknown:
        print("not in the CSV:", sorted(unknown))
    out_dir = args.out_dir or args.csv.parent / "pymol_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in pick.iterrows():
        export(row, args.shards, out_dir)


if __name__ == "__main__":
    main()
