#!/usr/bin/env python3
"""Step 1: cut the reference window out of an mdCATH trajectory.

Input
-----
--shard     One of:
            * a repo shard, mdcath_shard_*.h5, with
              trajectories/<traj_id>/features of shape (n_frames, n_residues, 18);
            * an original mdCATH file, mdcath_dataset_<domain>.h5, with
              <domain>/<temperature>/<replica>/coords and <domain>/pdbProteinAtoms.
--traj-id   <domain>_<temperature>_<replica>, e.g. 3j7yg00_320_4.
--frame-start, --frame-end
            0-based frame indices. Both frames are part of the window, so a
            window of N frames covers (N - 1) x 1 ns. The start_idx/end_idx
            columns of the sheaf model's W&B table have an exclusive end:
            pass end_idx - 1.
--pred-seq  Sequence from the inverse folding model (1-letter codes).

Output (in --out-dir)
---------------------
ref_window.pdb   First frame of the window, backbone atoms N, CA, C, O.
ref_window.xtc   All frames of the window. Time is in ps from the window start.
seqs.fasta       Records "pred" and "native" for ColabFold.
prep.json        Window size, sequences, sequence identity, chain breaks.
"""
import argparse
import json
import tempfile
from pathlib import Path

import h5py
import mdtraj as md
import numpy as np

from common import (ATOM_COLS, BACKBONE, backbone_topology, chain_breaks, parse_traj_id,
                    sequence_of, to_backbone)


def read_shard(f, traj_id, start, end):
    g = f["trajectories"][traj_id]
    n_total = g["features"].shape[0]
    feats = g["features"][start:end + 1].astype(np.float64)
    resname = [r.decode() if isinstance(r, bytes) else str(r) for r in g["residue_name"][:]]
    resid = g["residue_id"][:]
    # (n_frames, n_residues, 4 atoms, 3) in Angstrom -> (n_frames, n_atoms, 3) in nm.
    xyz = np.stack([feats[:, :, ATOM_COLS[a]] for a in BACKBONE], axis=2)
    xyz = xyz.reshape(xyz.shape[0], -1, 3) / 10.0
    return md.Trajectory(xyz, backbone_topology(resname, resid)), n_total


def read_raw_mdcath(f, traj_id, start, end):
    domain, temp, rep = parse_traj_id(traj_id)
    g = f[domain]
    pdb_text = g["pdbProteinAtoms"][()]
    pdb_text = pdb_text.decode() if isinstance(pdb_text, bytes) else str(pdb_text)
    with tempfile.NamedTemporaryFile("w", suffix=".pdb") as tmp:
        tmp.write(pdb_text)
        tmp.flush()
        top = md.load(tmp.name).topology
    coords = g[f"{temp}/{rep}/coords"]
    n_total = coords.shape[0]
    xyz = coords[start:end + 1].astype(np.float64) / 10.0
    return to_backbone(md.Trajectory(xyz, top)), n_total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard", type=Path, required=True)
    p.add_argument("--traj-id", required=True)
    p.add_argument("--frame-start", type=int, required=True)
    p.add_argument("--frame-end", type=int, required=True)
    p.add_argument("--pred-seq", required=True)
    p.add_argument("--frame-interval-ns", type=float, default=1.0)
    p.add_argument("--allow-breaks", action="store_true",
                   help="continue when the domain has chain breaks")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    args = p.parse_args()

    if not 0 <= args.frame_start < args.frame_end:
        raise SystemExit("need 0 <= frame_start < frame_end")
    with h5py.File(args.shard, "r") as f:
        reader = read_shard if "trajectories" in f else read_raw_mdcath
        traj, n_total = reader(f, args.traj_id, args.frame_start, args.frame_end)
    if args.frame_end >= n_total:
        raise SystemExit(f"frame_end {args.frame_end} is outside the trajectory "
                         f"({n_total} frames)")

    native = sequence_of(traj.topology)
    pred = args.pred_seq.strip().upper()
    if len(pred) != len(native):
        raise SystemExit(f"predicted sequence has {len(pred)} residues, "
                         f"the trajectory has {len(native)}")
    bad = sorted(set(pred) - set("ACDEFGHIKLMNPQRSTVWY"))
    if bad:
        raise SystemExit(f"predicted sequence has non-standard letters: {bad}")

    breaks = chain_breaks(traj)
    if breaks and not args.allow_breaks:
        raise SystemExit(
            f"the domain has chain breaks after residue indices {breaks}. ColabFold would "
            "fold the sequence as one continuous chain. Use --allow-breaks to continue.")

    traj.time = np.arange(traj.n_frames) * args.frame_interval_ns * 1000.0  # ps
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    traj[0].save_pdb(str(out / "ref_window.pdb"))
    traj.save_xtc(str(out / "ref_window.xtc"))
    (out / "seqs.fasta").write_text(f">pred\n{pred}\n>native\n{native}\n")

    domain, temperature, replica = parse_traj_id(args.traj_id)
    info = dict(traj_id=args.traj_id, domain=domain, temperature_K=temperature,
                replica=replica, frame_start=args.frame_start, frame_end=args.frame_end,
                n_frames=traj.n_frames, n_residues=traj.n_residues,
                frame_interval_ns=args.frame_interval_ns,
                window_ns=(traj.n_frames - 1) * args.frame_interval_ns,
                native_seq=native, pred_seq=pred,
                sequence_identity=sum(a == b for a, b in zip(native, pred)) / len(native),
                chain_breaks_after_residue_index=breaks)
    (out / "prep.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
