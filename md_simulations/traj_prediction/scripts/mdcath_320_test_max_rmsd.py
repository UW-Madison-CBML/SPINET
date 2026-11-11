#!/usr/bin/env python3
"""Find the two most different frames in each mdCATH 320 K test trajectory.

Input
-----
--split   CSV with columns domain,split. The script keeps rows with split == test.
--shards  Folder with mdcath_shard_*.h5. Each shard stores trajectories under
          trajectories/<domain>_<temperature>_<replicate>/features with shape
          (n_frames, n_residues, 18). Columns 9,10,11 are the CA x,y,z in Angstrom.

Output
------
One CSV row per trajectory, sorted by max_rmsd_A from largest to smallest.
frame_i and frame_j are 0-based frame indices with frame_i < frame_j.

Method
------
For each pair of frames the script superposes the CA atoms with the Kabsch
method (best rigid rotation plus translation) and computes the RMSD.
The script reports the pair with the largest RMSD.

Example
-------
python mdcath_320_test_max_rmsd.py \
  --split /mnt/researchdrive/CHTC/sheaf_protein_dynamics/mdcath_320_0_topo_split.csv \
  --shards /mnt/researchdrive/CHTC/sheaf_protein_dynamics/mdcath \
  --out ../mdcath_320_test_max_rmsd.csv
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CA_COLS = slice(9, 12)  # CA_x, CA_y, CA_z in the 18-column feature array
META_ATTRS = ["cath_full", "pdb_accession", "chain", "sequence_length", "resolution",
              "standard_split"]


def pairwise_rmsd(ca):
    """Return the (n_frames, n_frames) matrix of Kabsch RMSD values.

    ca has shape (n_frames, n_atoms, 3).
    """
    x = ca.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)          # move each frame to its centroid
    n_frames, n_atoms, _ = x.shape
    g = (x ** 2).sum(axis=(1, 2))                  # sum of squared coordinates per frame
    out = np.zeros((n_frames, n_frames))
    for i in range(n_frames - 1):
        # 3x3 covariance between frame i and each later frame.
        h = np.einsum("ak,jal->jkl", x[i], x[i + 1:])
        s = np.linalg.svd(h, compute_uv=False)
        # A negative determinant means the best fit is a mirror image.
        # The sign change on the smallest singular value forbids the mirror image.
        s[:, 2] *= np.sign(np.linalg.det(h))
        msd = (g[i] + g[i + 1:] - 2.0 * s.sum(axis=1)) / n_atoms
        out[i, i + 1:] = np.sqrt(np.clip(msd, 0.0, None))
    return out + out.T


def process_shard(args):
    """Run read_shard. Try again when the network drive gives a read error."""
    for attempt in range(5):
        try:
            return read_shard(args)
        except (OSError, KeyError) as e:
            print(f"read error on {Path(args[0]).name}, attempt {attempt + 1}: {e}", flush=True)
            time.sleep(30)
    return read_shard(args)


def read_shard(args):
    shard, test_domains, temperature = args
    rows = []
    with h5py.File(shard, "r") as f:
        trajs = f["trajectories"]
        for traj_id in trajs:
            domain, temp, rep = traj_id.split("_")
            if temp != temperature or domain not in test_domains:
                continue
            g = trajs[traj_id]
            ca = g["features"][:, :, CA_COLS]
            d = pairwise_rmsd(ca)
            i, j = np.unravel_index(np.argmax(d), d.shape)
            i, j = sorted((int(i), int(j)))
            # Distance between neighbour CA atoms. A change of several Angstrom
            # between frames shows that the periodic box split the chain.
            bond = np.linalg.norm(np.diff(ca, axis=1), axis=2)
            row = dict(domain=domain, traj_id=traj_id, temperature=int(temp),
                       replicate=int(rep), n_frames=ca.shape[0], n_residues=ca.shape[1],
                       max_rmsd_A=round(float(d[i, j]), 3), frame_i=i, frame_j=j,
                       mean_pairwise_rmsd_A=round(float(d[np.triu_indices_from(d, 1)].mean()), 3),
                       max_ca_bond_change_A=round(float((bond.max(0) - bond.min(0)).max()), 3),
                       shard=Path(shard).name)
            for a in META_ATTRS:
                row[a] = g.attrs.get(a, "")
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--shards", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--temperature", default="320")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()

    split = pd.read_csv(args.split, dtype=str)
    test_domains = set(split.loc[split["split"] == "test", "domain"])
    shards = sorted(args.shards.glob("mdcath_shard_*.h5"))
    print(f"{len(test_domains)} test domains, {len(shards)} shards", flush=True)

    rows = []
    partial = args.out.with_suffix(".partial.csv")
    if partial.exists():
        # Resume: keep the rows of an interrupted run and skip their shards.
        # The script writes the rows of a shard only after it finished the shard.
        old = pd.read_csv(partial)
        rows = old.to_dict("records")
        shards = [s for s in shards if s.name not in set(old["shard"])]
        print(f"resume: {len(rows)} rows kept, {len(shards)} shards left", flush=True)
    jobs = [(str(s), test_domains, args.temperature) for s in shards]
    with Pool(args.workers) as pool:
        for k, part in enumerate(pool.imap_unordered(process_shard, jobs), 1):
            rows.extend(part)
            print(f"shard {k}/{len(shards)} done, {len(rows)} trajectories", flush=True)
            # Save partial results so that an interrupted run keeps its work.
            pd.DataFrame(rows).to_csv(partial, index=False)

    df = pd.DataFrame(rows).sort_values("max_rmsd_A", ascending=False)
    df.to_csv(args.out, index=False)
    partial.unlink(missing_ok=True)
    missing = sorted(test_domains - set(df["domain"]))
    print(f"wrote {len(df)} rows for {df['domain'].nunique()} domains to {args.out}")
    print(f"{len(missing)} test domains not found in the shards: {missing[:20]}")


if __name__ == "__main__":
    main()
