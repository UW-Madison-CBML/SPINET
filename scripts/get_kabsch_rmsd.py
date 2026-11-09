# get_kabsch_rmsd.py
"""Self-consistency RMSD of a table of designed sequences.

Each row of ``<seq_csv>.csv`` (columns ``pdb``, ``seq``) is folded with ESMFold and
Kabsch-superposed onto the protein's relaxed reference structure -- the same reference every
comparison model trains on (lib/relaxed_pdb.py):

* ATLAS ids (``1a0a_A``): the deposited RCSB chain.
* mdCATH ids (``12asA00``): the CATH domain file, from the extracted ``mdcath_pdbs.tar.gz``
  in ``--cath-dir``, used whole.

A design and its reference are NOT assumed to be positionally identical. Models design over
different residue frames -- the sheaf model and DynamicMPNN over the MD trajectory's residues,
MapDiff/PiFold over the relaxed structure's -- and a deposited chain routinely has unresolved
loops, tags, or (for mdCATH) residues outside the domain. So each design is paired with its
reference through its *native* sequence:

1. The design's native sequence is the csv's ``native`` column when it has one (e.g. a cropped
   MapDiff/PiFold ATLAS design), else, with ``--traj-h5``, the trajectory's residue list. If the
   design is exactly as long as that native sequence, the native is aligned to the reference's sequence
   (`relaxed_pdb.align_sequences`, identity pairs only) and only the aligned residues are
   superposed.
2. Otherwise, if the design is exactly as long as the reference, residue i pairs with i.
3. Otherwise the row is dropped and reported -- never silently truncated.

Writes ``<seq_csv>_scrmsd.csv`` (one row per scored design, with how it was paired and how many
residues were superposed) and prints the mean +- std.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import relaxed_pdb
from scrmsd import evaluate_scrmsd, load_esmfold

ATLAS_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}_[A-Za-z0-9]$")
CATH_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}[A-Za-z0-9][0-9]{2}$")

# Aligned residues below this fraction of the design mean the native and the reference are
# not the same protein region; the RMSD would be meaningless.
MIN_COVERAGE = 0.5


def load_references(pdb_ids, cath_dir, pdb_cache):
    """``{id: record}`` from the loaders the comparison models use, plus ``{id: reason}``."""
    atlas_ids = [p for p in pdb_ids if ATLAS_ID.match(p)]
    cath_ids = [p for p in pdb_ids if CATH_ID.match(p)]
    failures = {p: "id matches neither ATLAS (PPPP_C) nor mdCATH (PPPPCNN) format"
                for p in pdb_ids if p not in set(atlas_ids) | set(cath_ids)}
    records = {}
    if atlas_ids:
        found, failed = relaxed_pdb.load_relaxed_structures(atlas_ids, cache_dir=pdb_cache)
        records.update(found)
        failures.update(failed)
    if cath_ids:
        found, failed = relaxed_pdb.load_cath_structures(cath_ids, cath_dir)
        records.update(found)
        failures.update(failed)
    return records, failures


def pair_residues(design, reference_seq, native_seq):
    """``(design_idx, ref_idx, how)`` or ``(None, None, reason)`` -- see the module docstring."""
    if native_seq is not None and len(native_seq) == len(design):
        design_idx, ref_idx = relaxed_pdb.align_sequences(native_seq, reference_seq)
        if len(design_idx) < MIN_COVERAGE * len(design):
            return None, None, "native aligns to only {}/{} reference residues".format(
                len(design_idx), len(design))
        return design_idx, ref_idx, "native_alignment"
    if len(design) == len(reference_seq):
        idx = np.arange(len(design))
        return idx, idx, "identity"
    return None, None, "design length {} matches neither the reference ({}){}".format(
        len(design), len(reference_seq),
        "" if native_seq is None else " nor the trajectory ({})".format(len(native_seq)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seq_csv", help="designs table, with or without the .csv extension")
    parser.add_argument("--traj-h5", default=None,
                        help="trajectory store the designs' dataset came from; supplies the "
                             "native sequence for designs made over trajectory residues")
    parser.add_argument("--cath-dir", default=relaxed_pdb.DEFAULT_CATH_DIR,
                        help="extracted mdcath_pdbs.tar.gz (mdCATH references)")
    parser.add_argument("--pdb-cache", default="pdb_cache", help="RCSB download cache (ATLAS)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-pdbs", default="pred_pdbs",
                        help="directory for the ESMFold predictions ('' to skip)")
    args = parser.parse_args()

    stem = args.seq_csv[:-4] if args.seq_csv.endswith(".csv") else args.seq_csv
    seq_df = pd.read_csv(os.path.abspath(stem + ".csv"))
    seq_df["pdb"] = seq_df["pdb"].astype(str).str.strip()
    seq_df["seq"] = seq_df["seq"].astype(str).str.strip().str.upper()

    pdb_ids = sorted(seq_df["pdb"].unique())
    references, failures = load_references(pdb_ids, args.cath_dir, args.pdb_cache)
    natives = relaxed_pdb.reference_seqs_from_h5(args.traj_h5, pdb_ids) if args.traj_h5 else {}

    rows, dropped = [], []
    for row_idx, row in seq_df.iterrows():
        pdb_id, design = row["pdb"], row["seq"]
        if pdb_id not in references:
            dropped.append((pdb_id, failures.get(pdb_id, "no reference structure")))
            continue
        reference = references[pdb_id]
        native = natives.get(pdb_id)
        if "native" in seq_df.columns and isinstance(row["native"], str) and row["native"].strip():
            native = row["native"].strip().upper()
        design_idx, ref_idx, how = pair_residues(design, reference["seq"], native)
        if design_idx is None:
            dropped.append((pdb_id, how))
            continue
        rows.append({"row": row_idx, "pdb": pdb_id, "seq": design, "pairing": how,
                     "n_superposed": len(design_idx), "design_idx": design_idx,
                     "ref_idx": ref_idx, "coords": reference["coords"]})

    print("{} designs scorable, {} dropped".format(len(rows), len(dropped)))
    for pdb_id, reason in dropped:
        print("  dropped {}: {}".format(pdb_id, reason))
    if not rows:
        raise SystemExit("nothing to score")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_esmfold(device=device)
    if args.save_pdbs:
        os.makedirs(args.save_pdbs, exist_ok=True)

    rmsds = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="running eval"):
        batch = rows[start:start + args.batch_size]
        save_pdbs = ([os.path.abspath(os.path.join(args.save_pdbs, "{}_{}.pdb".format(r["pdb"], r["row"])))
                      for r in batch] if args.save_pdbs else None)
        rmsd = evaluate_scrmsd([r["seq"] for r in batch], [r["coords"] for r in batch],
                               tokenizer, model,
                               pred_indices=[r["design_idx"] for r in batch],
                               ref_indices=[r["ref_idx"] for r in batch],
                               device=device, save_pdbs=save_pdbs)
        rmsds.extend(rmsd.tolist())

    out = pd.DataFrame([{k: r[k] for k in ("row", "pdb", "seq", "pairing", "n_superposed")}
                        for r in rows])
    out["scrmsd"] = rmsds
    out.to_csv(stem + "_scrmsd.csv", index=False)
    rmsds = np.array(rmsds)
    print(out["pairing"].value_counts().to_string())
    print(f"${rmsds.mean().item():.3f} \\pm {rmsds.std().item():.3f}$")


if __name__ == "__main__":
    main()
