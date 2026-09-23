#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    p = argparse.ArgumentParser(description="Create a headerless HTCondor queue file from a packages TSV.")
    p.add_argument("--packages-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    df = pd.read_csv(args.packages_manifest, sep="\t")
    required = {"package_id", "package_tar"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["package_id"].duplicated().any():
        raise ValueError("Duplicate package_id values found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["package_id", "package_tar"]].to_csv(args.out, sep="\t", index=False, header=False)
    print(f"Wrote {len(df)} queue rows to {args.out}")
    print(args.out.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
