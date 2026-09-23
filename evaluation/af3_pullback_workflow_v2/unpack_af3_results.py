#!/usr/bin/env python3
from __future__ import annotations

import argparse, tarfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Unpack AF3 result archives into one output tree.")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tarballs = sorted(args.results_dir.glob("af3_results_*.tar.gz"))
    if not tarballs:
        raise FileNotFoundError(f"No af3_results_*.tar.gz in {args.results_dir}")
    for i, tar_path in enumerate(tarballs, 1):
        marker = args.out / ".unpacked" / tar_path.stem
        if args.skip_existing and marker.exists():
            print(f"[{i}/{len(tarballs)}] skip {tar_path.name}")
            continue
        print(f"[{i}/{len(tarballs)}] {tar_path.name}")
        with tarfile.open(tar_path, "r:gz") as tf:
            members = []
            for member in tf.getmembers():
                parts = Path(member.name).parts
                if parts and parts[0] == "af3_output" and len(parts) > 1:
                    member.name = str(Path(*parts[1:]))
                    members.append(member)
            tf.extractall(args.out, members=members)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    print(f"Unpacked {len(tarballs)} archives into {args.out}")


if __name__ == "__main__":
    main()
