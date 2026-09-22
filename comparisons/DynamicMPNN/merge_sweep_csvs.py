"""Merge chunked `sweep_k.py` result CSVs into one recovery-vs-k table.

`sweep_k.py` rewrites its results CSV with `w` after every completed point, so a grid split
across several condor jobs (see sweep_k.sub) comes home as one CSV per chunk. This stitches
them back into the single table the recovery-vs-k plot reads, sorted by k.

Chunks may carry different column sets -- a later chunk that OOMs before logging some metric
just leaves those cells empty -- so the union of columns is kept and missing cells are blank.
A k appearing in more than one chunk is an error rather than a silent last-writer-wins: two
runs of the same k are two different trainings, and picking between them is not this
script's call.

    python merge_sweep_csvs.py sweep_k_atlas.csv sweep_k_atlas_k2_20.csv sweep_k_atlas_k24_32.csv
"""
import csv
import sys
from pathlib import Path


def main(argv):
    if len(argv) < 3:
        sys.exit(f"usage: {Path(argv[0]).name} <out.csv> <chunk.csv> [<chunk.csv> ...]")
    out_path, chunk_paths = Path(argv[1]), [Path(p) for p in argv[2:]]

    columns, rows_by_k, source_of_k = [], {}, {}
    for chunk in chunk_paths:
        with open(chunk, newline="") as handle:
            for row in csv.DictReader(handle):
                k = int(row["k"])
                if k in rows_by_k:
                    sys.exit(f"k={k} appears in both {source_of_k[k]} and {chunk}; "
                             f"drop one before merging.")
                rows_by_k[k], source_of_k[k] = row, chunk
                columns.extend(c for c in row if c not in columns)

    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows_by_k[k] for k in sorted(rows_by_k))

    print(f"Merged {len(chunk_paths)} chunk(s) -> {out_path}: "
          f"k = {sorted(rows_by_k)} ({len(columns)} columns)")


if __name__ == "__main__":
    main(sys.argv)
