#!/usr/bin/env python3
"""
batch_prep_pdb_for_gmx.py

For a DIRECTORY of unrelated .cif structures (e.g. a motion-classification
dataset spanning many different proteins), this does the sane thing:
survey the whole set first, decide policy once, then apply it -- instead
of eyeballing 52 files one at a time and getting inconsistent about it.

Two phases:

  1. survey   - walk every .cif in a directory, report:
                  - every non-standard-AA residue code seen, how many
                    files/copies it appears in
                  - the longest missing-residue gap per file (X-ray gaps
                    are usually short; this tells you which files are the
                    exception and need eyeballing before you trust any
                    auto-fill)
                  - files that failed to parse at all (so one malformed
                    entry doesn't quietly poison the whole batch)
                Writes a CSV you actually read before doing anything else.

  2. apply    - given a policy file (JSON: {"keep": [...], "strip": [...]})
                built from what you learned in step 1, fixes missing
                residues/atoms per file and writes a cleaned structure,
                logging failures per-file instead of dying mid-batch.

Install:
    pip install pdbfixer openmm biopython

Usage:
    # Phase 1 - look before you leap
    python batch_prep_pdb_for_gmx.py survey ./cif_dataset --out survey.csv

    # write policy.json by hand based on survey.csv, e.g.:
    #   {"keep": ["ZN", "MG"], "strip": ["GOL", "PEG", "SO4", "ACT", "EDO"]}
    # anything NOT listed in either bucket is left for manual review and
    # excluded from the cleaned output by default (see --keep-unlisted).

    # Phase 2 - apply the policy across the whole directory
    python batch_prep_pdb_for_gmx.py apply ./cif_dataset --policy policy.json \
        --outdir ./cleaned --gap-flag-len 8
"""
from tqdm import tqdm
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HSD", "HSE", "HSP", "HID", "HIE", "HIP",
}
WATER_RESN = {"HOH", "WAT"}


def hetero_codes_in_file(cif_path):
    """All non-standard-AA residue codes present, deduped."""
    from Bio.PDB import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", str(cif_path))
    codes = set()
    model = next(iter(structure))
    for chain in model:
        for res in chain:
            resn = res.resname.strip()
            if resn not in STANDARD_AA:
                codes.add(resn)
    return codes


def missing_residue_gaps(cif_path):
    """Longest contiguous missing-residue run, using PDBFixer's mmCIF reader."""
    from pdbfixer import PDBFixer
    fixer = PDBFixer(pdbxfile=str(cif_path))
    fixer.findMissingResidues()
    if not fixer.missingResidues:
        return 0, 0
    lengths = [len(v) for v in fixer.missingResidues.values()]
    return max(lengths), sum(lengths)


def cmd_survey(args):
    cif_dir = Path(args.directory)
    files = sorted(cif_dir.glob("*.cif"))
    if not files:
        print(f"No .cif files found in {cif_dir}", file=sys.stderr)
        sys.exit(1)

    code_to_files = defaultdict(set)
    rows = []
    failures = []

    for f in tqdm(files):
        try:
            codes = hetero_codes_in_file(f)
        except Exception as e:
            failures.append((f.name, f"hetero scan failed: {e}"))
            codes = set()

        try:
            max_gap, total_missing = missing_residue_gaps(f)
        except Exception as e:
            failures.append((f.name, f"gap scan failed: {e}"))
            max_gap, total_missing = -1, -1

        for c in codes:
            code_to_files[c].add(f.name)

        rows.append({
            "file": f.name,
            "max_gap_len": max_gap,
            "total_missing_residues": total_missing,
            "hetero_codes": ";".join(sorted(codes - WATER_RESN)),
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "max_gap_len",
                                            "total_missing_residues", "hetero_codes"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== Survey of {len(files)} files ===")
    print(f"Wrote per-file detail -> {args.out}\n")

    print("Non-standard residue codes across the WHOLE dataset "
          "(decide keep/strip ONCE per code, not per file):\n")
    for code, fnames in sorted(code_to_files.items(), key=lambda kv: -len(kv[1])):
        tag = "water" if code in WATER_RESN else ""
        print(f"  {code:6s} {tag:6s} in {len(fnames)}/{len(files)} files")

    long_gap_files = [r for r in rows if r["max_gap_len"] not in (-1,) and r["max_gap_len"] >= args.gap_flag_len]
    print(f"\nFiles with a missing-residue run >= {args.gap_flag_len} residues "
          f"(eyeball these before trusting any auto-fill): {len(long_gap_files)}")
    for r in long_gap_files:
        print(f"  {r['file']}  max_gap={r['max_gap_len']}")

    if failures:
        print(f"\n[!] {len(failures)} files had parse problems - fix these by hand, "
              f"don't let them silently drop out of your dataset:")
        for name, reason in failures:
            print(f"  {name}: {reason}")


def cmd_apply(args):
    cif_dir = Path(args.directory)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    policy = json.loads(Path(args.policy).read_text())
    keep = set(policy.get("keep", [])) | STANDARD_AA | WATER_RESN
    strip = set(policy.get("strip", []))

    files = sorted(cif_dir.glob("*.cif"))
    log = []

    for f in files:
        try:
            from pdbfixer import PDBFixer
            from openmm.app import PDBFile

            fixer = PDBFixer(pdbxfile=str(f))
            fixer.findMissingResidues()
            gaps = fixer.missingResidues
            if gaps:
                max_gap = max(len(v) for v in gaps.values())
                if max_gap >= args.gap_flag_len:
                    log.append((f.name, "WARNING",
                                f"max missing-residue gap = {max_gap}; "
                                f"auto-fill is geometric guesswork above "
                                f"~{args.gap_flag_len} residues, verify by hand"))
            fixer.findNonstandardResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()

            fixed_path = outdir / f"{f.stem}_fixed.pdb"
            PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_path, "w"))

            # second pass: strip per policy using Biopython on the fixed file
            from Bio.PDB import PDBParser, PDBIO, Select
            structure = PDBParser(QUIET=True).get_structure("s", str(fixed_path))

            unlisted = set()

            class PolicySelect(Select):
                def accept_residue(self, res):
                    resn = res.resname.strip()
                    if resn in STANDARD_AA or resn in WATER_RESN:
                        return True
                    if resn in keep:
                        return True
                    if resn in strip:
                        return False
                    unlisted.add(resn)
                    return args.keep_unlisted

            final_path = outdir / f"{f.stem}_cleaned.pdb"
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(final_path), PolicySelect())

            if unlisted:
                log.append((f.name, "UNLISTED",
                            f"codes not in policy.json, "
                            f"{'kept' if args.keep_unlisted else 'dropped'} by default: "
                            f"{sorted(unlisted)}"))

            log.append((f.name, "OK", f"-> {final_path.name}"))

        except Exception as e:
            log.append((f.name, "FAILED", str(e)))

    print(f"\n=== Applied policy to {len(files)} files ===")
    for name, status, msg in log:
        print(f"[{status:8s}] {name}: {msg}")

    n_failed = sum(1 for _, s, _ in log if s == "FAILED")
    n_warned = sum(1 for _, s, _ in log if s in ("WARNING", "UNLISTED"))
    print(f"\n{len(files) - n_failed}/{len(files)} succeeded, "
          f"{n_failed} failed outright, {n_warned} flagged for review.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("survey", help="Survey a directory of .cif files before touching anything")
    sp.add_argument("directory")
    sp.add_argument("--out", default="survey.csv")
    sp.add_argument("--gap-flag-len", type=int, default=8)
    sp.set_defaults(func=cmd_survey)

    ap_apply = sub.add_parser("apply", help="Fix gaps + apply a keep/strip policy across the directory")
    ap_apply.add_argument("directory")
    ap_apply.add_argument("--policy", required=True, help="JSON: {\"keep\": [...], \"strip\": [...]}")
    ap_apply.add_argument("--outdir", default="./cleaned")
    ap_apply.add_argument("--gap-flag-len", type=int, default=8)
    ap_apply.add_argument("--keep-unlisted", action="store_true",
                           help="Keep residue codes not mentioned in policy.json "
                                "(default: drop them, but log every one)")
    ap_apply.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    sys.exit(main())
