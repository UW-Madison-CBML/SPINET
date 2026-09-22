#!/usr/bin/env python3
"""
Export a minimal AF3 pullback example from a raw mdCATH HDF5 file + a CSV of
predicted sequences.

This version deliberately parses chain/residue identity from `pdbProteinAtoms`
instead of assuming that the HDF5 `/chain` dataset stores PDB chain labels.

Outputs:
    <out>/<domain>/
        states/state_00.pdb
        states/state_01.pdb
        sequences/native.fasta
        sequences/spinet.fasta
        sequences/all_predicted_sequences.fasta
        export_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HID": "H", "HIE": "H", "HIP": "H",
    "HSD": "H", "HSE": "H", "HSP": "H",
    "CYX": "C", "CYM": "C",
    "MSE": "M",
}


RESNAME_MAP = {
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
}


def decode_pdb_protein_atoms(dataset) -> str:
    """
    mdCATH files may store pdbProteinAtoms either as one scalar string/bytes
    object or as an array of PDB lines. Normalize both representations.
    """
    value = dataset[()]

    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, str):
        return value

    arr = np.asarray(value)

    if arr.shape == ():
        scalar = arr.item()
        return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)

    lines = []
    for item in arr.reshape(-1):
        if isinstance(item, bytes):
            lines.append(item.decode("utf-8").rstrip("\n"))
        else:
            lines.append(str(item).rstrip("\n"))
    return "\n".join(lines) + "\n"


def pdb_atom_records(pdb_text: str) -> list[dict]:
    """
    Parse ATOM/HETATM records in the exact order present in pdbProteinAtoms.
    That order is the atom axis used by mdCATH `coords`.
    """
    records = []

    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue

        padded = line.ljust(80)

        try:
            resseq = int(padded[22:26])
        except ValueError:
            continue

        records.append(
            {
                "line": line,
                "atom_name": padded[12:16].strip(),
                "altloc": padded[16].strip(),
                "resname": padded[17:20].strip().upper(),
                "chain": padded[21].strip(),
                "resseq": resseq,
                "icode": padded[26].strip(),
            }
        )

    if not records:
        raise ValueError("No ATOM/HETATM records found in pdbProteinAtoms")

    return records


def choose_chain(records: list[dict], requested_chain: str | None) -> str:
    """
    Resolve the chain to use from actual PDB records.

    If the requested chain label is absent but the structure contains exactly one
    protein chain, use that single chain automatically. This handles mdCATH/PDB
    records whose chain column is blank.
    """
    protein_chains = []
    for rec in records:
        if rec["resname"] not in AA3_TO_1:
            continue
        if rec["chain"] not in protein_chains:
            protein_chains.append(rec["chain"])

    if not protein_chains:
        raise ValueError("No recognizable protein chains found in pdbProteinAtoms")

    if requested_chain is not None and requested_chain in protein_chains:
        return requested_chain

    if len(protein_chains) == 1:
        actual = protein_chains[0]
        label = actual if actual else "<blank>"
        if requested_chain is not None:
            print(
                f"Requested chain '{requested_chain}' is not present in pdbProteinAtoms; "
                f"using the only protein chain {label} instead."
            )
        return actual

    available = [chain if chain else "<blank>" for chain in protein_chains]
    raise ValueError(
        f"Requested chain '{requested_chain}' not found and multiple protein chains exist: "
        f"{available}. Pass --chain explicitly using the PDB chain label."
    )


def native_sequence_from_pdb_records(
    records: list[dict],
    chain_id: str,
) -> tuple[str, list[dict]]:
    """
    Build the native sequence from unique PDB residues in the selected chain.
    """
    residues = []
    seen = set()

    for rec in records:
        if rec["chain"] != chain_id:
            continue
        if rec["resname"] not in AA3_TO_1:
            continue

        key = (rec["chain"], rec["resseq"], rec["icode"])
        if key in seen:
            continue
        seen.add(key)

        residues.append(
            {
                "chain": rec["chain"],
                "resid": rec["resseq"],
                "insertion_code": rec["icode"],
                "resname": rec["resname"],
                "aa": AA3_TO_1[rec["resname"]],
            }
        )

    if not residues:
        label = chain_id if chain_id else "<blank>"
        raise ValueError(f"No protein residues found for selected chain {label}")

    return "".join(r["aa"] for r in residues), residues


def write_frame_pdb(
    records: list[dict],
    frame_coords_angstrom: np.ndarray,
    selected_chain: str,
    output_path: Path,
) -> int:
    """
    Replace coordinates in pdbProteinAtoms with one MD frame, keeping only the
    selected protein chain.
    """
    if len(records) != frame_coords_angstrom.shape[0]:
        raise ValueError(
            "pdbProteinAtoms / coords atom-count mismatch: "
            f"{len(records)} PDB atom records vs "
            f"{frame_coords_angstrom.shape[0]} coordinate rows"
        )

    out_lines = []
    kept_atoms = 0

    for atom_idx, rec in enumerate(records):
        if rec["chain"] != selected_chain:
            continue
        if rec["resname"] not in AA3_TO_1:
            continue

        line = rec["line"].ljust(80)
        x, y, z = frame_coords_angstrom[atom_idx]

        # Normalize MD force-field residue names to canonical PDB residue names.
        resname = RESNAME_MAP.get(rec["resname"], rec["resname"])
        line = line[:17] + f"{resname:>3}" + line[20:]

        # Replace coordinates with the selected MD frame.
        line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]

        out_lines.append(line.rstrip())
        kept_atoms += 1

    if kept_atoms == 0:
        raise ValueError("No protein atoms were selected for state export")

    out_lines.extend(["TER", "END"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return kept_atoms


def infer_chain_id(domain: str) -> str | None:
    # CATH domain naming convention: 4-character PDB accession + chain + domain no.
    return domain[4] if len(domain) >= 5 else None


def read_predicted_sequences(csv_path: Path, domain: str) -> list[str]:
    df = pd.read_csv(csv_path)
    required = {"pdb", "seq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    rows = df.loc[df["pdb"].astype(str) == domain, "seq"]
    sequences = [str(seq).strip().upper() for seq in rows if pd.notna(seq)]

    if not sequences:
        raise ValueError(f"No predicted sequences found for '{domain}' in {csv_path}")

    return sequences


def write_fasta(path: Path, name: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">{name}\n{sequence}\n", encoding="utf-8")


def parse_frame_spec(spec: str | None, n_frames: int) -> list[int]:
    if spec is None:
        indices = [0, n_frames // 2]
    else:
        indices = [int(token.strip()) for token in spec.split(",") if token.strip()]

    if not indices:
        raise ValueError("At least one frame index is required")

    resolved = []
    for idx in indices:
        if idx < 0:
            idx = n_frames + idx
        if idx < 0 or idx >= n_frames:
            raise IndexError(
                f"Frame index {idx} is outside trajectory range [0, {n_frames - 1}]"
            )
        resolved.append(idx)

    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate frame indices after resolution: {resolved}")

    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--domain", default="12asA00")
    parser.add_argument(
        "--chain",
        default=None,
        help=(
            "PDB chain label. If omitted, infer from the CATH domain name; "
            "if that label is absent but only one protein chain exists, use it."
        ),
    )
    parser.add_argument("--temperature", default="320")
    parser.add_argument("--replicate", default="0")
    parser.add_argument("--frames", default=None)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("pullback_example"))
    args = parser.parse_args()

    predicted_sequences = read_predicted_sequences(args.csv, args.domain)
    if args.sequence_index < 0 or args.sequence_index >= len(predicted_sequences):
        raise IndexError(
            f"--sequence-index={args.sequence_index}, but {args.domain} has "
            f"{len(predicted_sequences)} predicted sequences"
        )
    predicted_sequence = predicted_sequences[args.sequence_index]

    with h5py.File(args.h5, "r") as h5:
        if args.domain not in h5:
            raise KeyError(
                f"Domain '{args.domain}' not found. First root keys: {list(h5.keys())[:10]}"
            )

        domain_group = h5[args.domain]

        if "pdbProteinAtoms" not in domain_group:
            raise KeyError(f"/{args.domain}/pdbProteinAtoms is missing")

        pdb_text = decode_pdb_protein_atoms(domain_group["pdbProteinAtoms"])
        atom_records = pdb_atom_records(pdb_text)

        requested_chain = args.chain if args.chain is not None else infer_chain_id(args.domain)
        selected_chain = choose_chain(atom_records, requested_chain)

        native_sequence, residue_records = native_sequence_from_pdb_records(
            atom_records, selected_chain
        )

        trajectory_path = f"{args.temperature}/{args.replicate}"
        if trajectory_path not in domain_group:
            available = [
                key for key in domain_group.keys()
                if isinstance(domain_group.get(key, None), h5py.Group)
            ]
            raise KeyError(
                f"/{args.domain}/{trajectory_path} not found. "
                f"Available top-level groups: {available}"
            )

        traj_group = domain_group[trajectory_path]
        if "coords" not in traj_group:
            raise KeyError(f"/{args.domain}/{trajectory_path}/coords is missing")

        coords_ds = traj_group["coords"]
        if coords_ds.ndim != 3 or coords_ds.shape[-1] != 3:
            raise ValueError(
                f"Unexpected coords shape {coords_ds.shape}; expected (frames, atoms, 3)"
            )

        n_frames, n_atoms, _ = coords_ds.shape
        if n_atoms != len(atom_records):
            raise ValueError(
                f"coords has {n_atoms} atoms but pdbProteinAtoms has "
                f"{len(atom_records)} atom records"
            )

        frame_indices = parse_frame_spec(args.frames, n_frames)

        example_dir = args.out / args.domain
        states_dir = example_dir / "states"
        seq_dir = example_dir / "sequences"

        state_records = []
        for state_idx, frame_idx in enumerate(frame_indices):
            state_id = f"state_{state_idx:02d}"
            output_pdb = states_dir / f"{state_id}.pdb"

            frame_coords = np.asarray(coords_ds[frame_idx], dtype=np.float64)

            kept_atoms = write_frame_pdb(
                records=atom_records,
                frame_coords_angstrom=frame_coords,
                selected_chain=selected_chain,
                output_path=output_pdb,
            )

            state_records.append(
                {
                    "state_id": state_id,
                    "frame_index": int(frame_idx),
                    "temperature": str(args.temperature),
                    "replicate": str(args.replicate),
                    "pdb_path": str(output_pdb),
                    "n_atoms": int(kept_atoms),
                }
            )

    if len(predicted_sequence) != len(native_sequence):
        raise ValueError(
            f"Sequence-length mismatch for {args.domain}: "
            f"native={len(native_sequence)}, predicted={len(predicted_sequence)}. "
            "Do not truncate automatically; inspect whether your model output uses "
            "a domain/alignment mask."
        )

    example_dir = args.out / args.domain
    seq_dir = example_dir / "sequences"

    write_fasta(
        seq_dir / "native.fasta",
        f"{args.domain}__native",
        native_sequence,
    )
    write_fasta(
        seq_dir / "spinet.fasta",
        f"{args.domain}__spinet_{args.sequence_index:02d}",
        predicted_sequence,
    )

    (seq_dir / "all_predicted_sequences.fasta").write_text(
        "".join(
            f">{args.domain}__pred_{i:02d}\n{seq}\n"
            for i, seq in enumerate(predicted_sequences)
        ),
        encoding="utf-8",
    )

    manifest = {
        "domain": args.domain,
        "requested_chain": requested_chain,
        "selected_pdb_chain": selected_chain,
        "source_h5": str(args.h5),
        "source_csv": str(args.csv),
        "temperature": str(args.temperature),
        "replicate": str(args.replicate),
        "native_sequence_length": len(native_sequence),
        "predicted_sequence_count": len(predicted_sequences),
        "selected_prediction_index": args.sequence_index,
        "native_fasta": str(seq_dir / "native.fasta"),
        "predicted_fasta": str(seq_dir / "spinet.fasta"),
        "states": state_records,
        "residues": residue_records,
    }

    manifest_path = example_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    selected_display = selected_chain if selected_chain else "<blank>"
    print(f"Exported {args.domain}")
    print(f"  requested chain:  {requested_chain}")
    print(f"  selected PDB chain:{selected_display}")
    print(f"  native length:    {len(native_sequence)}")
    print(f"  predictions:      {len(predicted_sequences)}")
    print(f"  selected design:  {args.sequence_index}")
    print(f"  trajectory:       {args.temperature} K / replicate {args.replicate}")
    print(f"  frames:           {frame_indices}")
    print(f"  output:           {example_dir}")


if __name__ == "__main__":
    main()
