from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamicmpnn.eval.af3 import write_af3_jsons
from dynamicmpnn.eval.inputs import PreparedState, parse_structure
from dynamicmpnn.eval.templates import _artifact_from_state
from dynamicmpnn.eval.types import AF3JobSpec, AF3ProteinSpec


def read_fasta(path: Path) -> str:
    lines = path.read_text().splitlines()
    return "".join(
        line.strip()
        for line in lines
        if line.strip() and not line.startswith(">")
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--example-dir",
        type=Path,
        required=True,
        help="Directory produced by export_states.py, e.g. pullback_example/12asA00",
    )

    parser.add_argument(
        "--state",
        required=True,
        help="State name, e.g. state_00",
    )

    parser.add_argument(
        "--sequence",
        choices=["native", "spinet"],
        default="native",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    example_dir = args.example_dir
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Read export metadata
    # --------------------------------------------------

    manifest_path = example_dir / "export_manifest.json"

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    chain_id = str(manifest["selected_pdb_chain"])

    # --------------------------------------------------
    # Input structure
    # --------------------------------------------------

    state_name = args.state

    pdb_path = (
        example_dir
        / "states"
        / f"{state_name}.pdb"
    )

    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    # --------------------------------------------------
    # Input sequence
    # --------------------------------------------------

    fasta_path = (
        example_dir
        / "sequences"
        / f"{args.sequence}.fasta"
    )

    sequence = read_fasta(fasta_path)

    print(f"State:          {state_name}")
    print(f"Sequence type:  {args.sequence}")
    print(f"Sequence length:{len(sequence)}")
    print(f"PDB chain:      {chain_id}")

    # --------------------------------------------------
    # Let DynamicMPNN parse the structure
    # --------------------------------------------------

    parsed = parse_structure(
        name=state_name,
        pdb_path=pdb_path,
        chain_id=chain_id,
    )

    target_chain = parsed.target_chain

    structure_length = len(target_chain.sequence)

    print(f"Structure length: {structure_length}")

    if structure_length != len(sequence):
        raise ValueError(
            f"Sequence length {len(sequence)} != "
            f"structure length {structure_length}"
        )

    # Complete mdCATH state:
    #
    # query residue i <-> template residue i
    #
    mapping = tuple(range(len(sequence)))

    # --------------------------------------------------
    # Minimal PreparedState adapter
    # --------------------------------------------------

    prepared_state = PreparedState(
        name=state_name,
        pdb_path=pdb_path,
        chain_id=chain_id,
        sequence=target_chain.sequence,
        aligned_sequence=target_chain.sequence,
        residues_3=target_chain.residues_3,
        residue_numbers=target_chain.residue_numbers,
        residue_insertions=target_chain.residue_insertions,
        coords=target_chain.coords,
        query_indices=mapping,
        template_indices=mapping,
        context_chains=parsed.context_chains,
        pyg_data=None,
    )

    # --------------------------------------------------
    # DynamicMPNN writes AF3-compatible mmCIF
    # --------------------------------------------------

    artifact = _artifact_from_state(
        prepared_state,
        structure_kind="target",
        output_dir=out_dir,
    )

    print(f"Template CIF: {artifact.mmcif_path}")
    print(f"Template PDB: {artifact.pdb_path}")

    # --------------------------------------------------
    # Build AF3 job
    #
    # IMPORTANT:
    # use only basename in JSON.
    # On CHTC the JSON + CIF will live in the same scratch dir.
    # --------------------------------------------------

    sequence_id = args.sequence

    job_name = (
        f"{sequence_id}__"
        f"{state_name}__target"
    )

    protein = AF3ProteinSpec(
        entity_id="A",
        sequence=sequence,

        # relative path for CHTC portability
        template_mmcif=Path(artifact.mmcif_path.name),

        query_indices=mapping,
        template_indices=mapping,

        # our pullback protocol does not use MSA
        use_msa=False,
    )

    job = AF3JobSpec(
        job_name=job_name,
        sequence_id=sequence_id,
        state_name=state_name,
        structure_kind="target",

        input_dir=out_dir,
        json_path=out_dir / f"{job_name}.json",

        # actual CHTC output location is controlled
        # by run_alphafold.py
        output_dir=Path("af3_output") / job_name,

        proteins=(protein,),
        model_seeds=(1,),
        predicted_target_chain_id="A",
    )

    write_af3_jsons((job,))

    # --------------------------------------------------
    # Save a tiny adapter manifest
    # --------------------------------------------------

    prep_manifest = {
        "domain": manifest.get("domain", example_dir.name),
        "state_name": state_name,
        "sequence_id": sequence_id,
        "sequence_length": len(sequence),
        "source_pdb": str(pdb_path),
        "source_chain": chain_id,
        "template_cif": artifact.mmcif_path.name,
        "af3_json": job.json_path.name,
        "query_indices": list(mapping),
        "template_indices": list(mapping),
    }

    (
        out_dir / "prep_manifest.json"
    ).write_text(
        json.dumps(prep_manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("Prepared AF3 pullback job:")
    print(f"  JSON:     {job.json_path}")
    print(f"  template: {artifact.mmcif_path}")
    print(f"  residues: {len(mapping)}")


if __name__ == "__main__":
    main()