from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dynamicmpnn.eval.af3 import AF3RunResult
from dynamicmpnn.eval.aggregation import build_all_af3_results_dataframe
from dynamicmpnn.eval.inputs import parse_structure
from dynamicmpnn.eval.sampling import SampleRecord
from dynamicmpnn.eval.scoring import score_jobs
from dynamicmpnn.eval.templates import TemplateArtifact
from dynamicmpnn.eval.types import AF3JobSpec, AF3ProteinSpec


def load_prepared_job(
    prepared_dir: Path,
    af3_output_dir: Path,
):
    """
    Reconstruct the minimal DynamicMPNN objects needed by score_jobs()
    from files produced by prepare_pullback_with_dynamicmpnn.py.
    """

    # --------------------------------------------------
    # 1. Read preparation manifest
    # --------------------------------------------------

    manifest_path = prepared_dir / "prep_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing prep manifest: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    state_name = manifest["state_name"]
    sequence_id = manifest["sequence_id"]
    source_chain = str(manifest["source_chain"])

    query_indices = tuple(
        int(x) for x in manifest["query_indices"]
    )
    template_indices = tuple(
        int(x) for x in manifest["template_indices"]
    )

    # --------------------------------------------------
    # 2. Read AF3 input JSON
    # --------------------------------------------------

    json_path = prepared_dir / manifest["af3_json"]

    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing AF3 JSON: {json_path}"
        )

    af3_payload = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    job_name = af3_payload["name"]

    protein_payload = af3_payload["sequences"][0]["protein"]

    sequence = protein_payload["sequence"]

    model_seeds = tuple(
        int(seed)
        for seed in af3_payload["modelSeeds"]
    )

    # --------------------------------------------------
    # 3. Locate target template
    # --------------------------------------------------

    template_cif = prepared_dir / manifest["template_cif"]

    # prepare_pullback_with_dynamicmpnn.py creates both
    # state_00_target.cif and state_00_target.pdb
    template_pdb = template_cif.with_suffix(".pdb")

    if not template_cif.exists():
        raise FileNotFoundError(
            f"Missing template CIF: {template_cif}"
        )

    if not template_pdb.exists():
        raise FileNotFoundError(
            f"Missing template PDB: {template_pdb}"
        )

    # --------------------------------------------------
    # 4. Let DynamicMPNN parse the reference structure
    # --------------------------------------------------

    parsed = parse_structure(
        name=state_name,
        pdb_path=template_pdb,
        chain_id=source_chain,
    )

    target_chain = parsed.target_chain

    if len(target_chain.sequence) != len(sequence):
        raise ValueError(
            "Reference/query length mismatch: "
            f"template={len(target_chain.sequence)}, "
            f"query={len(sequence)}"
        )

    # DynamicMPNN ChainRecord.coords is:
    #
    #   (n_residues, 3, 3)
    #
    # with atoms:
    #   N, CA, C
    #
    # score_jobs() expects TemplateArtifact.ca_coords
    # to contain one CA coordinate per residue.
    ca_coords = tuple(
        tuple(float(value) for value in xyz)
        for xyz in target_chain.coords[:, 1, :]
    )

    # --------------------------------------------------
    # 5. Reconstruct TemplateArtifact
    # --------------------------------------------------

    template = TemplateArtifact(
        state_name=state_name,
        structure_kind="target",
        pdb_path=template_pdb,
        mmcif_path=template_cif,
        query_indices=query_indices,
        template_indices=template_indices,
        ca_coords=ca_coords,
    )

    templates = {
        (state_name, "target"): template
    }

    # --------------------------------------------------
    # 6. Reconstruct AF3ProteinSpec
    # --------------------------------------------------

    protein = AF3ProteinSpec(
        entity_id="A",
        sequence=sequence,
        template_mmcif=template_cif,
        query_indices=query_indices,
        template_indices=template_indices,
        use_msa=False,
    )

    # --------------------------------------------------
    # 7. Reconstruct AF3JobSpec
    # --------------------------------------------------

    job = AF3JobSpec(
        job_name=job_name,
        sequence_id=sequence_id,
        state_name=state_name,
        structure_kind="target",
        input_dir=prepared_dir,
        json_path=json_path,
        output_dir=af3_output_dir,
        proteins=(protein,),
        model_seeds=model_seeds,
        predicted_target_chain_id="A",
    )

    # --------------------------------------------------
    # 8. Existing AF3 output becomes a completed run
    # --------------------------------------------------

    if not af3_output_dir.exists():
        raise FileNotFoundError(
            f"Missing AF3 output directory: {af3_output_dir}"
        )

    run_result = AF3RunResult(
        job_name=job_name,
        status="completed",
        returncode=0,

        # score_jobs does not need these files,
        # but AF3RunResult requires paths.
        stdout_path=af3_output_dir / "stdout.txt",
        stderr_path=af3_output_dir / "stderr.txt",

        output_dir=af3_output_dir,
        error=None,
    )

    sample = SampleRecord(
        sequence_id=sequence_id,
        sequence=sequence,
    )

    return (
        sample,
        job,
        templates,
        run_result,
    )


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the five AF3 diffusion samples for this sequence/state.

    DynamicMPNN's standard per-sequence aggregator assumes exactly
    state1/state2, so the k-state pullback pipeline uses its own
    state-generic aggregation.
    """

    completed = df[
        df["af3_status"] == "completed"
    ].copy()

    if completed.empty:
        return pd.DataFrame(
            [
                {
                    "af3_status": "failed",
                    "n_completed_samples": 0,
                }
            ]
        )

    row = {
        "sequence_id": completed["sequence_id"].iloc[0],
        "state_name": completed["state_name"].iloc[0],
        "structure_kind": completed["structure_kind"].iloc[0],
        "af3_status": "completed",
        "n_completed_samples": len(completed),

        # Higher is better
        "tm_score_mean": completed["tm_score"].mean(),
        "tm_score_best": completed["tm_score"].max(),

        "lddt_mean": completed["lddt"].mean(),
        "lddt_best": completed["lddt"].max(),

        "mean_plddt_mean": completed["mean_plddt"].mean(),
        "mean_plddt_best": completed["mean_plddt"].max(),

        "ranking_score_mean": completed["ranking_score"].mean(),
        "ranking_score_best": completed["ranking_score"].max(),

        # Lower is better
        "rmsd_mean": completed["rmsd"].mean(),
        "rmsd_best": completed["rmsd"].min(),
    }

    return pd.DataFrame([row])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prepared-dir",
        type=Path,
        required=True,
        help=(
            "Directory produced by "
            "prepare_pullback_with_dynamicmpnn.py"
        ),
    )

    parser.add_argument(
        "--af3-output-dir",
        type=Path,
        required=True,
        help=(
            "AF3 output directory for this job, e.g. "
            "af3_output/native__state_00__target"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scores"),
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        sample,
        job,
        templates,
        run_result,
    ) = load_prepared_job(
        prepared_dir=args.prepared_dir,
        af3_output_dir=args.af3_output_dir,
    )

    # --------------------------------------------------
    # DynamicMPNN scoring
    # --------------------------------------------------

    records = score_jobs(
        jobs=(job,),
        templates=templates,
        run_results={
            job.job_name: run_result
        },
    )

    # --------------------------------------------------
    # DynamicMPNN raw results dataframe
    # --------------------------------------------------

    df = build_all_af3_results_dataframe(
        samples=(sample,),
        records=records,
    )

    raw_path = args.out / "all_af3_results.csv"

    df.to_csv(
        raw_path,
        index=False,
    )

    # --------------------------------------------------
    # Our state-generic aggregation
    # --------------------------------------------------

    summary = build_summary(df)

    summary_path = (
        args.out / "summary_af3_results.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print()
    print("AF3 pullback scoring complete")
    print("============================")
    print()

    cols = [
        "seed",
        "af3_sample",
        "tm_score",
        "lddt",
        "rmsd",
        "mean_plddt",
        "ranking_score",
        "af3_status",
    ]

    available_cols = [
        col for col in cols
        if col in df.columns
    ]

    print(df[available_cols].to_string(index=False))

    print()
    print("Summary")
    print("-------")
    print(summary.to_string(index=False))

    print()
    print(f"Raw results: {raw_path}")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()