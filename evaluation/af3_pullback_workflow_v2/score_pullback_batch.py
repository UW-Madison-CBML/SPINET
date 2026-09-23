#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from dynamicmpnn.eval.af3 import AF3RunResult
from dynamicmpnn.eval.aggregation import build_all_af3_results_dataframe
from dynamicmpnn.eval.inputs import parse_structure
from dynamicmpnn.eval.sampling import SampleRecord
from dynamicmpnn.eval.scoring import score_jobs
from dynamicmpnn.eval.templates import TemplateArtifact
from dynamicmpnn.eval.types import AF3JobSpec, AF3ProteinSpec

REQUIRED = {"job_name","domain","temperature","method","design_id","sequence_id","state_id","sequence","sequence_length","query_indices","template_indices"}
METRICS = ("tm_score","lddt","rmsd","mean_plddt","ranking_score")
HIGHER = {"tm_score","lddt","mean_plddt","ranking_score"}

def parse_indices(value):
    return tuple(int(x) for x in json.loads(value) if True)

def detect_chain(pdb_path):
    model = next(PDBParser(QUIET=True).get_structure("ref", pdb_path).get_models())
    chains = [str(c.id) for c in model.get_chains() if any(is_aa(r, standard=False) and r.id[0] in {" ","H_MSE"} for r in c.get_residues())]
    if len(chains) != 1:
        raise ValueError(f"{pdb_path}: expected one protein chain, found {chains}")
    return chains[0]

def reference_pdb(references_dir, domain, state_id):
    path = references_dir / domain / f"{domain}__{state_id}__target.pdb"
    if path.exists():
        return path
    matches = list(references_dir.rglob(f"{domain}__{state_id}__target.pdb"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Missing reference PDB for {domain}/{state_id}")

def build_template(pdb_path, state_id, query_indices, template_indices):
    chain_id = detect_chain(pdb_path)
    parsed = parse_structure(name=state_id, pdb_path=pdb_path, chain_id=chain_id)
    target = parsed.target_chain
    ca_coords = tuple(tuple(float(v) for v in xyz) for xyz in target.coords[:,1,:])
    return TemplateArtifact(
        state_name=state_id, structure_kind="target", pdb_path=pdb_path,
        mmcif_path=pdb_path.with_suffix(".cif"), query_indices=query_indices,
        template_indices=template_indices, ca_coords=ca_coords,
    )

def reconstruct(manifest, references_dir, af3_output_dir):
    samples = {}
    templates = {}
    jobs = []
    run_results = {}
    for row in manifest.itertuples(index=False):
        sequence_id = str(row.sequence_id)
        state_id = str(row.state_id)
        job_name = str(row.job_name)
        sequence = str(row.sequence).strip().upper()
        qidx = parse_indices(row.query_indices)
        tidx = parse_indices(row.template_indices)

        if sequence_id not in samples:
            samples[sequence_id] = SampleRecord(sequence_id=sequence_id, sequence=sequence)
        elif samples[sequence_id].sequence != sequence:
            raise ValueError(f"{sequence_id}: inconsistent sequence across states")

        score_state_name = f"{row.domain}__{state_id}"
        key = (score_state_name, "target")
        if key not in templates:
            template = build_template(reference_pdb(references_dir, str(row.domain), state_id), state_id, qidx, tidx)
            templates[key] = TemplateArtifact(
                state_name=score_state_name, structure_kind="target", pdb_path=template.pdb_path,
                mmcif_path=template.mmcif_path, query_indices=template.query_indices,
                template_indices=template.template_indices, ca_coords=template.ca_coords,
            )

        protein = AF3ProteinSpec(
            entity_id="A", sequence=sequence, template_mmcif=templates[key].mmcif_path,
            query_indices=qidx, template_indices=tidx, use_msa=False,
        )
        output_dir = af3_output_dir / job_name
        jobs.append(AF3JobSpec(
            job_name=job_name, sequence_id=sequence_id, state_name=score_state_name, structure_kind="target",
            input_dir=Path("."), json_path=Path(f"{job_name}.json"), output_dir=output_dir,
            proteins=(protein,), model_seeds=(1,), predicted_target_chain_id="A",
        ))
        if output_dir.exists():
            run_results[job_name] = AF3RunResult(job_name, "completed", 0, output_dir/"stdout.txt", output_dir/"stderr.txt", output_dir, None)
        else:
            run_results[job_name] = AF3RunResult(job_name, "failed", None, output_dir/"stdout.txt", output_dir/"stderr.txt", output_dir, f"Missing AF3 output: {output_dir}")
    return tuple(samples.values()), tuple(jobs), templates, run_results

def summarize_sequence_state(raw):
    keys = ["domain","temperature","method","design_id","sequence_id","sequence","state_name"]
    rows = []
    for values, group in raw.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        done = group[group.af3_status == "completed"]
        row["af3_status"] = "completed" if len(done)==len(group) and len(group)>0 else ("failed" if done.empty else "partial")
        row["n_af3_samples"] = len(group)
        row["n_completed_samples"] = len(done)
        for m in METRICS:
            s = pd.to_numeric(done[m], errors="coerce").dropna()
            row[f"{m}_mean"] = s.mean() if len(s) else np.nan
            row[f"{m}_std"] = s.std(ddof=0) if len(s) else np.nan
            row[f"{m}_best"] = (s.max() if m in HIGHER else s.min()) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_sequences(df):
    keys = ["domain","temperature","method","design_id","sequence_id","sequence"]
    rows = []
    for values, group in df.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        row["n_states"] = len(group)
        row["n_completed_states"] = int((group.af3_status=="completed").sum())
        row["af3_status"] = "completed" if row["n_completed_states"]==row["n_states"] else ("failed" if row["n_completed_states"]==0 else "partial")
        for m in METRICS:
            s = pd.to_numeric(group[f"{m}_mean"], errors="coerce").dropna()
            row[f"{m}_state_mean"] = s.mean() if len(s) else np.nan
            row[f"{m}_state_median"] = s.median() if len(s) else np.nan
            row[f"{m}_worst_state"] = ((s.min() if m in HIGHER else s.max()) if len(s) else np.nan)
            row[f"{m}_best_state"] = ((s.max() if m in HIGHER else s.min()) if len(s) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_domains(df):
    keys = ["domain","temperature","method"]
    rows = []
    for values, group in df.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        row["n_sequences"] = len(group)
        row["n_completed_sequences"] = int((group.af3_status=="completed").sum())
        row["af3_status"] = "completed" if row["n_completed_sequences"]==row["n_sequences"] else ("failed" if row["n_completed_sequences"]==0 else "partial")
        for m in METRICS:
            s = pd.to_numeric(group[f"{m}_state_mean"], errors="coerce").dropna()
            row[f"{m}_sequence_mean"] = s.mean() if len(s) else np.nan
            row[f"{m}_sequence_median"] = s.median() if len(s) else np.nan
            row[f"{m}_sequence_q25"] = s.quantile(.25) if len(s) else np.nan
            row[f"{m}_sequence_q75"] = s.quantile(.75) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--references-dir", type=Path, required=True)
    p.add_argument("--af3-output-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    missing = REQUIRED - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if manifest.job_name.duplicated().any():
        raise ValueError("Duplicate job_name values found")

    samples, jobs, templates, run_results = reconstruct(manifest, args.references_dir, args.af3_output_dir)
    print(f"Manifest jobs: {len(jobs)}")
    print(f"Unique sequences: {len(samples)}")
    print(f"Outputs present: {sum(r.status=='completed' for r in run_results.values())}/{len(run_results)}")

    records = score_jobs(jobs=jobs, templates=templates, run_results=run_results)
    raw = build_all_af3_results_dataframe(samples=samples, records=records)

    meta = manifest[["domain","temperature","method","design_id","sequence_id","state_id"]].copy()
    meta["state_name"] = meta["domain"].astype(str) + "__" + meta["state_id"].astype(str)
    raw = raw.merge(meta.drop(columns=["state_id"]), on=["sequence_id","state_name"], how="left", validate="many_to_one")
    raw["state_name"] = raw["state_name"].str.rsplit("__", n=1).str[-1]

    args.out.mkdir(parents=True, exist_ok=True)
    seq_state = summarize_sequence_state(raw)
    seq = summarize_sequences(seq_state)
    dom = summarize_domains(seq)

    raw.to_csv(args.out/"all_af3_samples.csv", index=False)
    seq_state.to_csv(args.out/"sequence_state_summary.csv", index=False)
    seq.to_csv(args.out/"sequence_summary.csv", index=False)
    dom.to_csv(args.out/"domain_summary.csv", index=False)

    print(f"Raw rows: {len(raw)}")
    print(f"Sequence-state rows: {len(seq_state)}")
    print(f"Sequence rows: {len(seq)}")
    print(f"Domain rows: {len(dom)}")
    print(f"Output: {args.out}")

if __name__ == "__main__":
    main()
