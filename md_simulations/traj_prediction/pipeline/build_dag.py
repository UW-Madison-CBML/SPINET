#!/usr/bin/env python3
"""Write an HTCondor DAG that runs the pipeline for every row of a manifest.

Manifest CSV columns
--------------------
traj_id      <domain>_<temperature>_<replica>, e.g. 3j7yg00_320_4
shard        File name of the shard that holds the trajectory, e.g. mdcath_shard_39.h5
frame_start  First frame of the window (0-based)
frame_end    Last frame of the window (0-based, part of the window)
method       Name of the inverse folding model (used in the run folder name)
pred_seq     Predicted sequence, 1-letter codes

Per row the DAG runs:
  prep -> fold -> equil (predicted, native) -> prod (replicas) -> analyze
and writes everything to <out>/<traj_id>_<start>-<end>_<method>/.

Example
-------
python build_dag.py manifest.csv --out runs
condor_submit_dag runs/pipeline.dag
"""
import argparse
import re
from pathlib import Path

import csv

RESEARCH_DRIVE = "pelican://chtc.wisc.edu/researchdrive/dbhaskar3/CHTC/sheaf_protein_dynamics"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--replicas", type=int, default=3,
                   help="production replicas of the predicted sequence")
    p.add_argument("--native-replicas", type=int, default=3,
                   help="production replicas of the native sequence (0: no control)")
    p.add_argument("--budget-hours", type=float, default=5.0,
                   help="wall-clock limit for equilibration + production of one simulation")
    p.add_argument("--equil-ns", type=float, default=20.0, help="mdCATH used 20 ns")
    p.add_argument("--frame-interval-ns", type=float, default=1.0, help="mdCATH saves 1 ns")
    p.add_argument("--shard-url", default=f"{RESEARCH_DRIVE}/mdcath",
                   help="folder (path or URL) that holds the shards")
    p.add_argument("--shard-disk", default="40GB", help="disk request of the prep job")
    p.add_argument("--allow-breaks", action="store_true",
                   help="accept domains with chain breaks")
    p.add_argument("--af2-params", default=f"{RESEARCH_DRIVE}/colabfold_params.tar.gz",
                   help="path or URL of colabfold_params.tar.gz (default: download per job)")
    p.add_argument("--md-image", default="cmikulski4/traj-md:v1")
    p.add_argument("--fold-image", default="ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2")
    args = p.parse_args()

    pipeline = Path(__file__).resolve().parent
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    dag = []

    def job(name, sub, **variables):
        dag.append(f"JOB {name} {pipeline / 'condor' / sub}")
        variables.update(pipeline_dir=pipeline, md_image=args.md_image,
                         fold_image=args.fold_image)
        dag.append(f"VARS {name} " + " ".join(f'{k}="{v}"' for k, v in variables.items()))

    for row in rows:
        traj_id = row["traj_id"].strip()
        domain, temperature, _ = traj_id.rsplit("_", 2)
        start, end = int(row["frame_start"]), int(row["frame_end"])
        seq = row["pred_seq"].strip().upper()
        if not 0 <= start < end:
            raise SystemExit(f"{traj_id}: need 0 <= frame_start < frame_end")
        if not re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+", seq):
            raise SystemExit(f"{traj_id}: pred_seq has letters that are not amino acids")
        method = re.sub(r"[^A-Za-z0-9]+", "-", row.get("method", "") or "pred").strip("-")
        run_id = f"{traj_id}_{start}-{end}_{method}"
        tag = re.sub(r"[^A-Za-z0-9_]", "_", run_id)
        run_dir = out / run_id
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)

        groups = [("pred", args.replicas, 100)]
        if args.native_replicas > 0:
            groups.append(("native", args.native_replicas, 200))

        job(f"prep_{tag}", "prep.sub", run_dir=run_dir, shard=row["shard"].strip(),
            shard_url=f"{args.shard_url.rstrip('/')}/{row['shard'].strip()}",
            shard_disk=args.shard_disk, traj_id=traj_id, frame_start=start, frame_end=end,
            pred_seq=seq, frame_interval_ns=args.frame_interval_ns,
            prep_flags="--allow-breaks" if args.allow_breaks else "")
        fold_inputs = "seqs.fasta" + (f", {args.af2_params}" if args.af2_params else "")
        job(f"fold_{tag}", "fold.sub", run_dir=run_dir, fold_inputs=fold_inputs,
            fold_names=" ".join(who for who, _, _ in groups))
        dag.append(f"RETRY fold_{tag} 2")
        dag.append(f"PARENT prep_{tag} CHILD fold_{tag}")

        prod_jobs, prod_dirs = [], []
        for who, n_rep, seed0 in groups:
            job(f"equil_{who}_{tag}", "equil.sub", run_dir=run_dir, who=who,
                equil_ns=args.equil_ns, seed=seed0)
            dag.append(f"PARENT fold_{tag} CHILD equil_{who}_{tag}")
            for r in range(n_rep):
                name = f"prod_{who}_r{r}_{tag}"
                job(name, "prod.sub", run_dir=run_dir, who=who, replica=r,
                    temperature=temperature, n_frames=end - start + 1,
                    frame_interval_ps=args.frame_interval_ns * 1000.0,
                    budget_hours=args.budget_hours, seed=seed0 + 1 + r)
                dag.append(f"PARENT equil_{who}_{tag} CHILD {name}")
                prod_jobs.append(name)
                prod_dirs.append(f"prod_{who}_r{r}")
        job(f"analyze_{tag}", "analyze.sub", run_dir=run_dir, prod_dirs=", ".join(prod_dirs))
        dag.append(f"PARENT {' '.join(prod_jobs)} CHILD analyze_{tag}")
        dag.append("")

    dag_file = out / "pipeline.dag"
    dag_file.write_text("\n".join(dag) + "\n")
    n_gpu = sum(1 + (1 + args.replicas) + ((1 + args.native_replicas)
                                           if args.native_replicas else 0) for _ in rows)
    print(f"{len(rows)} manifest rows, {n_gpu} GPU jobs -> {dag_file}")
    print(f"submit with: condor_submit_dag {dag_file}")


if __name__ == "__main__":
    main()
