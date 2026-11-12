#!/usr/bin/env python3
"""Step 4: compare the new simulations with the mdCATH reference window.

Input: a run folder that holds
  ref_window.pdb, ref_window.xtc, prep.json     from prep_window.py
  fold/<name>_af2.pdb, fold/fold_scores.json    from fold.sh
  prod_pred_r<i>/, prod_native_r<i>/            from run_md.py prod

Output, in <run-dir>/analysis:
  report.md         Summary table and notes. Start here.
  metrics.json      Every number.
  per_residue.csv   Per-residue RMSF and divergences.
  arrays.npz        Maps (DCCM, contact and distance differences).
  *.png             Figures.

Groups: "reference" (the mdCATH window), "predicted" (replicas of the predicted
sequence) and "native" (replicas of the native sequence, the control).
All trajectories are cut to the same number of frames before the comparison.
"""
import argparse
import itertools
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

import compare as cmp
from common import to_backbone

GROUP_LABEL = {"reference": "mdCATH reference", "predicted": "Predicted sequence",
               "native": "Native sequence (control)"}
# Report rows: (metric key, label, number format).
REPORT_ROWS = [
    ("rama_jsd_mean", "phi/psi JS divergence, mean over residues (0-1)", "{:.3f}"),
    ("phi_wasserstein_deg_mean", "phi Wasserstein distance (deg)", "{:.1f}"),
    ("psi_wasserstein_deg_mean", "psi Wasserstein distance (deg)", "{:.1f}"),
    ("ca_dist_wasserstein_A_mean", "CA-CA distance Wasserstein distance (A)", "{:.2f}"),
    ("contact_freq_absdiff_mean", "Contact frequency, mean abs. difference", "{:.3f}"),
    ("state_occupancy_jsd", "State occupancy JS divergence (0-1)", "{:.3f}"),
    ("transition_row_jsd", "Transition probability JS divergence (0-1)", "{:.3f}"),
    ("transition_stay_prob_absdiff", "Probability to stay in a state, abs. difference", "{:.3f}"),
    ("rmsf_pearson", "RMSF Pearson correlation", "{:.2f}"),
    ("rmsf_absdiff_A_mean", "RMSF mean abs. difference (A)", "{:.2f}"),
    ("rmsf_ratio_mean", "RMSF ratio (first / second)", "{:.2f}"),
    ("dccm_pearson", "DCCM Pearson correlation", "{:.2f}"),
    ("dccm_rms_diff", "DCCM RMS difference", "{:.3f}"),
    ("mean_structure_rmsd_A", "RMSD between mean structures (A)", "{:.2f}"),
    ("mean_dist_map_absdiff_A_mean", "Mean distance map, mean abs. difference (A)", "{:.2f}"),
]


def load_runs(run_dir, ref):
    """Return {group: [backbone trajectory, ...]} and the MD summaries."""
    groups, summaries = {"reference": [ref]}, {}
    for who, group in (("pred", "predicted"), ("native", "native")):
        dirs = sorted(run_dir.glob(f"prod_{who}_r*"), key=lambda d: int(d.name.split("_r")[-1]))
        trajs = []
        for d in dirs:
            if not (d / "traj.xtc").exists():
                print(f"skip {d.name}: no traj.xtc")
                continue
            t = to_backbone(md.load(str(d / "traj.xtc"), top=str(d / "top.pdb")))
            if t.n_residues != ref.n_residues:
                raise SystemExit(f"{d.name} has {t.n_residues} residues, the reference has "
                                 f"{ref.n_residues}")
            trajs.append(t)
            summaries[d.name] = json.loads((d / "md_summary.json").read_text())
        if trajs:
            groups[group] = trajs
    return groups, summaries


def compare_pair(a, b, k):
    """Pooled comparison plus one comparison per replica of a (and of b when a has one)."""
    pooled_scalars, arrays = cmp.compare_groups(a, b, k)
    result = dict(pooled=pooled_scalars)
    if len(a) > 1:
        result["per_replica"] = cmp.summarise([cmp.compare_groups([f], b, k)[0] for f in a])
    return result, arrays


def within_group(group, k):
    pairs = list(itertools.combinations(group, 2))
    if not pairs:
        return None
    return dict(per_replica=cmp.summarise([cmp.compare_groups([x], [y], k)[0]
                                           for x, y in pairs]))


def cell(result, key, fmt):
    if result is None:
        return "-"
    if "per_replica" in result:
        s = result["per_replica"][key]
        if s["mean"] is None:
            return "-"
        if s["n"] < 2:
            return fmt.format(s["mean"])
        return f"{fmt.format(s['mean'])} ± {fmt.format(s['std'])}"
    v = result["pooled"][key]
    return "-" if v is None else fmt.format(v)


def write_report(path, meta, comparisons, group_stats, fold):
    cols = [("predicted_vs_reference", "Predicted vs reference"),
            ("native_vs_reference", "Native vs reference"),
            ("predicted_vs_native", "Predicted vs native"),
            ("within_predicted", "Between predicted replicas"),
            ("within_native", "Between native replicas")]
    cols = [(k, label) for k, label in cols if comparisons.get(k)]
    lines = [f"# Dynamics comparison: {meta['traj_id']} frames "
             f"{meta['frame_start']}-{meta['frame_end']}", ""]
    lines += [f"- Temperature: {meta['temperature_K']} K. Residues: {meta['n_residues']}. "
              f"Sequence identity predicted/native: {meta['sequence_identity']:.2f}.",
              f"- Window: {meta['n_frames_window']} frames ({meta['window_ns']:g} ns). "
              f"Frames compared: {meta['n_frames_compared']} "
              f"({meta['compared_ns']:g} ns)."]
    if meta["n_frames_compared"] < meta["n_frames_window"]:
        lines.append("- **Truncated:** at least one simulation stopped at the time budget, so "
                     "every trajectory was cut to the shortest one.")
    for name, f in fold.items():
        lines.append(f"- AF2 model of the {name} sequence: mean pLDDT {f['mean_plddt']:.1f}, "
                     f"CA RMSD to the first reference frame {f['ca_rmsd_to_ref_frame0_A']:.2f} A.")
    lines += ["", "## Summary", "",
              "Values with ± are the mean and standard deviation over replicas (or replica "
              "pairs). The \"between replicas\" columns show how much two runs of the same "
              "sequence differ: a difference from the reference that is no larger than "
              "that is within the noise of one short run.", "",
              "| Metric | " + " | ".join(label for _, label in cols) + " |",
              "|---|" + "---|" * len(cols)]
    for key, label, fmt in REPORT_ROWS:
        lines.append(f"| {label} | " + " | ".join(cell(comparisons[k], key, fmt)
                                                  for k, _ in cols) + " |")
    lines += ["", "## Per-group kinetics and drift", "",
              "| Group | Trajectories | State switches per ns | Slowest implied timescale "
              "at lag 1 (ns) | Mean RMSD to first reference frame (A) |", "|---|---|---|---|---|"]
    for g, s in group_stats.items():
        its = s["implied_timescales_ns_by_lag"].get("1", [None])[0]
        sw = s["state_switches_per_ns"]
        lines.append(f"| {GROUP_LABEL[g]} | {s['n_trajectories']} | "
                     f"{'-' if sw is None else format(sw, '.3f')} | "
                     f"{'-' if its is None else format(its, '.1f')} | "
                     f"{s['rmsd_to_ref_frame0_A_mean']:.2f} |")
    lines += ["", "## Notes", "",
              "- The reference window starts from a conformation in the middle of an mdCATH "
              "run. The new simulations start from an AF2 model after equilibration. The "
              "native-sequence control shows how much difference this and the change of MD "
              "engine (ACEMD to OpenMM) cause.",
              f"- States: {meta['n_states']} k-means clusters on TICA/PCA coordinates of "
              "phi/psi and CA-CA distances, fitted on all trajectories together. With this "
              "few frames, transition probabilities and implied timescales have large "
              "uncertainty. Compare them with the between-replica columns.",
              "- The pooled values in metrics.json use all replicas of a group together; the "
              "table uses one replica at a time, which matches the sampling of the reference.",
              "", "Figures: rmsf.png, rmsd.png, rama_jsd.png, dccm.png, dccm_diff.png, contacts.png, "
              "states.png.",
              ""]
    path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=Path("."))
    p.add_argument("--states", type=int, default=5, help="largest number of states")
    args = p.parse_args()
    run, out = args.run_dir, args.run_dir / "analysis"
    out.mkdir(exist_ok=True)

    prep = json.loads((run / "prep.json").read_text())
    frame_ns = prep["frame_interval_ns"]
    ref = md.load(str(run / "ref_window.xtc"), top=str(run / "ref_window.pdb"))
    trajs, md_summaries = load_runs(run, ref)
    if len(trajs) < 2:
        raise SystemExit("no simulation trajectories found")

    n = min(t.n_frames for group in trajs.values() for t in group)
    if n < 3:
        raise SystemExit(f"only {n} common frames; need at least 3")
    trajs = {g: [t[:n] for t in group] for g, group in trajs.items()}
    feats = {g: [cmp.featurize(t, f"{g}_{i}") for i, t in enumerate(group)]
             for g, group in trajs.items()}
    everything = [f for group in feats.values() for f in group]
    projections, k = cmp.fit_states(everything, k=args.states)

    comparisons, arrays = {}, {}
    for a, b in (("predicted", "reference"), ("native", "reference"), ("predicted", "native")):
        if a in feats and b in feats:
            comparisons[f"{a}_vs_{b}"], arr = compare_pair(feats[a], feats[b], k)
            arrays.update({f"{a}_vs_{b}__{key}": v for key, v in arr.items()})
    for g in ("predicted", "native"):
        if g in feats:
            comparisons[f"within_{g}"] = within_group(feats[g], k)

    ref0 = trajs["reference"][0][0]
    group_stats, rmsd = {}, {}
    for g, group in feats.items():
        rmsd[g] = [cmp.rmsd_series(t, ref0) for t in trajs[g]]
        stats = cmp.state_summary(group, k, frame_ns)
        stats.update(n_trajectories=len(group),
                     rmsd_to_ref_frame0_A_mean=float(np.mean(rmsd[g])),
                     rmsd_to_ref_frame0_A_final=float(np.mean([r[-1] for r in rmsd[g]])),
                     rmsd_to_own_frame0_A_mean=float(np.mean(
                         [cmp.rmsd_series(t, t[0]).mean() for t in trajs[g]])),
                     rmsf_A_mean=float(cmp.group_mean(group, "rmsf").mean()))
        group_stats[g] = stats

    fold = {}
    scores_file = run / "fold" / "fold_scores.json"
    if scores_file.exists():
        scores = json.loads(scores_file.read_text())
        ca = ref0.topology.select("name CA")
        for name, label in (("pred", "predicted"), ("native", "native")):
            model = run / "fold" / f"{name}_af2.pdb"
            if name in scores and model.exists():
                m = to_backbone(md.load(str(model)))
                fold[label] = dict(scores[name], ca_rmsd_to_ref_frame0_A=cmp.kabsch_rmsd(
                    m.xyz[0, ca] * 10.0, ref0.xyz[0, ca] * 10.0))

    meta = dict(traj_id=prep["traj_id"], temperature_K=prep["temperature_K"],
                frame_start=prep["frame_start"], frame_end=prep["frame_end"],
                n_residues=prep["n_residues"], sequence_identity=prep["sequence_identity"],
                n_frames_window=prep["n_frames"], window_ns=prep["window_ns"],
                n_frames_compared=n, compared_ns=(n - 1) * frame_ns,
                frame_interval_ns=frame_ns, n_states=k,
                native_seq=prep["native_seq"], pred_seq=prep["pred_seq"])
    (out / "metrics.json").write_text(json.dumps(
        dict(meta=meta, fold=fold, comparisons=comparisons, groups=group_stats,
             md=md_summaries), indent=2) + "\n")

    table = pd.DataFrame(dict(residue=np.arange(1, prep["n_residues"] + 1),
                              native=list(prep["native_seq"]), predicted=list(prep["pred_seq"])))
    for g, group in feats.items():
        table[f"rmsf_A_{g}"] = cmp.group_mean(group, "rmsf")
    for name in ("predicted_vs_reference", "native_vs_reference", "predicted_vs_native"):
        for key in ("rama_jsd", "phi_wasserstein_deg", "psi_wasserstein_deg",
                    "ca_dist_wasserstein_A"):
            if f"{name}__{key}" in arrays:
                table[f"{key}_{name}"] = arrays[f"{name}__{key}"]
    table.to_csv(out / "per_residue.csv", index=False, float_format="%.4f")

    for g, group in feats.items():
        arrays[f"dccm_{g}"] = cmp.group_mean(group, "dccm")
    np.savez_compressed(out / "arrays.npz", **arrays)

    write_report(out / "report.md", meta, comparisons, group_stats, fold)

    import plots
    plots.make_all(out, feats, rmsd, arrays, projections, group_stats, frame_ns, k)
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
