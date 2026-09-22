"""Hyperparameter sweep over DynamicMPNN's `k` -- the number of MD conformers the model
sees per protein -- on a single dataset (ATLAS by default).

`k` is the one hyperparameter that is *about* the thing this comparison exists to measure:
MapDiff and PiFold each see one static structure, DynamicMPNN sees an ensemble, and this
sweep is what says how much of DynamicMPNN's recovery is bought by ensemble size and where
that curve flattens. Everything else (splits, seed, epochs, lr, optimizer batch, scoring)
is held fixed across the grid so `k` is the only thing that moves.

This reuses comparisons/DynamicMPNN/train.py wholesale -- same preprocessing, same
featurizer wiring, same `run_validation` metric set, same W&B keys -- and only adds the
three things a sweep needs that a single run does not:

  1. **Shared preprocessing.** The conformer *pool* (`build_processed_dataset`) is built
     ONCE, at `--pool-size` = max(grid), and every k draws its k conformers out of that same
     pool. Rebuilding it per k would repeat the O(n_frames^2) pairwise-TM pass for nothing,
     and -- worse -- would let each k sample from a differently-chosen pool, which is a
     second variable moving alongside the one being swept.

  2. **A VRAM budget instead of a fixed batch size.** `forward_single_chain` embeds and
     encodes all k conformers before pooling them (node/edge activations are [N, k, D] and
     [E, k, D] through all 8 encoder layers; the decoder runs on the pooled, k-independent
     representation). So activation memory is linear in k -- and a little worse than linear,
     because the edge topology is the *union* of the k per-conformer knn_32 graphs, so E
     grows with k too before saturating. A fixed `--batch-size 2` that fits at k=2 therefore
     OOMs by k=32. Instead `--conf-budget` caps k * batch_size (the real memory unit is the
     conformer-graph, not the protein), and gradient accumulation makes up the difference so
     that every k still takes an optimizer step every `--proteins-per-step` proteins. Without
     that accumulation the effective batch size would shrink as k grows and the sweep would
     be measuring batch size as much as ensemble size.

  3. **k > the old 10-conformer ceiling.** Nothing in the architecture caps k; the ceiling
     was `--pool-size`'s default of 10, i.e. how many frames the preprocessing bothered to
     save. `ProteinGraphFeaturiserSingleChain.get_entries` *silently duplicates* conformers
     when the pool is smaller than k (its `n < self.k` branch), so a k above the pool size
     does not error -- it quietly feeds the model the same frame twice and flattens the top
     of the curve for a reason that has nothing to do with dynamics. `check_pool_sizes`
     below turns that silent failure into a loud one.

Each k gets its own W&B run (named `<run-name>_k<k>`, all sharing one `group=` so they
overlay in a single plot), logging exactly the keys train.py logs. After the grid finishes,
a final `<run-name>_summary` run logs one `recovery_vs_k` table of every k's held-out test
metrics, and the same table is written to `--results-csv` so it survives the job.

Usage (see sweep_k.sub for the CHTC submission):

    python sweep_k.py --ds-name atlas --k-grid 2,3,4,5,6,8,10,12,16,20,24,32
"""

import argparse
import csv
import gc
import json
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from tqdm import tqdm

# `import train` must come first: train.py is what puts lib/, scripts/ and
# DynamicMPNN/src on sys.path (it mirrors the flat-import convention the rest of the repo's
# CHTC scripts use), so the `dataset_splits` / `dynamicmpnn` imports below resolve
# only after it has run. Do not let an import sorter reorder these.
from train import (
    DEVICE,
    GAP_TOKEN,
    THIS_DIR,
    REPO_ROOT,
    PdbTrackingFeaturiser,
    build_processed_dataset,
    filter_by_length,
    load_pretrained_weights,
    run_validation,
)

import dataset_splits

from dynamicmpnn import constants
from dynamicmpnn.datamodules.pt_dataset import PTFileDataset
from dynamicmpnn.datamodules.sampler import safe_collate
from dynamicmpnn.types import DISTANCE_EPS

# The sweep's default grid: dense where the recovery-vs-k curve is expected to bend (every
# integer up to 6, where each added conformer is a large relative increase in ensemble
# coverage) and log-spaced through the tail, where it is not. 32 is not a limit of anything
# -- it is just the point past which conformers cost more than they are likely to buy.
DEFAULT_K_GRID = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32)

# k * batch_size, i.e. how many conformer-graphs may be resident in one forward/backward.
# 20 is the load the existing single run already carries on bhaskargpu4000 (train.sub's
# `--k 10` at `--batch-size 2`), so holding the product there keeps every point in the grid
# inside a footprint that is known to fit rather than one that is merely believed to.
DEFAULT_CONF_BUDGET = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- what to sweep ---
    parser.add_argument("--k-grid", type=str, default=",".join(str(k) for k in DEFAULT_K_GRID),
                        help="comma-separated conformer counts to train, in order "
                             f"(default: {','.join(str(k) for k in DEFAULT_K_GRID)})")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="conformer pool saved per protein (default: max of --k-grid). "
                             "Must be >= every k, or the featurizer pads by duplicating frames.")
    parser.add_argument("--conf-budget", type=int, default=DEFAULT_CONF_BUDGET,
                        help="cap on k * batch_size, the real unit of activation memory "
                             f"(default: {DEFAULT_CONF_BUDGET})")
    parser.add_argument("--proteins-per-step", type=int, default=2,
                        help="proteins per optimizer step, held constant across the grid by "
                             "gradient accumulation (default: 2)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override the --conf-budget-derived per-k batch size (not "
                             "recommended: a fixed batch size makes memory scale with k)")

    # --- everything below mirrors train.py's flags, same defaults ---
    parser.add_argument("--ds-name", type=str, default="atlas", choices=dataset_splits.DATASETS)
    parser.add_argument("--traj-h5", type=Path, default=None)
    parser.add_argument("--index-csv", type=Path, default=None)
    parser.add_argument("--val-fold", type=int, default=dataset_splits.DEFAULT_VAL_FOLD)
    parser.add_argument("--test-fold", type=int, default=dataset_splits.DEFAULT_TEST_FOLD)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=dataset_splits.MAX_LENGTH)
    parser.add_argument("--course-grain", type=int, default=25)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=dataset_splits.SEED)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="override the default single_chain_k{k}.ckpt (resolved per k)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="W&B run-name stem; each point logs as <stem>_k<k> (default: "
                             "DynamicMPNN_<ds>_ksweep)")
    parser.add_argument("--results-csv", type=Path, default=None,
                        help="where to write the recovery-vs-k table (default: "
                             "./sweep_k_<ds>.csv)")
    return parser.parse_args()


def parse_k_grid(raw: str) -> list:
    grid = sorted({int(token) for token in raw.split(",") if token.strip()})
    if not grid:
        raise ValueError("--k-grid is empty")
    # get_entries' n == 1 branch raises and its n < 2 branch returns None: a single conformer
    # is not an ensemble, and this model has no single-structure mode to fall back on.
    if grid[0] < 2:
        raise ValueError(f"k must be >= 2 (DynamicMPNN pools conformers); got {grid[0]}")
    return grid


def batch_size_for_k(k: int, conf_budget: int, proteins_per_step: int, override) -> tuple:
    """Per-k (batch_size, grad_accum) holding both the memory footprint and the optimizer
    batch fixed across the grid.

    batch_size falls as k rises so that k * batch_size stays under `conf_budget`; grad_accum
    rises to compensate so an optimizer step still consumes `proteins_per_step` proteins.
    It is also capped ABOVE at `proteins_per_step` -- the budget leaves room for a batch of
    10 at k=2, but taking it would mean small k optimizing over 10 proteins per step and
    large k over 2, which is the same confound running the other way. So the optimizer batch
    is exactly `proteins_per_step` at every k, and only how it is split changes.

    At large k the budget floors batch_size at 1 -- one protein's k conformers is the
    smallest forward this model has -- so the footprint does keep creeping up past
    k == conf_budget. That is the point where a bigger card, not a smaller batch, is the
    remaining lever.
    """
    batch_size = override if override else max(1, min(proteins_per_step, conf_budget // k))
    grad_accum = max(1, -(-proteins_per_step // batch_size))  # ceil, so >= proteins_per_step
    return batch_size, grad_accum


def check_pool_sizes(processed_dir: Path, pdb_ids: list, max_k: int) -> None:
    """Fail before training if any protein's saved pool is smaller than the largest k.

    `ProteinGraphFeaturiserSingleChain.get_entries` handles `n < k` by duplicating conformers
    at random, which is silent: the run completes, the assert on `len(confs_list) == k`
    passes, and the model is simply fed the same frame more than once. Across a sweep that
    reads as the recovery curve saturating, so it has to be an error here rather than a
    warning nobody sees. A pool comes up short when the trajectory has fewer coarse-grained
    frames than max_k (`farthest_point_sample` clamps to what exists), i.e. when
    --course-grain is too aggressive for a short trajectory.
    """
    short = {}
    for pdb_id in pdb_ids:
        path = processed_dir / f"{pdb_id}.pt"
        if not path.exists():  # build_processed_dataset already warned and skipped it
            continue
        n = len(torch.load(path, weights_only=False).cluster_members)
        if n < max_k:
            short[pdb_id] = n
    if short:
        worst = min(short.values())
        raise RuntimeError(
            f"{len(short)} protein(s) have a conformer pool smaller than the largest k "
            f"({max_k}); the smallest is {worst}. The featurizer would pad these by "
            f"duplicating frames, which looks like the recovery curve flattening. "
            f"Offenders: {dict(sorted(short.items(), key=lambda kv: kv[1])[:10])}"
            f"{' ...' if len(short) > 10 else ''}. Lower the top of --k-grid to {worst}, or "
            f"lower --course-grain so more frames survive subsampling, then --force-rebuild."
        )


def build_datasets(pdb_lists: dict, features_cfg, processed_dir: Path) -> tuple:
    """Featurize all three splits at the current k.

    in_memory=True throughout, matching train.py: the featurizer's k-of-pool draw happens at
    construction and then holds for every epoch, so the val and test sets are fixed rather
    than resampled each epoch. That has to stay true at every k -- if large k fell back to
    on-the-fly featurization, the sweep would be varying "fixed vs. resampled conformers"
    alongside k. It costs RAM linear in k, which is what sweep_k.sub's request_memory covers.
    """
    datasets, pdb_orders = {}, {}
    for split in ("train", "val", "test"):
        # Only val/test need PdbTrackingFeaturiser -- run_validation maps batches back to
        # pdb_codes through it to label each design -- but using it for all three keeps the
        # "featurized n/m" bookkeeping uniform.
        tracker = PdbTrackingFeaturiser(hydra.utils.instantiate(
            features_cfg, split=split, device="cpu", distance_eps=DISTANCE_EPS))
        datasets[split] = PTFileDataset(
            pdb_codes=pdb_lists[split],
            cfg_features=tracker,
            processed_dir=processed_dir,
            split=split,
            in_memory=True,
        )
        pdb_orders[split] = tracker.kept_pdb_codes
    return datasets, pdb_orders


def train_one_k(k, args, pdb_lists, processed_dir, group_name):
    """One full train+val+test run at a single k. Returns its test metrics."""
    batch_size, grad_accum = batch_size_for_k(
        k, args.conf_budget, args.proteins_per_step, args.batch_size)

    # Reseed identically for every k. DynamicMPNN's featurizer draws its k-of-pool with
    # `random` / `np.random` (get_entries -> compute_sequential_probabilities), so without
    # this each point in the grid would also differ by its conformer draw and its weight
    # init -- the sweep would not be a controlled comparison.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name=f"{args.run_name}_k{k}",
        group=group_name,
        reinit=True,
        config={
            "ds_name": args.ds_name,
            "val_fold": args.val_fold,
            "test_fold": args.test_fold,
            "max_length": args.max_length,
            "k": k,
            "pool_size": args.pool_size,
            "course_grain": args.course_grain,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "proteins_per_step": batch_size * grad_accum,
            "conf_budget": args.conf_budget,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "from_scratch": args.from_scratch,
            "sweep_group": group_name,
            "task": f"predicting residues from {args.ds_name} MD conformer ensembles "
                    f"(DynamicMPNN, k={k})",
        },
    )

    features_cfg = OmegaConf.load(
        constants.HYDRA_CONFIG_PATH / "features" / "ca_bb_single_chain.yaml")
    features_cfg.k = k
    model_cfg = OmegaConf.load(
        constants.HYDRA_CONFIG_PATH / "model" / "AR1_single_chain.yaml")
    model = hydra.utils.instantiate(model_cfg).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.log({"params": num_params})

    if not args.from_scratch:
        # Upstream ships one checkpoint per k (single_chain_k2 / k3 / ...), so the resolved
        # path moves with the sweep. Most of the grid has no matching checkpoint, and mixing
        # pretrained and from-scratch points would make the curve uninterpretable -- see
        # train.sub on why this comparison trains from scratch in the first place.
        ckpt_path = args.ckpt or (THIS_DIR / "DynamicMPNN" / "checkpoints" / f"single_chain_k{k}.ckpt")
        if ckpt_path.exists():
            load_pretrained_weights(model, ckpt_path)
        else:
            logger.warning(f"k={k}: no checkpoint at {ckpt_path}, training from scratch instead.")

    datasets, pdb_orders = build_datasets(pdb_lists, features_cfg, processed_dir)
    for split in ("val", "test"):
        logger.info(f"k={k}: featurized {len(pdb_orders[split])}/{len(pdb_lists[split])} "
                    f"{split} proteins; dropped "
                    f"{sorted(set(pdb_lists[split]) - set(pdb_orders[split]))}")
    run.summary["split_counts"] = {s: len(datasets[s]) for s in datasets}
    run.summary["val_split_ids"] = list(pdb_orders["val"])
    run.summary["test_split_ids"] = list(pdb_orders["test"])

    loaders = {
        split: DataLoader(datasets[split], batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=args.num_workers, collate_fn=safe_collate)
        for split in datasets
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=GAP_TOKEN)

    logger.info(f"k={k}: batch_size {batch_size} x grad_accum {grad_accum} "
                f"({batch_size * grad_accum} proteins/step, {k * batch_size} conformer-graphs "
                f"resident), {len(datasets['train'])} train proteins, {args.epochs} epochs")

    step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for micro_step, batch in enumerate(tqdm(
                loaders["train"], desc=f"k={k} Epoch {epoch} Train")):
            batch = batch.to(DEVICE)
            logits, valid_mask = model(batch)
            loss = crit(logits[valid_mask], batch.seq[valid_mask])

            # Scale before backward so the accumulated gradient is the mean over the whole
            # effective batch, not its sum -- otherwise the effective learning rate would
            # scale with grad_accum, i.e. with k, which is exactly the confound the
            # accumulation is here to remove.
            (loss / grad_accum).backward()

            if (micro_step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            step += 1
            run.log({"train_loss": loss.item(), "epoch": epoch, "step": step})

        # A trailing partial accumulation group (len(loader) not divisible by grad_accum)
        # still holds gradients; spend them rather than carrying them into the next epoch.
        if len(loaders["train"]) % grad_accum:
            optimizer.step()
            optimizer.zero_grad()

        logger.info(f"k={k} epoch {epoch}: train loss "
                    f"{total_loss / max(len(loaders['train']), 1):.4f}")

        metrics = run_validation(model, loaders["val"], crit, pdb_orders["val"], epoch, run,
                                 num_params=num_params)
        print(f"k={k} epoch {epoch}: top-1 {metrics['val_top1_acc_mean']:.4f} "
              f"perp {metrics['val_perp_mean']:.4f}")

    # One test score, with the final weights. No early stopping and no best-epoch restore, so
    # "final" and "selected" are the same model at every k -- nothing the test split touched,
    # and no per-k model selection to make the points incomparable.
    logger.info(f"k={k}: scoring the held-out test split ({len(pdb_orders['test'])} proteins)")
    test_metrics = run_validation(model, loaders["test"], crit, pdb_orders["test"], args.epochs - 1, run,
                                  num_params=num_params, split_name="test")
    run.summary.update({m: v for m, v in test_metrics.items() if isinstance(v, (int, float))})
    run.finish()

    row = {"k": k, "batch_size": batch_size, "grad_accum": grad_accum, "params": num_params}
    row.update({m: v for m, v in test_metrics.items() if isinstance(v, (int, float))})

    # The in-memory featurized splits are the sweep's largest allocation and they scale with
    # k, so release them (and the CUDA cache) before the next, larger k builds its own.
    del loaders, datasets, model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def result_columns(rows) -> list:
    columns = sorted({key for row in rows for key in row})
    return ["k"] + [c for c in columns if c != "k"]


def write_results_csv(rows, path) -> None:
    """Rewritten after every sweep point, so an eviction part-way through still leaves the
    points that did finish -- they are the expensive part and they do not need redoing."""
    columns = result_columns(rows)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} sweep point(s) to {path}")


def log_summary(rows, args, k_grid, group_name):
    """One run holding the whole curve: k on the x-axis, held-out test metrics on the y."""
    columns = result_columns(rows)

    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name=f"{args.run_name}_summary",
        group=group_name,
        reinit=True,
        config={"ds_name": args.ds_name, "k_grid": k_grid, "epochs": args.epochs,
                "seed": args.seed, "conf_budget": args.conf_budget,
                "task": f"recovery vs. ensemble size k on {args.ds_name} (DynamicMPNN)"},
    )
    # Same artifact train.py logs, so a sweep is reproducible from its own W&B run rather
    # than from whatever the working tree happens to hold now.
    artifact = wandb.Artifact(name="scripts", type="model_file")
    for dependency in (THIS_DIR / "sweep_k.py", THIS_DIR / "train.py",
                       REPO_ROOT / "lib" / "stats_utils.py",
                       REPO_ROOT / "lib" / "residue_classifier_dataset.py",
                       REPO_ROOT / "lib" / "dataset_splits.py",
                       REPO_ROOT / "scripts" / "load_dynamics.py"):
        if dependency.exists():
            artifact.add_file(str(dependency))
    run.log_artifact(artifact)

    table = wandb.Table(columns=columns,
                        data=[[row.get(c) for c in columns] for row in rows])
    run.log({"recovery_vs_k": table})
    # Logged as plain series too, so k is a real x-axis in W&B rather than a table column.
    for row in rows:
        run.log({f"sweep/{m}": v for m, v in row.items() if m != "k"} | {"sweep/k": row["k"]})
    if args.results_csv and Path(args.results_csv).exists():
        artifact = wandb.Artifact(name=f"sweep_k_{args.ds_name}", type="results")
        artifact.add_file(str(args.results_csv))
        run.log_artifact(artifact)
    run.finish()


def main():
    args = parse_args()
    k_grid = parse_k_grid(args.k_grid)
    max_k = k_grid[-1]

    # The pool is shared by every point in the grid, so it has to be big enough for the
    # largest one. Sizing it to max_k (rather than per-k) is also what makes the grid a
    # controlled comparison: every k samples from the same set of candidate frames.
    if args.pool_size is None:
        args.pool_size = max_k
    elif args.pool_size < max_k:
        raise ValueError(f"--pool-size {args.pool_size} is smaller than the largest k "
                         f"({max_k}); the featurizer would pad by duplicating conformers.")

    traj_h5 = args.traj_h5 or Path(dataset_splits.default_h5(args.ds_name))
    index_csv = args.index_csv or Path(dataset_splits.default_index_csv(args.ds_name))
    # Pool size is baked into the .pt files, so a sweep's pool must not be confused with the
    # pool_size=10 one train.py writes -- keep them in separate directories.
    processed_dir = args.processed_dir or (THIS_DIR / f"processed_data_{args.ds_name}_pool{args.pool_size}")
    args.run_name = args.run_name or f"DynamicMPNN_{args.ds_name}_ksweep"
    args.results_csv = args.results_csv or (THIS_DIR / f"sweep_k_{args.ds_name}.csv")
    group_name = f"{args.run_name}_{os.getenv('CONDOR_CLUSTER', 'local')}"

    wandb.login(key=os.getenv("WANDB_KEY"))

    train_pdbs, val_pdbs, test_pdbs = dataset_splits.get_splits(
        args.ds_name, index_csv, val_fold=args.val_fold, test_fold=args.test_fold)
    pdb_lists = {
        "train": filter_by_length(train_pdbs, traj_h5, args.max_length),
        "val": filter_by_length(val_pdbs, traj_h5, args.max_length),
        "test": filter_by_length(test_pdbs, traj_h5, args.max_length),
    }
    logger.info(f"{args.ds_name}: {len(pdb_lists['train'])} train / {len(pdb_lists['val'])} "
                f"val / {len(pdb_lists['test'])} test proteins (val fold {args.val_fold}, "
                f"test fold {args.test_fold}, max_length={args.max_length})")
    logger.info(f"Sweeping k over {k_grid} against a shared pool of {args.pool_size} "
                f"conformers per protein")

    all_pdbs = pdb_lists["train"] + pdb_lists["val"] + pdb_lists["test"]
    build_processed_dataset(
        pdb_ids=all_pdbs,
        traj_h5_path=traj_h5,
        processed_dir=processed_dir,
        pool_size=args.pool_size,
        course_grain=args.course_grain,
        force_rebuild=args.force_rebuild,
    )
    check_pool_sizes(processed_dir, all_pdbs, max_k)

    # Create the results file before the first point so it exists even if k=grid[0] OOMs
    # immediately. HTCondor's transfer_output_files lists it, and a missing output file
    # holds the job without transferring anything -- including the .err log that says which
    # k ran out of memory, which is the one thing worth having after that failure.
    Path(args.results_csv).touch()

    rows = []
    for k in k_grid:
        logger.info(f"===== sweep point k={k} ({k_grid.index(k) + 1}/{len(k_grid)}) =====")
        try:
            rows.append(train_one_k(k, args, pdb_lists, processed_dir, group_name))
        except torch.cuda.OutOfMemoryError:
            # Memory grows monotonically with k, so the first OOM is the end of the usable
            # grid -- stop and keep the points already scored rather than losing the sweep.
            logger.error(f"k={k} ran out of GPU memory (batch_size "
                         f"{batch_size_for_k(k, args.conf_budget, args.proteins_per_step, args.batch_size)[0]}). "
                         f"This is the VRAM ceiling for this card; stopping the sweep here "
                         f"and keeping k={[r['k'] for r in rows]}. Lower --conf-budget to "
                         f"retry the tail on the same card.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break

        print(json.dumps(rows[-1], indent=2))
        write_results_csv(rows, args.results_csv)

    if not rows:
        raise RuntimeError("No sweep point completed.")

    log_summary(rows, args, k_grid, group_name)

    print(f"\n{'k':>4} {'top1':>8} {'top5':>8} {'perp':>8}  (held-out test)")
    for row in rows:
        print(f"{row['k']:>4} {row.get('test_top1_acc_mean', float('nan')):>8.4f} "
              f"{row.get('test_top5_acc_mean', float('nan')):>8.4f} "
              f"{row.get('test_perp_mean', float('nan')):>8.4f}")


if __name__ == "__main__":
    main()
