"""Fine-tune a *released* DynamicMPNN checkpoint on ATLAS / mdCATH.

The third row of the comparison, between train.py (this architecture from scratch) and
eval_ckpt.py (upstream's weights, zero-shot): start from upstream's published weights
(``DynamicMPNN/checkpoints/single_chain_k{2,3,5}.ckpt``), keep training on our training split,
and score the same two held-out splits with the same metric set.

Everything that makes the released weights usable at all is imported from eval_ckpt.py rather
than redone -- see its docstring for the details:

* `config_from_ckpt` -- the model is built from the checkpoint's OWN config (the legacy pair
  model: 26/17 input scalars, 4+4 layers, representation=ca), not train.py's
  `AR1_single_chain.yaml`, which the weights do not fit.
* `LegacyPairAdapter` -- those weights' featuriser is the codnas *pair* featuriser, so every
  checkpoint here sees k = 2 conformers, whatever its file name says.
* `DecoderNodeBatching` -- realigns `batch.batch` with the pooled decoder logits so
  `run_validation`'s per-protein slicing is right.

What fine-tuning adds on top of that is the training side, matched to how upstream trained
these weights (the checkpoint's own `cfg`: Adam, label smoothing 0.05, GAP ignored,
`in_memory: False`):

* **Training samples are re-featurised on every draw** (train.py's `ResamplingDataset`). Upstream
  trained with `in_memory: False`, so each step saw a fresh conformer pair and fresh
  `noise_scale` coordinate noise. train.py's `in_memory=True` featurises once and freezes both
  for every epoch -- harmless for a from-scratch model that never knew otherwise, but it would
  quietly take away the augmentation these weights were trained under. `--train-pair-selection
  random` (the default) draws a random pair of the pool each time; `tm_min` pins the most
  dissimilar pair.
* **Held-out splits are featurised exactly as eval_ckpt.py does** (`tm_min` pair, no noise,
  frozen), so epoch 0 of the validation curve here reproduces the zero-shot number and every
  later point is directly the fine-tuning gain.
* **The paper's optimisation regime**: Adam at lr 1e-3, no scheduler, and a batch of 32
  proteins per optimizer step. Upstream reached 32 with 2 proteins per GPU across DDP ranks;
  here it is 2 per forward pass, gradients accumulated over 16 (`--effective-batch-size`).
* **Early stopping on val perplexity** (train.py's `EarlyStopping`, patience 10, max 200
  epochs), the protocol every comparison model shares. The best val epoch -- the zero-shot
  epoch -1 included -- is restored before test.

The saved checkpoint carries the same `hyper_parameters['cfg']` layout upstream's do, so
eval_ckpt.py can rescore a fine-tuned run with ``--ckpt weights/<run>.ckpt``, and it reloads
here with ``--ckpt`` to continue fine-tuning.

Usage (both datasets are separate runs, as everywhere else in this comparison):

    ./finetune.sh --ds-name atlas
    ./finetune.sh --ds-name atlas --ckpt DynamicMPNN/checkpoints/single_chain_k5.ckpt --lr 3e-5
"""

import argparse
import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# train.py owns the dataset conversion and the metric loop (and puts lib/, scripts/ and
# DynamicMPNN/src on sys.path); eval_ckpt.py owns loading the released weights.
from train import (  # noqa: E402
    DEFAULT_COURSE_GRAIN,
    DEVICE,
    GAP_TOKEN,
    REPO_ROOT,
    EarlyStopping,
    ResamplingDataset,
    build_processed_dataset,
    filter_by_length,
    load_pretrained_weights,
    run_validation,
    save_weights,
)
from eval_ckpt import (  # noqa: E402
    DEFAULT_CKPT,
    DecoderNodeBatching,
    LegacyPairAdapter,
    build_split_loader,
    check_feature_dims,
    config_from_ckpt,
)

import dataset_splits  # noqa: E402

from dynamicmpnn.datamodules.sampler import safe_collate  # noqa: E402
from dynamicmpnn.types import DISTANCE_EPS  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ds-name", type=str, default="atlas", choices=dataset_splits.DATASETS,
                        help="Which shared dataset to fine-tune/evaluate on; the two are run separately")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                        help="Checkpoint to start from: a released one (default: single_chain_k2.ckpt) "
                             "or a previous finetune.py run's weights/<run>.ckpt")
    parser.add_argument("--traj-h5", type=Path, default=None,
                        help="Trajectory store the conformer ensembles are built from "
                             "(default: the dataset's standard file name)")
    parser.add_argument("--index-csv", type=Path, default=None,
                        help="Split index csv (default: the dataset's standard file name)")
    parser.add_argument("--val-fold", type=int, default=dataset_splits.DEFAULT_VAL_FOLD,
                        help="ATLAS only: cross_val fold used as the validation split")
    parser.add_argument("--test-fold", type=int, default=dataset_splits.DEFAULT_TEST_FOLD,
                        help="ATLAS only: cross_val fold held out as the test split -- never "
                             "trained on, scored once, with the best val epoch's weights")
    parser.add_argument("--processed-dir", type=Path, default=None,
                        help="Where the .pt conformer ensembles live (default: ./processed_data_<ds>, "
                             "shared with train.py and eval_ckpt.py)")
    parser.add_argument("--max-length", type=int, default=dataset_splits.MAX_LENGTH,
                        help="Drop proteins longer than this many residues (0 disables)")
    parser.add_argument("--pool-size", type=int, default=10,
                        help="Conformer pool saved per protein's .pt file; training draws its pairs "
                             "from it")
    parser.add_argument("--course-grain", type=int, default=None,
                        help="time subsampling for the pairwise-TM matrix (default: "
                             f"{DEFAULT_COURSE_GRAIN}, keyed on --ds-name)")
    parser.add_argument("--train-pair-selection", type=str, default="random",
                        choices=("random", "tm_min"),
                        help="Which 2 of the pool each training draw uses: a fresh random pair "
                             "(default) or always the most TM-dissimilar one")
    parser.add_argument("--pair-selection", type=str, default="tm_min", choices=("tm_min", "random"),
                        help="Which 2 of the pool val/test score; tm_min matches eval_ckpt.py")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Proteins per forward pass -- upstream's per-GPU batch")
    parser.add_argument("--effective-batch-size", type=int, default=32,
                        help="Proteins per optimizer step (the paper's batch size); reached by "
                             "accumulating gradients over effective/--batch-size forward passes")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Upper bound on epochs (upstream's trainer max_epochs); early "
                             "stopping normally ends the run well before it")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Stop once val perplexity has not improved on its best for this many "
                             "epochs; 0 disables stopping (the best epoch is still restored)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam lr; the paper's 1e-3, which upstream trained these weights at")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=dataset_splits.SEED)
    parser.add_argument("--no-initial-eval", action="store_true",
                        help="Skip scoring val before the first update (the zero-shot baseline point)")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=Path, default=THIS_DIR / "weights",
                        help="fine-tuned weights land here as <run_name>_<wandb run id>.ckpt")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.effective_batch_size % args.batch_size:
        raise ValueError(f"--effective-batch-size {args.effective_batch_size} is not a multiple "
                         f"of --batch-size {args.batch_size}")
    accum_steps = args.effective_batch_size // args.batch_size
    if args.course_grain is None:
        args.course_grain = DEFAULT_COURSE_GRAIN[args.ds_name]
        logger.info(f"--course-grain defaulted to {args.course_grain} for {args.ds_name}")
    # Seeded for the same reason as train.py. Python's `random` drives the training pair draw;
    # DataLoader re-seeds it per worker from the torch seed, so the draws stay reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint at {args.ckpt}. The released weights ship inside DynamicMPNN.tar.gz "
            "under DynamicMPNN/checkpoints/ -- extract it first (finetune.sh does).")

    traj_h5 = args.traj_h5 or Path(dataset_splits.default_h5(args.ds_name))
    index_csv = args.index_csv or Path(dataset_splits.default_index_csv(args.ds_name))
    ds_tag = dataset_splits.dataset_tag(args.ds_name, traj_h5)
    processed_dir = args.processed_dir or (THIS_DIR / f"processed_data_{ds_tag}")
    run_name = args.run_name or f"DynamicMPNN_finetune_{ds_tag}_{args.ckpt.stem}"

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name=run_name,
        config={
            "ds_name": args.ds_name,
            "traj_h5": str(traj_h5),
            "index_csv": str(index_csv),
            "val_fold": args.val_fold,
            "test_fold": args.test_fold,
            "max_length": args.max_length,
            "pool_size": args.pool_size,
            "course_grain": args.course_grain,
            "batch_size": args.batch_size,
            "effective_batch_size": args.effective_batch_size,
            "epochs": args.epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "ckpt": str(args.ckpt),
            "from_scratch": False,
            "finetune": True,
            # The released checkpoints' featuriser consumes a PAIR regardless of the `k` in
            # their file name (see eval_ckpt.py), so this is the honest ensemble size.
            "k": 2,
            "train_pair_selection": args.train_pair_selection,
            "pair_selection": args.pair_selection,
            "task": f"fine-tuning {args.ckpt.name} for residue recovery on {args.ds_name} "
                    "MD conformer pairs",
        },
    )
    artifact = wandb.Artifact(name="scripts", type="model_file")
    artifact.add_file(os.path.abspath(__file__))
    for dependency in (THIS_DIR / "train.py",
                       THIS_DIR / "eval_ckpt.py",
                       REPO_ROOT / "lib" / "stats_utils.py",
                       REPO_ROOT / "lib" / "residue_classifier_dataset.py",
                       REPO_ROOT / "lib" / "dataset_splits.py",
                       REPO_ROOT / "scripts" / "load_dynamics.py"):
        if dependency.exists():
            artifact.add_file(str(dependency))
    run.log_artifact(artifact)

    train_pdbs, val_pdbs, test_pdbs = dataset_splits.get_splits(
        args.ds_name, index_csv, val_fold=args.val_fold, test_fold=args.test_fold)
    train_pdbs = filter_by_length(train_pdbs, traj_h5, args.max_length)
    val_pdbs = filter_by_length(val_pdbs, traj_h5, args.max_length)
    test_pdbs = filter_by_length(test_pdbs, traj_h5, args.max_length)
    logger.info(f"{args.ds_name}: {len(train_pdbs)} train / {len(val_pdbs)} val / "
                f"{len(test_pdbs)} test proteins (val fold {args.val_fold}, test fold "
                f"{args.test_fold}, max_length={args.max_length})")

    build_processed_dataset(
        pdb_ids=train_pdbs + val_pdbs + test_pdbs,
        traj_h5_path=traj_h5,
        processed_dir=processed_dir,
        pool_size=args.pool_size,
        course_grain=args.course_grain,
        force_rebuild=args.force_rebuild,
    )
    # build_processed_dataset only warns about ids with no trajectory group in the h5 and
    # writes no .pt for them; PTFileDataset torch.loads every id it is given, so drop them here.
    train_pdbs, val_pdbs, test_pdbs = (
        [p for p in ids if (processed_dir / f"{p}.pt").exists()]
        for ids in (train_pdbs, val_pdbs, test_pdbs))

    model_cfg, features_cfg = config_from_ckpt(args.ckpt)
    model = hydra.utils.instantiate(model_cfg).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Strict, as in eval_ckpt.py: a partial load would fine-tune from a partly random init and
    # the result would still look like "fine-tuning".
    load_pretrained_weights(model, args.ckpt)
    model = DecoderNodeBatching(model).to(DEVICE)
    run.log({"params": num_params})

    train_dataset = ResamplingDataset(
        train_pdbs,
        LegacyPairAdapter(hydra.utils.instantiate(features_cfg, split="train", device="cpu",
                                                  distance_eps=DISTANCE_EPS),
                          selection=args.train_pair_selection),
        processed_dir,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training protein survived featurisation.")
    check_feature_dims(train_dataset[0], model_cfg)

    # Same loaders eval_ckpt.py scores with, so the numbers line up with the zero-shot run.
    val_loader, val_dataset, val_pdb_order = build_split_loader(
        val_pdbs, features_cfg, processed_dir, "val", args)
    test_loader, test_dataset, test_pdb_order = build_split_loader(
        test_pdbs, features_cfg, processed_dir, "test", args)

    for split, wanted, kept in (("train", train_pdbs, train_dataset.pdb_codes),
                                ("val", val_pdbs, val_pdb_order),
                                ("test", test_pdbs, test_pdb_order)):
        logger.info(f"{args.ds_name}: featurized {len(kept)}/{len(wanted)} {split} proteins; "
                    f"dropped {sorted(set(wanted) - set(kept))}")
    run.summary["split_counts"] = {"train": len(train_dataset), "val": len(val_dataset),
                                   "test": len(test_dataset)}
    run.summary["val_split_ids"] = list(val_pdb_order)
    run.summary["test_split_ids"] = list(test_pdb_order)

    # Upstream's own loader: `drop_last=True`, `batch_size` proteins per forward pass (their
    # per-GPU batch). The paper's batch of 32 is what their DDP ranks summed to per optimizer
    # step; on one GPU the same step is `accum_steps` forward passes with the gradient averaged
    # over them, which is what DDP's gradient all-reduce computes across ranks.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, collate_fn=safe_collate,
        persistent_workers=args.num_workers > 0,
    )
    if len(train_loader) == 0:
        raise RuntimeError(f"{len(train_dataset)} training proteins is less than one "
                           f"--batch-size {args.batch_size} batch.")
    steps_per_epoch = -(-len(train_loader) // accum_steps)
    logger.info(f"{len(train_loader)} forward passes of {args.batch_size} per epoch, accumulated "
                f"{accum_steps} at a time -> {steps_per_epoch} optimizer steps of up to "
                f"{args.effective_batch_size} proteins")
    run.summary["steps_per_epoch"] = steps_per_epoch

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Upstream's own task config for these weights: label smoothing 0.05, GAP (20) ignored.
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=GAP_TOKEN)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.save_dir / f"{run_name}_{run.id}.ckpt"
    best_path = args.save_dir / f"{run_name}_{run.id}_best.ckpt"
    run.summary["weights_path"] = save_path.name
    run.summary["best_weights_path"] = best_path.name
    # Upstream's `hyper_parameters['cfg']` layout, so eval_ckpt.py's `config_from_ckpt` (and
    # this script's --ckpt) can rebuild the model from the saved file alone.
    ckpt_cfg = {"model": OmegaConf.to_container(model_cfg, resolve=True),
                "features": OmegaConf.to_container(features_cfg, resolve=True)}

    # Model selection is the protocol every comparison model shares (train.py's
    # EarlyStopping): best per-protein val perplexity, patience 10, best weights restored for
    # test. It stands in for the paper's "until validation sequence recovery plateaued or began
    # to decline", which upstream judged by eye -- the released runs carry no EarlyStopping
    # callback, only a per-epoch ModelCheckpoint on `val/sequence_recovery`.
    stopper = EarlyStopping(args.early_stopping_patience)

    def record(metrics, epoch):
        # `.model` unwraps DecoderNodeBatching, so the kept keys match upstream's `GNN_model.*`.
        if stopper.update(model.model, metrics["val_perp_mean"], epoch):
            # On disk as well, so an evicted job still brings home its best weights.
            save_weights(model.model, optimizer, best_path, epoch, step, model_cfg, features_cfg,
                         args, extra_hparams={"cfg": ckpt_cfg})
            run.log({"best_val_perp_mean": stopper.best_perplexity,
                     "best_val_epoch": stopper.best_epoch, "epoch": epoch})

    step = 0
    # Logged at epoch -1: the pretrained weights before any update, i.e. the zero-shot number
    # on this exact val set, as the origin of the fine-tuning curve. It is also the first
    # candidate for best: if no epoch beats it, test is scored on the released weights.
    if not args.no_initial_eval:
        record(run_validation(model, val_loader, crit, val_pdb_order, -1, run,
                              num_params=num_params), -1)

    stopped_epoch = args.epochs - 1
    for epoch in range(args.epochs):
        model.train()

        total_loss, window_loss = 0.0, 0.0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Train")):
            batch = batch.to(DEVICE)
            # The epoch's last window can be short; average over the passes it actually has.
            window_start = batch_idx - batch_idx % accum_steps
            window_size = min(accum_steps, len(train_loader) - window_start)

            logits, valid_mask = model(batch)
            loss = crit(logits[valid_mask], batch.seq[valid_mask])
            (loss / window_size).backward()
            total_loss += loss.item()
            window_loss += loss.item() / window_size

            if batch_idx - window_start + 1 == window_size:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                run.log({"train_loss": window_loss, "epoch": epoch, "step": step})
                window_loss = 0.0

        logger.info(f"Epoch {epoch}: train loss {total_loss / len(train_loader):.4f}")

        metrics = run_validation(model, val_loader, crit, val_pdb_order, epoch, run,
                                 num_params=num_params)
        print(f"Val Perplexity: {metrics['val_perp_mean']:.4f} \\pm {metrics['val_perp_std']:.4f}")
        print(f"Top-1 Recovery: {metrics['val_top1_acc_mean']:.4f} \\pm {metrics['val_top1_acc_std']:.4f}")

        # Overwritten every epoch so an evicted job still brings home its latest weights.
        # `.model` unwraps DecoderNodeBatching, so the keys match upstream's `GNN_model.*`.
        save_weights(model.model, optimizer, save_path, epoch, step, model_cfg, features_cfg,
                     args, extra_hparams={"cfg": ckpt_cfg})
        logger.info(f"Saved epoch {epoch} weights to {save_path}")
        record(metrics, epoch)
        if stopper.should_stop(epoch):
            run.log({"early_stopped_epoch": epoch, "epoch": epoch})
            stopped_epoch = epoch
            break

    run.summary.update({"best_val_perp_mean": stopper.best_perplexity,
                        "best_val_epoch": stopper.best_epoch, "stopped_epoch": stopped_epoch})

    # Test is scored exactly once, with the weights val selected -- the test split never
    # influenced either the weights or the choice of epoch.
    stopper.restore(model.model)
    logger.info(f"Scoring the held-out test split ({len(test_pdb_order)} proteins)")
    test_metrics = run_validation(model, test_loader, crit, test_pdb_order, stopper.best_epoch, run,
                                  num_params=num_params, split_name="test")
    for topk in (1, 5, 10):
        print(f"Test Top-{topk} Recovery: {test_metrics[f'test_top{topk}_acc_mean']:.4f} "
              f"\\pm {test_metrics[f'test_top{topk}_acc_std']:.4f}")
    run.summary.update({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})

    run.finish()


if __name__ == "__main__":
    main()
