"""Zero-shot evaluation of a *released* DynamicMPNN checkpoint on ATLAS / mdCATH.

train.py trains this architecture from scratch. This script does the other half of the
comparison: it takes one of upstream's published weights
(``DynamicMPNN/checkpoints/single_chain_k{2,3,5}.ckpt``) and scores our held-out splits with
it, no training at all. Same conformer ensembles, same splits, same metric set
(`lib.stats_utils.ResidueMetrics`), so the numbers drop straight into the same table as the
from-scratch run and as comparisons/{MapDiff,PiFold}.

Two things make this more than "load weights, call run_validation":

1. **The released checkpoints are NOT the config train.py instantiates.** Their own
   `hyper_parameters` record the *legacy* pair architecture -- `AutoregressiveMultiGNNv1`
   with node_in_dim (26, 2) / edge_in_dim (17, 1), 4 encoder + 4 decoder layers, no pooled
   encoder layers, and `representation: ca` (CA only, no `sequence` node feature and no
   `edge_type` edge feature). train.py's `AR1_single_chain.yaml` is 27/18 scalars, 8+4 layers
   and `pooling_strategy: single_chain_k`. Rather than hard-code either, `config_from_ckpt`
   reads the model/features config out of the checkpoint itself and rewrites its
   `dynamicprot_single.src.*` targets onto the vendored `dynamicmpnn.*` classes -- so k2, k3
   and k5 all work, and a new checkpoint with a different config does too.

2. **Those checkpoints' featuriser is the codnas *pair* featuriser.** `ProteinGraphFeaturiser`
   (as opposed to `ProteinGraphFeaturiserSingleChain`, which train.py uses) takes exactly TWO
   conformations no matter what `features.k` says -- its `get_entries` is hard-coded to return
   a pair, and `add_node_features` / `_flatten_for_pyg` index conformer 1 directly. So
   "single_chain_k5" does not mean this checkpoint sees 5 conformers here; it sees 2. It also
   keys `pyg_dict` by ``member.split('_')[0].upper()``, i.e. it expects codnas' ``<PDB>_<chain>``
   cluster members, whereas our .pt files key by ``<pdb>_frame<t>`` (build_ensemble). Feeding
   ours to it unadapted silently resolves both members to the same key -- the same conformer
   twice. `LegacyPairAdapter` below picks the pair out of our pool and re-presents it in the
   naming the featuriser expects, so the two frames stay distinct.

Everything else -- the hdf5 -> .pt conformer-pool conversion, the length filter, the splits
and the scored metrics -- is imported from train.py unchanged, which is the point: the only
difference between this script's numbers and the from-scratch run's is the weights.

Usage (both datasets are separate runs, as everywhere else in this comparison):

    ./eval_ckpt.sh --ds-name atlas
    ./eval_ckpt.sh --ds-name mdcath --ckpt DynamicMPNN/checkpoints/single_chain_k5.ckpt
"""

import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.data import Data

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# train.py owns the dataset conversion and the metric loop; importing it also puts
# lib/, scripts/ and DynamicMPNN/src on sys.path.
from train import (  # noqa: E402
    DEVICE,
    REPO_ROOT,
    build_processed_dataset,
    filter_by_length,
    load_pretrained_weights,
    run_validation,
)

import dataset_splits  # noqa: E402

from dynamicmpnn.datamodules.pt_dataset import PTFileDataset  # noqa: E402
from dynamicmpnn.datamodules.sampler import safe_collate  # noqa: E402
from dynamicmpnn.types import DISTANCE_EPS  # noqa: E402

DEFAULT_CKPT = THIS_DIR / "DynamicMPNN" / "checkpoints" / "single_chain_k2.ckpt"

# The released checkpoints were trained in upstream's pre-rename package. Their configs still
# point at it, so map the targets we know onto the vendored tree. `AutoregressiveMultiGNNv1`
# was renamed to `DynamicMPNN` (one class now covers both pooling strategies); the featurisers
# kept their names.
LEGACY_PACKAGE_PREFIX = "dynamicprot_single.src."
LEGACY_CLASS_RENAMES = {"AutoregressiveMultiGNNv1": "DynamicMPNN"}
# `features.k` is recorded on every released checkpoint but `ProteinGraphFeaturiser.__init__`
# has no such parameter -- see the module docstring: that featuriser is a pair featuriser.
UNSUPPORTED_FEATURE_KEYS = ("k",)


def retarget(target: str) -> str:
    """Rewrite a checkpoint's `_target_` onto the vendored `dynamicmpnn` package."""
    if not target.startswith(LEGACY_PACKAGE_PREFIX):
        return target
    module, _, cls = target[len(LEGACY_PACKAGE_PREFIX):].rpartition(".")
    return f"dynamicmpnn.{module}.{LEGACY_CLASS_RENAMES.get(cls, cls)}"


def config_from_ckpt(ckpt_path: Path) -> tuple:
    """Pull the model and features configs the checkpoint was trained with out of it.

    Lightning stores the whole hydra config under `hyper_parameters["cfg"]`, which is the only
    authoritative record of the architecture the weights fit -- the yaml files shipped in
    `configs/` describe the *current* single-chain model, not these checkpoints.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters")
    if not hparams or "cfg" not in hparams:
        raise RuntimeError(
            f"{ckpt_path.name} carries no `hyper_parameters['cfg']`, so the architecture it "
            "fits cannot be recovered. Point --ckpt at a checkpoint saved by upstream's "
            "Lightning trainer.")
    cfg = OmegaConf.create(hparams["cfg"])
    # Hydra saved this config in struct mode, and OmegaConf.create keeps that flag, which
    # forbids deleting (or adding) keys. Unlock it so the unsupported keys below can be dropped.
    OmegaConf.set_struct(cfg, False)

    model_cfg = cfg.model
    model_cfg._target_ = retarget(model_cfg._target_)

    features_cfg = cfg.features
    features_cfg._target_ = retarget(features_cfg._target_)
    for key in UNSUPPORTED_FEATURE_KEYS:
        if key in features_cfg:
            logger.info(f"Dropping features.{key}={features_cfg[key]} -- "
                        f"{features_cfg._target_.rsplit('.', 1)[-1]} takes no such argument.")
            del features_cfg[key]

    logger.info(f"{ckpt_path.name}: model {model_cfg._target_} "
                f"node_in_dim={list(model_cfg.node_in_dim)} edge_in_dim={list(model_cfg.edge_in_dim)}; "
                f"features {features_cfg._target_} representation={features_cfg.representation}")
    return model_cfg, features_cfg


class LegacyPairAdapter:
    """Feed one of our conformer-pool `.pt` files to the pair (codnas) featuriser.

    `ProteinGraphFeaturiser` resolves each cluster member ``<PDB>_<chain>`` to
    ``pyg_dict[PDB.upper()]`` and reads `chains` / `homo_idx` / `mask_seq` / `homomer_mapping`
    off each conformer -- codnas fields that `train.frame_to_pyg_data` has no reason to write,
    since `ProteinGraphFeaturiserSingleChain` synthesises them itself. Both of our frames also
    share one pdb id, so the unadapted featuriser would collapse the pair onto a single
    conformer (its ``pdb_ids[0] == pdb_ids[1]`` branch) and score a "2-conformer" model on one
    structure.

    So: choose two frames out of the pool, hand each a distinct synthetic pdb id and the
    single-chain field set (one chain, sequence fully masked -- the design target must not
    reach the encoder), and let the featuriser run untouched from there. That keeps every
    feature the checkpoint expects computed by upstream's own code.
    """

    def __init__(self, featuriser, selection: str = "tm_min"):
        self.featuriser = featuriser
        self.selection = selection
        self.kept_pdb_codes = []

    def select_pair(self, protein) -> list:
        """Indices of the two pool frames to score."""
        n = len(protein.cluster_members)
        tm_scores = getattr(protein, "tm_scores", None)

        if self.selection == "random" or tm_scores is None:
            if self.selection != "random":
                logger.warning("No tm_scores on this ensemble; falling back to a random pair.")
            return sorted(random.sample(range(n), 2))

        # `tm_min`: the most structurally distinct pair in the pool. Deterministic, which
        # matters for a zero-shot number -- there is no training run to average the draw out.
        # It is also the same criterion build_ensemble's farthest-point sampling used to pick
        # the pool, just applied pairwise.
        #
        # A NaN TM (compute_tm_score can return one for a degenerate frame) becomes 1.0, i.e.
        # "identical", so it is never the pair we pick.
        tm = torch.nan_to_num(tm_scores.float(), nan=1.0)
        tm.fill_diagonal_(float("inf"))
        flat = int(torch.argmin(tm))
        return sorted((flat // n, flat % n))

    def as_legacy_conformer(self, conf) -> Data:
        """One pool frame -> the field set `stack_conformations` reads off a codnas conformer."""
        num_residues = conf.residue_type.shape[0]
        return Data(
            coords=conf.coords.clone(),
            residue_type=conf.residue_type.clone(),
            residue_index=conf.residue_index.clone(),
            # One chain, one homomer group -- these only exist to let the featuriser group
            # residues, and a single-chain trajectory frame has exactly one group.
            chains=torch.zeros(num_residues, dtype=torch.long),
            homo_idx=torch.zeros(num_residues, dtype=torch.long),
            # Never expose the ground-truth residue identity to the encoder. (The checkpoints'
            # feature list has no `sequence` node feature anyway, so this is belt-and-braces.)
            mask_seq=torch.zeros(num_residues, dtype=torch.bool),
            homomer_mapping={"A": 0},
        )

    def __call__(self, protein, pdb_code: str = None):
        members = list(protein.cluster_members)
        if len(members) < 2:
            logger.warning(f"Skipping {pdb_code}: pool holds {len(members)} conformer(s), need 2.")
            return None

        first, second = self.select_pair(protein)
        confs = [self.as_legacy_conformer(protein.pyg_dict[members[i]]) for i in (first, second)]

        # ``FRAME<i>_A``: distinct "pdb" per conformer so get_entries keeps them apart, chain
        # "A" so stack_conformations' homomer_mapping lookup resolves.
        shim = SimpleNamespace(
            pyg_dict={f"FRAME{i}": conf for i, conf in zip((first, second), confs)},
            cluster_members=[f"FRAME{i}_A" for i in (first, second)],
        )
        result = self.featuriser(shim, pdb_code=pdb_code)
        if result is not None:
            self.kept_pdb_codes.append(pdb_code)
        return result

    def __getattr__(self, name):
        # Guarded so an attribute lookup before `featuriser` is set (unpickling in a DataLoader
        # worker asks for `__setstate__` first) raises AttributeError instead of recursing.
        if name == "featuriser":
            raise AttributeError(name)
        return getattr(self.featuriser, name)


def check_feature_dims(sample, model_cfg) -> None:
    """Fail loudly if the featuriser's output width is not what the weights expect.

    A dim mismatch otherwise surfaces as an opaque matmul error deep inside the first GVP, and
    a *silent* one is worse still -- see train.py's NM_TO_ANGSTROM note, where geometrically
    wrong-but-correctly-shaped features scored at chance and looked like a real result.
    """
    for name, got, want in (("node_s", sample.node_s.shape[-1], int(model_cfg.node_in_dim[0])),
                            ("node_v", sample.node_v.shape[-2], int(model_cfg.node_in_dim[1])),
                            ("edge_s", sample.edge_s.shape[-1], int(model_cfg.edge_in_dim[0])),
                            ("edge_v", sample.edge_v.shape[-2], int(model_cfg.edge_in_dim[1]))):
        if got != want:
            raise RuntimeError(
                f"Featurised {name} has width {got} but the checkpoint's model expects {want}. "
                "The checkpoint's features config and its model config disagree; check what "
                "`config_from_ckpt` pulled out of it.")


class DecoderNodeBatching(torch.nn.Module):
    """Make `batch.batch` line up with the logits before `run_validation` reads it.

    In the single-chain path train.py uses, a residue is ONE graph node carrying k
    conformations, so `batch.batch` (length = residues) indexes the logits directly. The pair
    path these checkpoints use flattens the two conformers into separate nodes, so the encoder
    graph has 2 nodes per residue while the decoder -- and therefore `seq`, `valid_mask` and
    the logits -- has one, pooled over the pair. `run_validation` would then slice
    `logits[batch.batch == p_idx]` with a mask twice as long as the logits.

    `num_decoder_nodes` (set per protein by the featuriser) is the pooled count, so rewrite
    `batch.batch` from it after the forward pass. run_validation re-reads the attribute, so it
    sees the corrected segmentation.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        logits, valid_mask = self.model(batch)

        counts = batch.num_decoder_nodes
        if not torch.is_tensor(counts):
            counts = torch.tensor([counts], device=logits.device)
        counts = counts.reshape(-1).to(logits.device)
        batch.batch = torch.repeat_interleave(
            torch.arange(counts.numel(), device=logits.device), counts)
        if batch.batch.numel() != logits.shape[0]:
            raise RuntimeError(
                f"num_decoder_nodes sums to {batch.batch.numel()} but the model returned "
                f"{logits.shape[0]} logits; the per-protein grouping would be wrong.")
        return logits, valid_mask


def build_split_loader(pdb_codes, features_cfg, processed_dir, split, args):
    """One held-out split -> (loader, dataset, the pdb ids that survived featurisation)."""
    featuriser = LegacyPairAdapter(
        hydra.utils.instantiate(features_cfg, split=split, device="cpu",
                                distance_eps=DISTANCE_EPS),
        selection=args.pair_selection,
    )
    dataset = PTFileDataset(
        pdb_codes=pdb_codes,
        cfg_features=featuriser,
        processed_dir=processed_dir,
        split=split,
        in_memory=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=safe_collate)
    # `kept_pdb_codes` is recorded in featurisation order, which is the order the loader (with
    # shuffle=False) hands batches back -- so run_validation can label each design.
    return loader, dataset, featuriser.kept_pdb_codes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ds-name", type=str, default="atlas", choices=dataset_splits.DATASETS,
                        help="Which shared dataset to score; the two are run separately")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                        help="Released checkpoint to score (default: single_chain_k2.ckpt)")
    parser.add_argument("--traj-h5", type=Path, default=None,
                        help="Trajectory store the conformer ensembles are built from "
                             "(default: the dataset's standard file name)")
    parser.add_argument("--index-csv", type=Path, default=None,
                        help="Split index csv (default: the dataset's standard file name)")
    parser.add_argument("--val-fold", type=int, default=dataset_splits.DEFAULT_VAL_FOLD,
                        help="ATLAS only: cross_val fold used as the validation split")
    parser.add_argument("--test-fold", type=int, default=dataset_splits.DEFAULT_TEST_FOLD,
                        help="ATLAS only: cross_val fold used as the test split")
    parser.add_argument("--splits", type=str, default="val,test",
                        help="Which held-out splits to score. Nothing here is trained on, so "
                             "unlike train.py both are scored in one pass.")
    parser.add_argument("--processed-dir", type=Path, default=None,
                        help="Where the .pt conformer ensembles live (default: "
                             "./processed_data_<ds>, i.e. train.py's -- they are reused as is)")
    parser.add_argument("--max-length", type=int, default=dataset_splits.MAX_LENGTH,
                        help="Drop proteins longer than this many residues (0 disables)")
    parser.add_argument("--pool-size", type=int, default=10,
                        help="Conformer pool saved per protein, if the ensembles must be built. "
                             "Only 2 of the pool are ever scored -- see the module docstring.")
    parser.add_argument("--course-grain", type=int, default=25,
                        help="time subsampling for the pairwise-TM pass")
    parser.add_argument("--pair-selection", type=str, default="tm_min", choices=("tm_min", "random"),
                        help="Which 2 of the pool to score: the most TM-dissimilar pair "
                             "(deterministic, the default) or a seeded random pair")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=dataset_splits.SEED)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Print the metrics but do not log a W&B run")
    return parser.parse_args()


def main():
    args = parse_args()
    # Seeded for the same reason train.py seeds: with --pair-selection random the featuriser's
    # draw decides which conformers the held-out set is *made of*, and in_memory=True freezes
    # that draw for the whole run.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint at {args.ckpt}. The released weights ship inside DynamicMPNN.tar.gz "
            "under DynamicMPNN/checkpoints/ -- extract it first (eval_ckpt.sh does).")

    traj_h5 = args.traj_h5 or Path(dataset_splits.default_h5(args.ds_name))
    index_csv = args.index_csv or Path(dataset_splits.default_index_csv(args.ds_name))
    ds_tag = dataset_splits.dataset_tag(args.ds_name, traj_h5)
    processed_dir = args.processed_dir or (THIS_DIR / f"processed_data_{ds_tag}")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    run_name = args.run_name or f"DynamicMPNN_zeroshot_{ds_tag}_{args.ckpt.stem}"

    run = None
    if not args.no_wandb:
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
                "seed": args.seed,
                "ckpt": str(args.ckpt),
                "zero_shot": True,
                "epochs": 0,
                # The released checkpoints' featuriser consumes a PAIR regardless of the `k` in
                # their file name, so this is the honest ensemble size for this run.
                "k": 2,
                "pair_selection": args.pair_selection,
                "task": f"zero-shot residue recovery on {args.ds_name} MD conformer pairs "
                        f"({args.ckpt.name}, no fine-tuning)",
            },
        )
        artifact = wandb.Artifact(name="scripts", type="model_file")
        artifact.add_file(os.path.abspath(__file__))
        for dependency in (THIS_DIR / "train.py",
                           REPO_ROOT / "lib" / "stats_utils.py",
                           REPO_ROOT / "lib" / "residue_classifier_dataset.py",
                           REPO_ROOT / "lib" / "dataset_splits.py",
                           REPO_ROOT / "scripts" / "load_dynamics.py"):
            if dependency.exists():
                artifact.add_file(str(dependency))
        run.log_artifact(artifact)

    train_pdbs, val_pdbs, test_pdbs = dataset_splits.get_splits(
        args.ds_name, index_csv, val_fold=args.val_fold, test_fold=args.test_fold)
    split_pdbs = {"train": train_pdbs, "val": val_pdbs, "test": test_pdbs}
    unknown = set(splits) - set(split_pdbs)
    if unknown:
        raise ValueError(f"Unknown split(s) {sorted(unknown)}; pick from {sorted(split_pdbs)}.")

    scored_pdbs = {s: filter_by_length(split_pdbs[s], traj_h5, args.max_length) for s in splits}
    logger.info(f"{args.ds_name} zero-shot: " +
                " / ".join(f"{len(v)} {s}" for s, v in scored_pdbs.items()) +
                f" proteins (val fold {args.val_fold}, test fold {args.test_fold}, "
                f"max_length={args.max_length})")

    # Reuses train.py's cache in --processed-dir untouched when it is already there: the
    # ensembles do not depend on the model, so a zero-shot run and a from-scratch run score
    # literally the same .pt files.
    build_processed_dataset(
        pdb_ids=[p for s in splits for p in scored_pdbs[s]],
        traj_h5_path=traj_h5,
        processed_dir=processed_dir,
        pool_size=args.pool_size,
        course_grain=args.course_grain,
        force_rebuild=args.force_rebuild,
    )

    model_cfg, features_cfg = config_from_ckpt(args.ckpt)
    model = hydra.utils.instantiate(model_cfg).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Strict: a partial load here is a silently random-initialised model, and there is no
    # training pass afterwards to make that obvious.
    load_pretrained_weights(model, args.ckpt)
    model = DecoderNodeBatching(model).to(DEVICE)
    model.eval()
    if run:
        run.log({"params": num_params})

    # Unused by run_validation's scoring (it builds its own plain CrossEntropyLoss so the
    # perplexity stays comparable with the sheaf model's), but part of its signature.
    crit = torch.nn.CrossEntropyLoss()

    all_metrics = {}
    for split in splits:
        loader, dataset, pdb_order = build_split_loader(
            scored_pdbs[split], features_cfg, processed_dir, split, args)
        if len(dataset) == 0:
            logger.warning(f"No {split} protein survived featurisation; skipping it.")
            continue
        check_feature_dims(dataset[0], model_cfg)

        logger.info(f"{args.ds_name}: featurized {len(pdb_order)}/{len(scored_pdbs[split])} "
                    f"{split} proteins; dropped {sorted(set(scored_pdbs[split]) - set(pdb_order))}")
        if run:
            run.summary[f"{split}_split_ids"] = list(pdb_order)

        metrics = run_validation(model, loader, crit, pdb_order, epoch=0, run=run,
                                 num_params=num_params, split_name=split)
        all_metrics.update(metrics)
        print(f"[{args.ckpt.name} zero-shot, {args.ds_name}/{split}]")
        print(f"  Perplexity:      {metrics[f'{split}_perp_mean']:.4f} \\pm {metrics[f'{split}_perp_std']:.4f}")
        for topk in (1, 5, 10):
            print(f"  Top-{topk} Recovery:  {metrics[f'{split}_top{topk}_acc_mean']:.4f} "
                  f"\\pm {metrics[f'{split}_top{topk}_acc_std']:.4f}")

    if run:
        run.summary["split_counts"] = {s: len(scored_pdbs[s]) for s in splits}
        run.summary.update({k: v for k, v in all_metrics.items() if isinstance(v, (int, float))})
        run.finish()


if __name__ == "__main__":
    main()
