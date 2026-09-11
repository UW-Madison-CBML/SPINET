"""Fine-tune / cross-validate DynamicMPNN's single-chain (k-conformer) checkpoints on ATLAS.

Pipeline (see README.md for the full reasoning):
  1. For every pdb in atlas_cross_val_index.csv, pull its trajectory out of the ATLAS hdf5
     store, compute a pairwise (Kabsch) CA/backbone RMSD matrix, and farthest-point-sample a
     small pool of structurally distinct frames.
  2. Repack that pool into DynamicMPNN's `.pt` schema (`pyg_dict` + `cluster_members`), which is
     exactly what `ProteinGraphFeaturiserSingleChain` / `PTFileDataset` expect.
  3. Train using fold 0 of atlas_cross_val_index.csv as the only validation fold (folds 1-4 train).
"""

import argparse
import sys
from pathlib import Path

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

# `residue_classifier_dataset` / `scrmsd` (lib/) and `load_dynamics` (scripts/) use bare,
# flat imports -- mirror the convention the rest of the repo's CHTC scripts rely on.
for extra_path in (REPO_ROOT / "lib", REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

# DynamicMPNN is normally `pip install -e`'d into the training image; fall back to the
# vendored source tree here so this also works against a plain checkout.
DYNAMICMPNN_SRC = THIS_DIR / "DynamicMPNN" / "src"
if str(DYNAMICMPNN_SRC) not in sys.path:
    sys.path.insert(0, str(DYNAMICMPNN_SRC))

from load_dynamics import BACKBONE_ATOMS  # noqa: E402
from residue_classifier_dataset import ResidueClassifierDataset
from scrmsd import kabsch_rmsd  # noqa: E402

from dynamicmpnn import constants
from dynamicmpnn.datamodules.pt_dataset import PTFileDataset
from dynamicmpnn.datamodules.sampler import safe_collate
from dynamicmpnn.types import (
    BASE_AMINO_ACIDS,
    DISTANCE_EPS,
    STANDARD_AMINO_ACID_MAPPING_3_TO_1,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ATLAS hdf5 stores backbone atoms in this order; DynamicMPNN's "ca_bb" representation
# expects the first 3 atom channels to be N, CA, C (graphein's PROTEIN_ATOMS convention).
NCA_C_ATOM_IDX = [BACKBONE_ATOMS.index(atom) for atom in ("N", "CA", "C")]
GAP_TOKEN = len(BASE_AMINO_ACIDS)      # 20
UNKNOWN_TOKEN = len(BASE_AMINO_ACIDS) + 1  # 21


# --------------------------------------------------------------------------------------
# Step 1-3: ATLAS hdf5 -> DynamicMPNN .pt ensembles
# --------------------------------------------------------------------------------------

def index_h5_groups(h5_file: h5py.File) -> dict:
    """Map pdb_code -> hdf5 group path for every trajectory group in the file."""
    required = ResidueClassifierDataset.REQUIRED_DATASETS
    lookup = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and all(ds in obj for ds in required):
            lookup.setdefault(name.split("/")[0], name)

    h5_file.visititems(visit)
    return lookup


def resolve_group(pdb_id: str, group_lookup: dict):
    for candidate in (pdb_id, pdb_id.upper(), pdb_id.lower()):
        if candidate in group_lookup:
            return group_lookup[candidate]
    return None


def compute_pairwise_rmsd(coords: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
    """Full T x T Kabsch CA/backbone RMSD matrix for a single trajectory.

    Mirrors `build_kabsch_rmsd_index.get_traj_rmsd`, but skips its `mask` argument: within one
    protein's own frames there is no cross-protein padding to mask out, so the plain (unmasked)
    branch of `kabsch_rmsd` is the correct and simpler path here.

    coords: T, num_res, num_atoms, 3
    """
    torch.set_num_threads(1)  # don't let each worker fight the others for cores
    T, num_res, num_atoms, _ = coords.shape
    flat = coords.reshape(T, num_res * num_atoms, 3)

    pair_rmsd = kabsch_rmsd(flat[i_idx], flat[j_idx], mask=None, device="cpu")

    rmsd_matrix = torch.zeros(T, T, dtype=pair_rmsd.dtype)
    rmsd_matrix[i_idx, j_idx] = pair_rmsd
    rmsd_matrix[j_idx, i_idx] = pair_rmsd
    return rmsd_matrix


def farthest_point_sample(rmsd_matrix: torch.Tensor, pool_size: int) -> list:
    """Greedy max-min RMSD selection: same spirit as DynamicMPNN's own TM-dissimilarity
    k-selection (`ProteinGraphFeaturiserSingleChain.compute_sequential_probabilities`),
    just driven by structural RMSD instead of TM-score.
    """
    T = rmsd_matrix.shape[0]
    pool_size = min(pool_size, T)

    flat_idx = torch.argmax(rmsd_matrix).item()
    i0, j0 = divmod(flat_idx, T)
    selected = [i0, j0] if i0 != j0 else [i0]

    while len(selected) < pool_size:
        min_dist_to_selected = rmsd_matrix[selected].min(dim=0).values
        min_dist_to_selected[selected] = -1.0  # never re-pick an already-selected frame
        selected.append(int(torch.argmax(min_dist_to_selected).item()))

    return selected[:pool_size]


def residues_to_type_ids(raw_residues: np.ndarray) -> torch.Tensor:
    """3-letter residue codes (bytes, from the hdf5 `residues` dataset) -> DynamicMPNN's
    BASE_AMINO_ACIDS integer vocabulary (20 = gap, 21 = unknown/non-standard)."""
    ids = []
    for raw in raw_residues:
        code3 = raw.decode().strip()[:3].upper()
        one_letter = STANDARD_AMINO_ACID_MAPPING_3_TO_1.get(code3)
        if one_letter is not None and one_letter in BASE_AMINO_ACIDS:
            ids.append(BASE_AMINO_ACIDS.index(one_letter))
        else:
            ids.append(UNKNOWN_TOKEN)
    return torch.tensor(ids, dtype=torch.long)


def frame_to_pyg_data(frame_coords: np.ndarray, residue_type: torch.Tensor) -> Data:
    """One selected conformer -> the minimal per-conformer fields
    `ProteinGraphFeaturiserSingleChain.stack_conformations` actually reads.

    frame_coords: num_res, num_atoms(ATLAS order), 3
    """
    coords = torch.from_numpy(frame_coords[:, NCA_C_ATOM_IDX, :]).float()  # num_res, 3 (N,CA,C), 3
    residue_index = torch.arange(coords.shape[0], dtype=torch.long)
    return Data(coords=coords, residue_type=residue_type, residue_index=residue_index)


def build_ensemble(
    h5_file: h5py.File,
    group_name: str,
    pdb_id: str,
    pool_size: int,
    course_grain: int,
) -> tuple:
    """Build one pdb's `.pt` contents: a pool of farthest-point-sampled conformers.

    Rather than committing to exactly `k` conformers here, we save a slightly larger pool and
    let `ProteinGraphFeaturiserSingleChain.get_entries` (already implemented in DynamicMPNN)
    subsample `k` of them at train/val time -- exactly how the multi-chain codnas pipeline
    defers pair/subset selection to the featurizer instead of preprocessing.
    """
    coords_ds = h5_file[group_name + "/coordinates"]  # num_res, T, num_atoms, 3

    coarse_coords = torch.from_numpy(coords_ds[:, ::course_grain]).permute(1, 0, 2, 3).float()
    T_coarse = coarse_coords.shape[0]
    i_idx, j_idx = torch.triu_indices(T_coarse, T_coarse, offset=1)

    rmsd_matrix = compute_pairwise_rmsd(coarse_coords, i_idx, j_idx)
    is_flat = bool(rmsd_matrix.numel() and rmsd_matrix.max().item() < 1.0)  # ~1 Angstrom noise floor

    coarse_states = farthest_point_sample(rmsd_matrix, pool_size)
    real_frame_idx = sorted(int(c) * course_grain for c in coarse_states)

    residue_type = residues_to_type_ids(h5_file[group_name + "/residues"][:])
    pool_frames = coords_ds[:, real_frame_idx]  # num_res, pool, num_atoms, 3

    pyg_dict = {}
    for pool_pos, frame_idx in enumerate(real_frame_idx):
        pyg_dict[f"{pdb_id}_frame{frame_idx}"] = frame_to_pyg_data(pool_frames[:, pool_pos], residue_type)

    ensemble = Data(pyg_dict=pyg_dict, cluster_members=list(pyg_dict.keys()))
    return ensemble, is_flat


def build_processed_dataset(
    pdb_ids: list,
    atlas_h5_path: Path,
    processed_dir: Path,
    pool_size: int,
    course_grain: int,
    force_rebuild: bool,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)

    pending = [p for p in pdb_ids if force_rebuild or not (processed_dir / f"{p}.pt").exists()]
    if not pending:
        logger.info(f"All {len(pdb_ids)} ensembles already built in {processed_dir}, skipping.")
        return
    logger.info(f"Building {len(pending)}/{len(pdb_ids)} ensembles into {processed_dir}")

    flat_trajectories = []
    with h5py.File(atlas_h5_path, "r") as h5_file:
        group_lookup = index_h5_groups(h5_file)

        for pdb_id in tqdm(pending, desc="Building ATLAS ensembles"):
            group_name = resolve_group(pdb_id, group_lookup)
            if group_name is None:
                logger.warning(f"{pdb_id}: no matching trajectory group in {atlas_h5_path}, skipping.")
                continue

            ensemble, is_flat = build_ensemble(h5_file, group_name, pdb_id, pool_size, course_grain)
            torch.save(ensemble, processed_dir / f"{pdb_id}.pt")
            if is_flat:
                flat_trajectories.append(pdb_id)

    if flat_trajectories:
        logger.warning(
            f"{len(flat_trajectories)} trajectories have near-flat RMSD (< 1.0 A): "
            f"{flat_trajectories} -- these are an easy win for any model, keep in mind when "
            "comparing recovery numbers."
        )


# --------------------------------------------------------------------------------------
# Step 4: fold 0 held out for validation, folds 1-4 for training
# --------------------------------------------------------------------------------------

def get_fold_split(atlas_index: pd.DataFrame):
    val_pdbs = atlas_index.loc[atlas_index["cross_val"] == 0, "pdb"].tolist()
    train_pdbs = atlas_index.loc[atlas_index["cross_val"] != 0, "pdb"].tolist()
    return train_pdbs, val_pdbs


def load_pretrained_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = "GNN_model."
    stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing:
        logger.warning(f"Missing keys loading {ckpt_path.name}: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys loading {ckpt_path.name}: {unexpected}")
    logger.info(f"Loaded pretrained weights from {ckpt_path}")


def top_k_acc(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    if logits.shape[0] == 0:
        return float("nan")
    k = min(k, logits.shape[-1])
    top_k = logits.topk(k, dim=-1).indices
    correct = (top_k == targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def run_validation(model, val_loader, crit):
    model.eval()

    val_losses, val_acc_top_1, val_acc_top_5, val_acc_top_10 = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = batch.to(DEVICE)
            logits, valid_mask = model(batch)
            target = batch.seq

            num_proteins_in_batch = batch.batch.max().item() + 1
            for p_idx in range(num_proteins_in_batch):
                protein_mask = (batch.batch == p_idx) & valid_mask
                if not protein_mask.any():
                    continue

                masked_logits = logits[protein_mask]
                masked_seq = target[protein_mask]
                loss = crit(masked_logits, masked_seq).item()

                val_losses.append(loss)
                val_acc_top_1.append(top_k_acc(masked_logits, masked_seq, 1))
                val_acc_top_5.append(top_k_acc(masked_logits, masked_seq, 5))
                val_acc_top_10.append(top_k_acc(masked_logits, masked_seq, 10))

    val_perps = np.exp(val_losses)
    return {
        "ppl_mean": np.mean(val_perps), "ppl_std": np.std(val_perps),
        "top1_mean": np.mean(val_acc_top_1), "top1_std": np.std(val_acc_top_1),
        "top5_mean": np.mean(val_acc_top_5), "top5_std": np.std(val_acc_top_5),
        "top10_mean": np.mean(val_acc_top_10), "top10_std": np.std(val_acc_top_10),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-h5", type=Path, default=Path("atlas_data.h5"))
    parser.add_argument("--cross-val-csv", type=Path, default=THIS_DIR / "atlas_cross_val_index.csv")
    parser.add_argument("--processed-dir", type=Path, default=THIS_DIR / "processed_data")
    parser.add_argument("--k", type=int, default=5, help="conformers sampled per protein at train/val time")
    parser.add_argument("--pool-size", type=int, default=10, help="conformer pool saved per protein's .pt file")
    parser.add_argument("--course-grain", type=int, default=25, help="time subsampling for the RMSD matrix")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-scratch", action="store_true", help="skip loading a pretrained checkpoint")
    parser.add_argument("--ckpt", type=Path, default=None, help="override the default single_chain_k{k}.ckpt")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    atlas_index = pd.read_csv(args.cross_val_csv)

    build_processed_dataset(
        pdb_ids=atlas_index["pdb"].tolist(),
        atlas_h5_path=args.atlas_h5,
        processed_dir=args.processed_dir,
        pool_size=args.pool_size,
        course_grain=args.course_grain,
        force_rebuild=args.force_rebuild,
    )

    train_pdbs, val_pdbs = get_fold_split(atlas_index)
    logger.info(f"Fold 0 held out for validation: {len(train_pdbs)} train / {len(val_pdbs)} val proteins")

    model_cfg = OmegaConf.load(constants.HYDRA_CONFIG_PATH / "model" / "AR1_single_chain.yaml")
    features_cfg = OmegaConf.load(constants.HYDRA_CONFIG_PATH / "features" / "ca_bb_single_chain.yaml")
    features_cfg.k = args.k

    model = hydra.utils.instantiate(model_cfg).to(DEVICE)

    if not args.from_scratch:
        ckpt_path = args.ckpt or (THIS_DIR / "DynamicMPNN" / "checkpoints" / f"single_chain_k{args.k}.ckpt")
        if ckpt_path.exists():
            load_pretrained_weights(model, ckpt_path)
        else:
            logger.warning(f"No checkpoint found at {ckpt_path}, training from scratch instead.")

    # in_memory=True pre-filters proteins the featurizer rejects (too short / no valid coords);
    # PTFileDataset's on-disk path can hand `None` to safe_collate otherwise, which hard-errors.
    train_dataset = PTFileDataset(
        pdb_codes=train_pdbs,
        cfg_features=hydra.utils.instantiate(features_cfg, split="train", device="cpu", distance_eps=DISTANCE_EPS),
        processed_dir=args.processed_dir,
        split="train",
        in_memory=True,
    )
    val_dataset = PTFileDataset(
        pdb_codes=val_pdbs,
        cfg_features=hydra.utils.instantiate(features_cfg, split="val", device="cpu", distance_eps=DISTANCE_EPS),
        processed_dir=args.processed_dir,
        split="val",
        in_memory=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=safe_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=safe_collate,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=GAP_TOKEN)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            logits, valid_mask = model(batch)
            loss = crit(logits[valid_mask], batch.seq[valid_mask])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch}: train loss {total_loss / max(len(train_loader), 1):.4f}")

        metrics = run_validation(model, val_loader, crit)
        print(f"Val Perplexity: {metrics['ppl_mean']:.4f} \\pm {metrics['ppl_std']:.4f}")
        print(f"Top-1 Recovery: {metrics['top1_mean']:.4f} \\pm {metrics['top1_std']:.4f}")
        print(f"Top-5 Recovery: {metrics['top5_mean']:.4f} \\pm {metrics['top5_std']:.4f}")
        print(f"Top-10 Recovery: {metrics['top10_mean']:.4f} \\pm {metrics['top10_std']:.4f}")


if __name__ == "__main__":
    main()
