"""Fine-tune / cross-validate DynamicMPNN's single-chain (k-conformer) checkpoints on our
trajectory datasets.

Unlike MapDiff and PiFold -- which are static-structure models and so train on each
protein's relaxed (deposited) PDB entry -- DynamicMPNN takes an **ensemble of conformers
sampled from the MD trajectory**, and that ensemble input is precisely the thing being
benchmarked. So the input pipeline below deliberately stays on the trajectory store. The
relaxed structures are pulled in only as the scRMSD *reference*, so all three comparison
models are scored against the same ground truth.

Pipeline (see README.md for the full reasoning):
  1. For every protein in the dataset's split index, pull its trajectory out of the hdf5
     store, compute a pairwise (Kabsch) CA/backbone RMSD matrix, and farthest-point-sample a
     small pool of structurally distinct frames.
     * Should replace pairwise matrix calculation with Jens' script.
  2. Repack that pool into DynamicMPNN's `.pt` schema (`pyg_dict` + `cluster_members`), which is
     exactly what `ProteinGraphFeaturiserSingleChain` / `PTFileDataset` expect.
  3. Train on the dataset's training split, validate on its held-out split (lib/dataset_splits.py):
     ATLAS holds out one `cross_val` fold (`--val-fold`, 0 by default -- the fold
     scripts/train_residue_classifier.py and comparisons/{MapDiff,PiFold} also hold out);
     mdCATH trains on its topology split's train+test rows and evaluates on `validation` ONLY.
     Validation is also the only split scored: there is no further held-out test set.
     * Running the other four ATLAS folds is a matter of repeating the job with --val-fold 1..4.

ATLAS and mdCATH are trained and evaluated separately -- one run each (`--ds-name`) --
matching scripts/train_residue_classifier.py.

Logs to a single Weights & Biases run:
per-step train loss, per-epoch validation recovery
(top-1/5/10) + perplexity, an amino-acid confusion matrix, and scRMSD (predicted sequence
folded with ESMFold, Kabsch-RMSD'd against the protein's ground-truth *relaxed* backbone).
"""

import argparse
import os
import random
import sys
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

# `residue_classifier_dataset` / `scrmsd` / `stats_utils` (lib/) and `load_dynamics`
# (scripts/) use bare, flat imports -- mirror the convention the rest of the repo's CHTC
# scripts rely on.
for extra_path in (REPO_ROOT / "lib", REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

# DynamicMPNN is normally `pip install -e`'d into the training image; fall back to the
# vendored source tree here so this also works against a plain checkout.
DYNAMICMPNN_SRC = THIS_DIR / "DynamicMPNN" / "src"
if str(DYNAMICMPNN_SRC) not in sys.path:
    sys.path.insert(0, str(DYNAMICMPNN_SRC))

import dataset_splits  # noqa: E402
import relaxed_pdb
from load_dynamics import BACKBONE_ATOMS
from residue_classifier_dataset import ResidueClassifierDataset
from scrmsd import kabsch_rmsd, load_esmfold, evaluate_scrmsd
from stats_utils import get_confusion_matrix

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

# ATLAS coordinates come from `traj.xyz` (scripts/load_dynamics.py:81), and mdtraj stores xyz
# in NANOMETRES -- process_traj never converts. DynamicMPNN was trained on Angstroms: upstream's
# own val_pt_single_chain files measure N-CA 1.459 / CA-C 1.525 / C(i)-N(i+1) 1.329 A. Feeding nm
# leaves knn topology and all the angle features (alpha/kappa/dihedrals) untouched -- they are
# scale-invariant -- but compresses every `edge_distance` / `rbf_16` channel into the lowest bin,
# which is why single_chain_k2.ckpt scored 35.8% recovery on upstream .pt files and 7.3% (loss
# 116) on ours. It is also why 38/40 trajectories tripped the "near-flat RMSD < 1.0 A" warning:
# that threshold was being compared against nanometre RMSDs.
NM_TO_ANGSTROM = 10.0
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

    No ground-truth backbone is stored alongside: scRMSD scores designs against the
    protein's *relaxed* (deposited) structure, which `load_scrmsd_references` fetches
    separately, not against a trajectory frame.

    frame_coords: num_res, num_atoms (hdf5 order == BACKBONE_ATOMS order, CA/N/C/O), 3
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
    coords_ds = h5_file[group_name + "/coordinates"]  # num_res, T, num_atoms, 3 (NANOMETRES)

    coarse_coords = (torch.from_numpy(coords_ds[:, ::course_grain]).permute(1, 0, 2, 3).float()
                     * NM_TO_ANGSTROM)
    T_coarse = coarse_coords.shape[0]
    i_idx, j_idx = torch.triu_indices(T_coarse, T_coarse, offset=1)

    rmsd_matrix = compute_pairwise_rmsd(coarse_coords, i_idx, j_idx)
    is_flat = bool(rmsd_matrix.numel() and rmsd_matrix.max().item() < 1.0)  # ~1 Angstrom noise floor

    coarse_states = farthest_point_sample(rmsd_matrix, pool_size)
    real_frame_idx = sorted(int(c) * course_grain for c in coarse_states)

    residue_type = residues_to_type_ids(h5_file[group_name + "/residues"][:])
    pool_frames = coords_ds[:, real_frame_idx] * NM_TO_ANGSTROM  # num_res, pool, num_atoms, 3

    pyg_dict = {}
    for pool_pos, frame_idx in enumerate(real_frame_idx):
        pyg_dict[f"{pdb_id}_frame{frame_idx}"] = frame_to_pyg_data(pool_frames[:, pool_pos], residue_type)

    ensemble = Data(pyg_dict=pyg_dict, cluster_members=list(pyg_dict.keys()))
    return ensemble, is_flat


def build_processed_dataset(
    pdb_ids: list,
    traj_h5_path: Path,
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
    with h5py.File(traj_h5_path, "r") as h5_file:
        group_lookup = index_h5_groups(h5_file)

        for pdb_id in tqdm(pending, desc="Building conformer ensembles"):
            group_name = resolve_group(pdb_id, group_lookup)
            if group_name is None:
                logger.warning(f"{pdb_id}: no matching trajectory group in {traj_h5_path}, skipping.")
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


def filter_by_length(pdb_ids: list, traj_h5_path: Path, max_length: int) -> list:
    """Drop proteins longer than the shared residue cutoff (lib/dataset_splits.MAX_LENGTH).

    DynamicMPNN itself has no length limit, but MapDiff does (its IPA node encoder's fixed
    positional-encoding table) and PiFold applies the same cutoff so that the two see the same
    proteins. Applying it here too is what keeps the *held-out set* identical across all three
    -- otherwise DynamicMPNN would be scored on a handful of long proteins nothing else sees.
    """
    if not max_length:
        return list(pdb_ids)

    with h5py.File(traj_h5_path, "r") as h5_file:
        group_lookup = index_h5_groups(h5_file)
        lengths = {}
        for pdb_id in pdb_ids:
            group_name = resolve_group(pdb_id, group_lookup)
            if group_name is not None:
                lengths[pdb_id] = h5_file[group_name + "/residues"].shape[0]

    # Ids with no trajectory group keep a length of 0 and pass here; `build_processed_dataset`
    # is the one that warns about and skips them.
    kept = [p for p in pdb_ids if lengths.get(p, 0) <= max_length]
    dropped = [p for p in pdb_ids if lengths.get(p, 0) > max_length]
    if dropped:
        logger.info(f"Dropped {len(dropped)}/{len(pdb_ids)} proteins over max_length={max_length} "
                     f"residues: {dropped}")
    return kept


def load_scrmsd_references(traj_h5_path: Path, pdb_ids: list, pdb_cache: Path) -> dict:
    """Ground-truth *relaxed* (deposited) structures to score designs against.

    DynamicMPNN designs over the trajectory's residues, but the deposited chain resolves a
    different (and, for an mdCATH domain, larger) set of them, so a plain positional
    comparison would be wrong. `relaxed_pdb.crop_to_reference` aligns the deposited chain
    against the trajectory's own residue sequence and hands back both the cropped
    coordinates and `reference_index` -- the trajectory positions those coordinates
    correspond to -- which is exactly the correspondence `lib.scrmsd.evaluate_scrmsd` wants.

    Returns ``{pdb_id: {'coords': (n, 4, 3) tensor, 'pred_index': (n,) int array,
    'num_residues': int}}``, where `num_residues` is the trajectory's residue count (used to
    check that a batch's design lines up before scoring it).
    """
    reference_seqs = relaxed_pdb.reference_seqs_from_h5(traj_h5_path, pdb_ids)
    records, failures = relaxed_pdb.load_relaxed_structures(
        pdb_ids, cache_dir=str(pdb_cache), reference_seqs=reference_seqs)
    if failures:
        logger.warning(f"{len(failures)}/{len(pdb_ids)} validation proteins have no usable "
                        "deposited structure; they are skipped for scRMSD.")

    references = {}
    for pdb_id, record in records.items():
        if record.get("reference_index") is None:
            # No trajectory residue sequence to align against, so there is no correspondence
            # between the design and the deposited chain -- scoring it would be meaningless.
            logger.warning(f"{pdb_id}: no reference residue sequence in {traj_h5_path}; "
                            "skipped for scRMSD.")
            continue
        references[pdb_id] = {
            "coords": torch.from_numpy(record["coords"]).float(),
            "pred_index": record.get("reference_index"),
            "num_residues": len(reference_seqs.get(pdb_id, "")),
        }
    return references


# --------------------------------------------------------------------------------------
# Step 4: the dataset's held-out split for validation (see lib/dataset_splits.py)
# --------------------------------------------------------------------------------------

def load_pretrained_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = "GNN_model."
    stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not stripped:
        raise RuntimeError(f"No '{prefix}' keys found in {ckpt_path.name}; nothing to load.")
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    # `strict=False` is needed only to tolerate the checkpoint's Lightning wrapper keys; any
    # remaining delta means part of the model is still randomly initialised. Fail here rather
    # than train on it. NB: a clean load is necessary but NOT sufficient -- both released
    # checkpoints load 100% cleanly into this architecture and still score at chance, so
    # validate any new checkpoint by eval recovery, not by key counts (see probe_init.py).
    if missing or unexpected:
        raise RuntimeError(
            f"Partial load of {ckpt_path.name} ({len(stripped)} checkpoint tensors): "
            f"{len(missing)} missing key(s) {sorted(missing)}, "
            f"{len(unexpected)} unexpected key(s) {sorted(unexpected)}. "
            "Reconcile the model config with the checkpoint's hyper_parameters, "
            "or pass --from-scratch."
        )
    logger.info(f"Loaded pretrained weights from {ckpt_path} ({len(stripped)} tensors)")


def top_k_acc(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    if logits.shape[0] == 0:
        return float("nan")
    k = min(k, logits.shape[-1])
    top_k = logits.topk(k, dim=-1).indices
    correct = (top_k == targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


class PdbTrackingFeaturiser:
    """Wraps a DynamicMPNN featuriser to record, in order, which pdb_codes survive
    featurization. `PTFileDataset(..., return_pdb_codes=True, in_memory=True)` can't be used
    for this: once any protein is skipped, `self.data` (filtered) and `self.pdb_codes`
    (unfiltered) go out of sync but `__getitem__` still indexes both with the same
    `real_idx` -- so we track the correspondence ourselves instead.
    """

    def __init__(self, featuriser):
        self.featuriser = featuriser
        self.kept_pdb_codes = []

    def __call__(self, protein, pdb_code=None):
        result = self.featuriser(protein, pdb_code=pdb_code)
        if result is not None:
            self.kept_pdb_codes.append(pdb_code)
        return result

    def __getattr__(self, name):
        return getattr(self.featuriser, name)


def build_confusion_matrix_image(confusion_mat, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_mat.numpy().astype(int),
        display_labels=BASE_AMINO_ACIDS,
    )
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title(title)
    image = wandb.Image(fig)
    plt.close(fig)
    return image


def compute_batch_scrmsd(pred_seqs, refs, esmfold_tokenizer, esmfold_model, device):
    """Fold each protein's designed sequence with ESMFold and Kabsch-RMSD it against that
    protein's ground-truth *relaxed* (deposited) backbone (CA, N, C, O) -- the same
    self-consistency metric comparisons/{MapDiff,PiFold} compute.

    `refs` are `load_scrmsd_references` entries: the design spans the trajectory's residues
    and the reference only the deposited subset of them, so `pred_index` carries the
    correspondence and the reference is scored over all of its own residues.
    """
    return evaluate_scrmsd(
        pred_seqs,
        [ref["coords"] for ref in refs],
        esmfold_tokenizer, esmfold_model,
        pred_indices=[ref["pred_index"] for ref in refs],
        ref_indices=[np.arange(ref["coords"].shape[0]) for ref in refs],
        device=device,
    ).tolist()


def run_validation(model, val_loader, crit, val_pdb_order, scrmsd_refs, esmfold_tokenizer, esmfold_model, epoch, run):
    model.eval()

    val_losses, val_acc_top_1, val_acc_top_5, val_acc_top_10 = [], [], [], []
    scrmsd_values = []
    global_confusion_mat = torch.zeros((len(BASE_AMINO_ACIDS), len(BASE_AMINO_ACIDS)))
    sample_offset = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = batch.to(DEVICE)
            logits, valid_mask = model(batch)
            target = batch.seq

            num_proteins_in_batch = batch.batch.max().item() + 1
            batch_pdb_ids = val_pdb_order[sample_offset:sample_offset + num_proteins_in_batch]
            sample_offset += num_proteins_in_batch

            pred_seqs, batch_refs = [], []
            for p_idx, pdb_id in zip(range(num_proteins_in_batch), batch_pdb_ids):
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

                pred_idx = masked_logits.argmax(dim=-1).cpu()
                gt_idx = masked_seq.cpu()
                keep = gt_idx < len(BASE_AMINO_ACIDS)  # drop rare GAP/UNKNOWN targets, if any slip through
                if keep.any():
                    global_confusion_mat += get_confusion_matrix(gt_idx[keep], pred_idx[keep], len(BASE_AMINO_ACIDS))

                ref = scrmsd_refs.get(pdb_id)
                # `pred_index` indexes the trajectory's residues, so the design has to span
                # all of them -- skip any protein the featurizer's valid_mask trimmed.
                if ref is not None and ref["num_residues"] == pred_idx.shape[0]:
                    pred_seqs.append(''.join(BASE_AMINO_ACIDS[i] for i in pred_idx.tolist()))
                    batch_refs.append(ref)

            if pred_seqs:
                scrmsd_values.extend(compute_batch_scrmsd(
                    pred_seqs, batch_refs, esmfold_tokenizer, esmfold_model, DEVICE))

    val_perps = np.exp(val_losses)
    metrics = {
        "ppl_mean": np.mean(val_perps), "ppl_std": np.std(val_perps),
        "top1_mean": np.mean(val_acc_top_1), "top1_std": np.std(val_acc_top_1),
        "top5_mean": np.mean(val_acc_top_5), "top5_std": np.std(val_acc_top_5),
        "top10_mean": np.mean(val_acc_top_10), "top10_std": np.std(val_acc_top_10),
    }
    if scrmsd_values:
        scrmsd_t = torch.tensor(scrmsd_values)
        metrics["scrmsd_mean"] = scrmsd_t.mean().item()
        metrics["scrmsd_std"] = scrmsd_t.std().item()

    if run:
        log_dict = {f"val_{k}": v for k, v in metrics.items()}
        log_dict["val_aa_confusion_matrix"] = build_confusion_matrix_image(
            global_confusion_mat, "Validation Amino Acid Confusion Matrix")
        log_dict["epoch"] = epoch
        run.log(log_dict)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds-name", type=str, default="atlas", choices=dataset_splits.DATASETS,
                         help="Which shared dataset to train/evaluate on; the two are run separately")
    parser.add_argument("--traj-h5", type=Path, default=None,
                         help="Trajectory store the conformer ensembles are built from "
                              "(default: the dataset's standard file name)")
    parser.add_argument("--index-csv", type=Path, default=None,
                         help="Split index csv (default: the dataset's standard file name)")
    parser.add_argument("--val-fold", type=int, default=dataset_splits.DEFAULT_VAL_FOLD,
                         help="ATLAS only: cross_val fold held out for validation (folds != this train). "
                              "mdCATH ignores it -- its split column is categorical.")
    parser.add_argument("--pdb-cache", type=Path, default=THIS_DIR / "pdb_cache",
                         help="Where relaxed (deposited) RCSB entries are cached for scRMSD")
    parser.add_argument("--processed-dir", type=Path, default=None,
                         help="Where the .pt conformer ensembles live (default: ./processed_data_<ds>)")
    parser.add_argument("--max-length", type=int, default=dataset_splits.MAX_LENGTH,
                         help="Drop proteins longer than this many residues, the cutoff MapDiff and "
                              "PiFold also apply (lib/dataset_splits.MAX_LENGTH). Pass 0 to disable.")
    parser.add_argument("--k", type=int, default=2, help="conformers sampled per protein at train/val time")
    parser.add_argument("--pool-size", type=int, default=10, help="conformer pool saved per protein's .pt file")
    parser.add_argument("--course-grain", type=int, default=25, help="time subsampling for the RMSD matrix")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=dataset_splits.SEED)
    parser.add_argument("--from-scratch", action="store_true", help="skip loading a pretrained checkpoint")
    parser.add_argument("--ckpt", type=Path, default=None, help="override the default single_chain_k{k}.ckpt")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    # `random` and `np.random`, not just torch: DynamicMPNN's featurizer picks which k of each
    # protein's saved conformer pool to use with `random.sample` / `np.random.choice`
    # (ProteinGraphFeaturiserSingleChain.get_entries). Because the datasets below are built
    # with in_memory=True, that draw happens once and then holds for every epoch -- so leaving
    # these unseeded would mean the *validation set itself* differed between runs.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    traj_h5 = args.traj_h5 or Path(dataset_splits.default_h5(args.ds_name))
    index_csv = args.index_csv or Path(dataset_splits.default_index_csv(args.ds_name))
    processed_dir = args.processed_dir or (THIS_DIR / f"processed_data_{args.ds_name}")
    run_name = args.run_name or f"DynamicMPNN_{args.ds_name}"

    # set up wandb
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
            "max_length": args.max_length,
            "k": args.k,
            "pool_size": args.pool_size,
            "course_grain": args.course_grain,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "from_scratch": args.from_scratch,
            "task": f"predicting residues from {args.ds_name} MD conformer ensembles (DynamicMPNN)",
        },
    )

    # WANDB artifact logging
    artifact = wandb.Artifact(name="scripts", type="model_file")
    artifact.add_file(os.path.abspath(__file__))
    for dependency in (REPO_ROOT / "lib" / "scrmsd.py", REPO_ROOT / "lib" / "stats_utils.py",
                       REPO_ROOT / "lib" / "residue_classifier_dataset.py",
                       REPO_ROOT / "lib" / "relaxed_pdb.py", REPO_ROOT / "lib" / "dataset_splits.py",
                       REPO_ROOT / "scripts" / "load_dynamics.py"):
        if dependency.exists():
            artifact.add_file(str(dependency))
    run.log_artifact(artifact)

    train_pdbs, val_pdbs = dataset_splits.get_splits(args.ds_name, index_csv, val_fold=args.val_fold)
    train_pdbs = filter_by_length(train_pdbs, traj_h5, args.max_length)
    val_pdbs = filter_by_length(val_pdbs, traj_h5, args.max_length)
    logger.info(f"{args.ds_name}: {len(train_pdbs)} train / {len(val_pdbs)} held-out proteins "
                f"(fold {args.val_fold} held out, max_length={args.max_length})")

    build_processed_dataset(
        pdb_ids=train_pdbs + val_pdbs,
        traj_h5_path=traj_h5,
        processed_dir=processed_dir,
        pool_size=args.pool_size,
        course_grain=args.course_grain,
        force_rebuild=args.force_rebuild,
    )

    # scRMSD scores designs against the *relaxed* (deposited) structure, the same ground
    # truth comparisons/{MapDiff,PiFold} use. Only the validation split needs them.
    scrmsd_refs = load_scrmsd_references(traj_h5, val_pdbs, args.pdb_cache)
    logger.info(f"Loaded relaxed reference structures for {len(scrmsd_refs)}/{len(val_pdbs)} "
                "validation proteins (scRMSD)")

    esmfold_tokenizer, esmfold_model = load_esmfold(device=DEVICE)

    model_cfg = OmegaConf.load(constants.HYDRA_CONFIG_PATH / "model" / "AR1_single_chain.yaml")
    features_cfg = OmegaConf.load(constants.HYDRA_CONFIG_PATH / "features" / "ca_bb_single_chain.yaml")
    features_cfg.k = args.k
    # Used exactly as shipped -- this pairing IS the repo's own `experiment=single_chain_k_conf`
    # (configs/experiment/single_chain_k_conf.yaml overrides nothing but `features.k`), i.e. the
    # architecture the authors train for single-chain. We train it from scratch on ATLAS; see
    # train.sub for why we deliberately do not fine-tune from checkpoints/single_chain_k*.ckpt.
    #
    # That checkpoint needs a DIFFERENT config than this one (26 node / 17 edge scalars, 4+4
    # layers, representation=ca -- recorded in its own hyper_parameters, and reproduced by
    # zenodo_compare.py:match_single_chain_ckpt). It is a useful instrument for validating the
    # ATLAS conversion -- a pretrained model scoring ~36% recovery proves our features are real
    # protein geometry, which a from-scratch run cannot tell you -- but it is not our init.

    model = hydra.utils.instantiate(model_cfg).to(DEVICE)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.log({"params": pytorch_total_params})

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
        processed_dir=processed_dir,
        split="train",
        in_memory=True,
    )
    # Wrapped in PdbTrackingFeaturiser so run_validation can map each batch's samples back to
    # the pdb_code they came from (needed to look up ground-truth backbones for scRMSD).
    val_tracker = PdbTrackingFeaturiser(
        hydra.utils.instantiate(features_cfg, split="val", device="cpu", distance_eps=DISTANCE_EPS))
    val_dataset = PTFileDataset(
        pdb_codes=val_pdbs,
        cfg_features=val_tracker,
        processed_dir=processed_dir,
        split="val",
        in_memory=True,
    )
    val_pdb_order = val_tracker.kept_pdb_codes
    # The proteins actually scored, after the featurizer dropped whatever it could not use.
    # MapDiff and PiFold log the same summary key, so the three held-out sets can be diffed
    # rather than assumed identical -- they are cut from the same split index, but each model
    # drops its own failures.
    logger.info(f"{args.ds_name}: featurized {len(val_pdb_order)}/{len(val_pdbs)} held-out proteins; "
                f"dropped {sorted(set(val_pdbs) - set(val_pdb_order))}")
    run.summary["split_counts"] = {"train": len(train_dataset), "val": len(val_dataset)}
    run.summary["test_split_ids"] = list(val_pdb_order)

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

    step = 0
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
            step += 1
            run.log({"train_loss": loss.item(), "epoch": epoch, "step": step})

        logger.info(f"Epoch {epoch}: train loss {total_loss / max(len(train_loader), 1):.4f}")

        metrics = run_validation(model, val_loader, crit, val_pdb_order, scrmsd_refs,
                                  esmfold_tokenizer, esmfold_model, epoch, run)
        print(f"Val Perplexity: {metrics['ppl_mean']:.4f} \\pm {metrics['ppl_std']:.4f}")
        print(f"Top-1 Recovery: {metrics['top1_mean']:.4f} \\pm {metrics['top1_std']:.4f}")
        print(f"Top-5 Recovery: {metrics['top5_mean']:.4f} \\pm {metrics['top5_std']:.4f}")
        print(f"Top-10 Recovery: {metrics['top10_mean']:.4f} \\pm {metrics['top10_std']:.4f}")
        if "scrmsd_mean" in metrics:
            print(f"scRMSD: {metrics['scrmsd_mean']:.4f} \\pm {metrics['scrmsd_std']:.4f}")

    run.finish()


if __name__ == "__main__":
    main()
