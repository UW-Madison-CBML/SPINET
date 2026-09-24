"""
Unlike MapDiff and PiFold -- which are static-structure models and so train on each
protein's relaxed (deposited) PDB entry -- DynamicMPNN takes an **ensemble of conformers
sampled from the MD trajectory**, and that ensemble input is precisely the thing being
benchmarked. So the input pipeline below deliberately stays on the trajectory store.

Pipeline (see README.md for the full reasoning):
  1. For every protein in the dataset's split index, pull its trajectory out of the hdf5
     store, compute a pairwise TM-score matrix with DynamicMPNN's own
     `dynamicmpnn.eval.scoring.compute_tm_score`, and farthest-point-sample a small pool of
     structurally distinct frames off its dissimilarity (1 - TM).
  2. Repack that pool into DynamicMPNN's `.pt` schema (`pyg_dict` + `cluster_members` +
     `tm_scores` / `tm_score_representatives`), which is exactly what
     `ProteinGraphFeaturiserSingleChain` / `PTFileDataset` expect -- the TM matrix is what
     makes the featuriser draw its k conformers by TM dissimilarity rather than at random.
  3. Train on the dataset's training split and score its two held-out splits
     (lib/dataset_splits.py): ATLAS validates on `cross_val` fold `--val-fold` (0 by default)
     and tests on fold `--test-fold` (4), training on the remaining three; mdCATH uses its
     topology split's train/validation/test rows, one apiece. These are the same three splits
     scripts/train_residue_classifier.py and comparisons/{MapDiff,PiFold} cut.
     * Validation is scored every epoch, and training early-stops on val perplexity
       (patience 10, `EarlyStopping`); the **test** split is scored once, with the best val
       epoch's weights, and is never trained on.
     * Each training draw is re-featurised (`ResamplingDataset`): a fresh draw of k conformers
       from the pool and fresh coordinate noise, as upstream's `in_memory: False` trained.
     * The paper's optimizer batch of 32 proteins per step, reached by gradient accumulation
       over `--batch-size` forward passes (`--effective-batch-size`), as in finetune.py.
     * Running the other ATLAS folds is a matter of repeating the job with --val-fold 1..3.

ATLAS and mdCATH are trained and evaluated separately -- one run each (`--ds-name`) --
matching scripts/train_residue_classifier.py.

Logs to a single Weights & Biases run, under the same keys
scripts/train_residue_classifier.py's `run_val` uses (see `lib.stats_utils.ResidueMetrics`):
per-step train loss, parameter count, and per-epoch per-protein recovery (top-1/5/10),
perplexity, per-residue precision/recall/f1, an amino-acid confusion matrix, a `pred_seqs`
table of every held-out design keyed by ``<pdb>_<chain>``.
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
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

for extra_path in (REPO_ROOT / "lib", REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

# DynamicMPNN is normally `pip install -e`'d into the training image; fall back to the
# vendored source tree here so this also works against a plain checkout.
DYNAMICMPNN_SRC = THIS_DIR / "DynamicMPNN" / "src"
if str(DYNAMICMPNN_SRC) not in sys.path:
    sys.path.insert(0, str(DYNAMICMPNN_SRC))

import dataset_splits  # noqa: E402
from load_dynamics import BACKBONE_ATOMS
from residue_classifier_dataset import ResidueClassifierDataset
from stats_utils import ResidueMetrics, three_letter_codes

from dynamicmpnn import constants
from dynamicmpnn.datamodules.pt_dataset import PTFileDataset
from dynamicmpnn.eval.scoring import compute_tm_score
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
# DynamicMPNN's own TM-score (dynamicmpnn.eval.scoring.compute_tm_score) is a CA-only metric.
CA_ATOM_IDX = BACKBONE_ATOMS.index("CA")

# ATLAS coordinates come from `traj.xyz` (scripts/load_dynamics.py:81), and mdtraj stores xyz
# in NANOMETRES -- process_traj never converts. DynamicMPNN was trained on Angstroms: upstream's
# own val_pt_single_chain files measure N-CA 1.459 / CA-C 1.525 / C(i)-N(i+1) 1.329 A. Feeding nm
# leaves knn topology and all the angle features (alpha/kappa/dihedrals) untouched -- they are
# scale-invariant -- but compresses every `edge_distance` / `rbf_16` channel into the lowest bin,
# which is why single_chain_k2.ckpt scored 35.8% recovery on upstream .pt files and 7.3% (loss
# 116) on ours. The conversion matters for conformer selection too: TM-score's d0 is in
# Angstroms, so scoring nanometre coordinates would call every frame identical.
NM_TO_ANGSTROM = 10.0

# A backbone CA-CA virtual bond is ~3.80 A / 0.380 nm in every folded protein, so measuring it
# in a store IS the unit test for that store's units. ATLAS (`traj.xyz` straight out of mdtraj)
# is in nm; the mdCATH store is already in Angstroms. Multiplying the latter by NM_TO_ANGSTROM
# is the same failure documented above, running the other way: it does not touch knn topology
# or the angle features, but it pushes every edge_distance / rbf_16 channel into the TOP bin
# and, worse, makes TM-score's d0 (an Angstrom quantity) call every frame maximally DISSIMILAR
# -- measured min-pairwise-TM over 40 mdCATH domains is 0.046 scaled vs 0.682 unscaled -- so
# farthest-point selection picks its pool out of noise and `is_flat` can never fire.
# Time subsampling for the pairwise-TM matrix, per dataset -- these are NOT interchangeable,
# because the two stores have very different trajectory lengths:
#
#   ATLAS : T = 1001 frames for every trajectory -> cg=25 gives a 41-frame pool everywhere.
#   mdCATH: T = 130..501 (median 500) over the 5327 split members -> cg=25 gives a pool of
#           6..21 (median 20), so NO mdCATH protein can support k > 21 and half cannot
#           support k = 20. cg=4 is the largest value that keeps the shortest trajectory at
#           >= 32 frames (130 -> 33), i.e. that runs the full k grid with nothing padded.
#
# ATLAS stays at 25 so the completed k = 2..20 sweep remains reproducible from this script;
# lowering it would silently re-pool ATLAS and make new points incomparable with the old CSV.
# The cost of mdCATH's cg=4 is preprocessing: compute_pairwise_tm is a serial O(n_coarse^2)
# Kabsch loop, measured at 0.21 s/protein at cg=25 and 1.38 s/protein at cg=4, i.e. ~19 min
# vs ~2 h for 5327 proteins. It is paid once per pool directory, but job scratch does not
# survive, so every chunked mdCATH submission pays it again.
DEFAULT_COURSE_GRAIN = {"atlas": 25, "mdcath": 4}

CA_CA_ANGSTROM = 3.80
CA_CA_TOLERANCE = 0.30  # observed spread over 60 trajectories per store is < 0.03 A


def detect_coord_scale(coords_ds) -> float:
    """Factor converting this store's coordinates to Angstroms, read off its own geometry.

    Returns 1.0 for an Angstrom store and NM_TO_ANGSTROM for a nanometre one, and raises if
    the CA-CA distance matches neither -- a store whose units cannot be identified must not be
    guessed at, because both wrong answers are silent (see the comment above).
    """
    ca = np.asarray(coords_ds[:, 0, CA_ATOM_IDX, :], dtype=np.float64)  # num_res, 3 @ frame 0
    if ca.shape[0] < 2:
        raise ValueError("cannot detect coordinate units from a single-residue chain")
    spacing = float(np.median(np.linalg.norm(np.diff(ca, axis=0), axis=1)))

    for scale in (1.0, NM_TO_ANGSTROM):
        if abs(spacing * scale - CA_CA_ANGSTROM) <= CA_CA_TOLERANCE:
            return scale
    raise ValueError(
        f"median consecutive CA-CA distance is {spacing:.4f}, which is neither "
        f"{CA_CA_ANGSTROM} A (Angstrom store) nor {CA_CA_ANGSTROM / NM_TO_ANGSTROM} "
        f"(nanometre store) within {CA_CA_TOLERANCE}. Refusing to guess the units: both wrong "
        f"answers corrupt the TM matrix and the RBF edge features silently."
    )
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


def compute_pairwise_tm(coords: torch.Tensor) -> torch.Tensor:
    """Full T x T TM-similarity matrix over one trajectory's (coarse-grained) frames.

    Uses DynamicMPNN's own TM-score -- `dynamicmpnn.eval.scoring.compute_tm_score`, the
    function its evaluation pipeline scores designs with -- rather than a Kabsch RMSD, so
    conformer selection here is driven by the same structural-similarity measure the model's
    authors use for k-conformer selection upstream (`ProteinGraphFeaturiserSingleChain.
    compute_sequential_probabilities` consumes exactly this kind of TM matrix).

    Upstream fills that matrix with Foldseek all-vs-all `qtmscore`/`ttmscore` averages
    (src/dynamicmpnn/scripts/pipeline/add_TM_scores_seq80_single.py), which needs a Foldseek
    binary and an alignment step. Neither is needed -- or correct -- here: every frame is the
    *same* chain, so the residue correspondence is the identity and `compute_tm_score`'s
    Kabsch superposition over CA atoms is the whole calculation. It is symmetric under that
    correspondence (both frames have the same length, so d0 and the normalisation match
    either way round), which is what upstream's qtm/ttm averaging is there to approximate.

    coords: T, num_res, num_atoms, 3 (BACKBONE_ATOMS order, Angstroms)
    returns: T, T float tensor of TM scores, 1.0 on the diagonal
    """
    torch.set_num_threads(1)  # don't let each worker fight the others for cores
    ca = coords[:, :, CA_ATOM_IDX, :].double().numpy()  # T, num_res, 3
    T = ca.shape[0]

    tm_matrix = np.eye(T, dtype=np.float32)
    for i in range(T):
        for j in range(i + 1, T):
            tm = compute_tm_score(ca[i], ca[j])
            tm_matrix[i, j] = tm
            tm_matrix[j, i] = tm
    return torch.from_numpy(tm_matrix)


def farthest_point_sample(dissimilarity: torch.Tensor, pool_size: int) -> list:
    """Greedy max-min selection over a frame-to-frame *dissimilarity* matrix (1 - TM).

    Same spirit as DynamicMPNN's own TM-dissimilarity k-selection
    (`ProteinGraphFeaturiserSingleChain.compute_sequential_probabilities`, which softmaxes
    `1 - avg TM to the already-selected`), made deterministic: this only has to carve a pool
    of structurally distinct frames out of the trajectory, and the featuriser still does the
    stochastic k-of-pool draw at train time from the TM matrix saved alongside the pool.
    """
    T = dissimilarity.shape[0]
    pool_size = min(pool_size, T)

    flat_idx = torch.argmax(dissimilarity).item()
    i0, j0 = divmod(flat_idx, T)
    selected = [i0, j0] if i0 != j0 else [i0]

    while len(selected) < pool_size:
        min_dist_to_selected = dissimilarity[selected].min(dim=0).values
        min_dist_to_selected[selected] = -1.0  # never re-pick an already-selected frame
        selected.append(int(torch.argmax(min_dist_to_selected).item()))

    return selected[:pool_size]


def residues_to_type_ids(raw_residues: np.ndarray) -> torch.Tensor:
    """3-letter residue codes (bytes, from the hdf5 `residues` dataset) -> DynamicMPNN's
    BASE_AMINO_ACIDS integer vocabulary (20 = gap, 21 = unknown/non-standard).

    Force-field residue names (CHARMM's HSD/HSE/HSP, AMBER's HIE/CYX, ...) are first mapped
    to their standard residue with the sheaf dataset's own RESIDUE_ALIASES, so both models
    see the same labels. This matters beyond the labels: DynamicMPNN's featuriser
    (`eliminate_gaps`) DELETES every position typed 20 or 21, so an unaliased HSD would drop
    every histidine from the protein and shorten the designed sequence."""
    aliases = ResidueClassifierDataset.RESIDUE_ALIASES
    ids = []
    for raw in raw_residues:
        code3 = raw.decode().strip()[:3].upper()
        code3 = aliases.get(code3, code3)
        one_letter = STANDARD_AMINO_ACID_MAPPING_3_TO_1.get(code3)
        if one_letter is not None and one_letter in BASE_AMINO_ACIDS:
            ids.append(BASE_AMINO_ACIDS.index(one_letter))
        else:
            ids.append(UNKNOWN_TOKEN)
    return torch.tensor(ids, dtype=torch.long)


def frame_to_pyg_data(frame_coords: np.ndarray, residue_type: torch.Tensor) -> Data:
    """One selected conformer -> the minimal per-conformer fields
    `ProteinGraphFeaturiserSingleChain.stack_conformations` actually reads.

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
    coords_ds = h5_file[group_name + "/coordinates"]  # num_res, T, num_atoms, 3
    # Units are read off the store rather than assumed: ATLAS is nm, mdCATH is already A.
    to_angstrom = detect_coord_scale(coords_ds)

    coarse_coords = (torch.from_numpy(coords_ds[:, ::course_grain]).permute(1, 0, 2, 3).float()
                     * to_angstrom)

    tm_matrix = compute_pairwise_tm(coarse_coords)
    off_diagonal = ~torch.eye(tm_matrix.shape[0], dtype=torch.bool)
    # TM >= 0.9 between every pair of frames means the trajectory never left its starting fold.
    is_flat = bool(off_diagonal.any() and tm_matrix[off_diagonal].min().item() >= 0.9)

    coarse_states = farthest_point_sample(1.0 - tm_matrix, pool_size)
    # `farthest_point_sample` returns the pool in selection (most-distinct-first) order; sort
    # for a stable, chronological pool, and carry the same permutation into the TM sub-matrix
    # so `tm_scores[i, j]` still refers to `cluster_members[i]` / `cluster_members[j]`.
    pool_order = sorted(int(c) for c in coarse_states)
    real_frame_idx = [c * course_grain for c in pool_order]
    pool_tm = tm_matrix[torch.tensor(pool_order)][:, torch.tensor(pool_order)]

    residue_type = residues_to_type_ids(h5_file[group_name + "/residues"][:])
    pool_frames = coords_ds[:, real_frame_idx] * to_angstrom  # num_res, pool, num_atoms, 3

    pyg_dict = {}
    for pool_pos, frame_idx in enumerate(real_frame_idx):
        pyg_dict[f"{pdb_id}_frame{frame_idx}"] = frame_to_pyg_data(pool_frames[:, pool_pos], residue_type)

    # `tm_scores` + `tm_score_representatives` are the two fields
    # `ProteinGraphFeaturiserSingleChain.get_entries` looks for: with them present it draws
    # the k conformers it trains on by TM dissimilarity (its own
    # `compute_sequential_probabilities`), and without them it falls back to `random.sample`.
    # Saving them here is what puts the final k-of-pool selection on DynamicMPNN's own TM
    # machinery instead of an RMSD proxy.
    ensemble = Data(pyg_dict=pyg_dict, cluster_members=list(pyg_dict.keys()),
                    tm_scores=pool_tm, tm_score_representatives=list(pyg_dict.keys()))
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

    def needs_rebuild(pdb_id):
        path = processed_dir / f"{pdb_id}.pt"
        if force_rebuild or not path.exists():
            return True
        try:
            return getattr(torch.load(path, weights_only=False), "tm_scores", None) is None
        except Exception:
            return True

    pending = [p for p in pdb_ids if needs_rebuild(p)]
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
            f"{len(flat_trajectories)} trajectories are conformationally flat (all pairwise TM >= 0.9): "
            f"{flat_trajectories} -- these are an easy win for any model, keep in mind when "
            "comparing recovery numbers."
        )


def check_pool_sizes(processed_dir: Path, pdb_ids: list, max_k: int, course_grain=None) -> None:
    """Fail before training if any protein's saved pool is smaller than the largest k.

    `ProteinGraphFeaturiserSingleChain.get_entries` handles `n < k` by duplicating conformers
    at random, which is silent: the run completes, the assert on `len(confs_list) == k`
    passes, and the model is simply fed the same frame more than once. In a single run that
    quietly shrinks the ensemble below --k; across a sweep it reads as the recovery curve
    saturating. Either way it has to be an error here rather than a warning nobody sees. A pool comes up short when the trajectory has fewer coarse-grained
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
            f"{' ...' if len(short) > 10 else ''}. {_pool_remedy(worst, max_k, course_grain)}"
        )


def _pool_remedy(worst: int, max_k: int, course_grain) -> str:
    """The actionable half of check_pool_sizes' error.

    Capping k (--k, or the top of sweep_k.py's --k-grid) at `worst` is the literal fix but almost never the right one: when a
    whole dataset comes up short it is because --course-grain was tuned for a different
    store's trajectory length, and the fix is to subsample less aggressively rather than to
    abandon the larger k. `worst` frames survived at the current stride, so the
    trajectory holds roughly `worst * course_grain` frames and a stride of
    `worst * course_grain // max_k` puts max_k of them back in reach.
    """
    if not course_grain:
        return (f"Lower --course-grain so more frames survive subsampling (then "
                f"--force-rebuild), or lower k to {worst}.")
    suggested = max(1, (worst * course_grain) // max_k)
    if suggested >= course_grain:
        return (f"The shortest trajectory has only ~{worst * course_grain} frames, so k={max_k} "
                f"is out of reach at any stride; lower k to {worst} "
                f"or drop these proteins.")
    return (f"--course-grain is {course_grain}, tuned for a longer-trajectory store; the "
            f"shortest trajectory here holds only ~{worst * course_grain} frames. Rerun with "
            f"--course-grain {suggested} --force-rebuild to fit k={max_k}, or lower k to "
            f"{worst}.")


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


# --------------------------------------------------------------------------------------
# Step 4: the dataset's two held-out splits, val and test (see lib/dataset_splits.py)
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
            "or pass --from-scratch. (To fine-tune a released checkpoint, use finetune.py, "
            "which builds the model from the checkpoint's own config.)"
        )
    logger.info(f"Loaded pretrained weights from {ckpt_path} ({len(stripped)} tensors)")


def save_weights(model: torch.nn.Module, optimizer: torch.optim.Optimizer, save_path: Path,
                 epoch: int, step: int, model_cfg, features_cfg, args,
                 extra_hparams: dict = None) -> None:
    # Keys carry the same `GNN_model.` prefix as upstream's Lightning checkpoints, so a saved
    # run reloads through `load_pretrained_weights` via `--ckpt <path>` like any other.
    checkpoint = {
        "state_dict": {f"GNN_model.{k}": v for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "hyper_parameters": {
            "model": OmegaConf.to_container(model_cfg, resolve=True),
            "features": OmegaConf.to_container(features_cfg, resolve=True),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            **(extra_hparams or {}),
        },
    }
    # Write-then-rename, so an eviction mid-save never leaves a truncated file behind.
    tmp_path = save_path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(save_path)


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



class ResamplingDataset(torch.utils.data.Dataset):
    """Training split that re-featurises each protein on every `__getitem__`.

    The raw conformer pools are small, so they are held in memory; only the featurisation --
    which conformers of the pool, and the featuriser's own train-time coordinate noise
    (`noise_scale`, applied when `split == "train"`) -- is redone per draw, as upstream's
    `in_memory: False` training did. `PTFileDataset(in_memory=True)` instead featurises once,
    freezing both the conformer draw and the noise for every epoch.

    `featuriser` is anything called as `featuriser(protein, pdb_code=...)`: the k-conformer
    featuriser train.py builds, or finetune.py's `LegacyPairAdapter`.

    Proteins the featuriser rejects are dropped once, up front: `safe_collate` hard-errors on
    a `None` sample, so one must never reach the loader. A rejection is a property of the
    protein (too short / no valid coordinates), not of which conformers were drawn, so a
    protein that survives the up-front pass survives later draws too; if one ever does not, it
    falls back to its up-front featurisation rather than taking down the job.
    """

    def __init__(self, pdb_codes, featuriser, processed_dir):
        self.featuriser = featuriser
        self.proteins, self.fallback, self.pdb_codes = [], [], []
        for pdb_code in tqdm(pdb_codes, desc="Loading train ensembles"):
            path = processed_dir / f"{pdb_code}.pt"
            if not path.exists():
                continue  # build_processed_dataset already warned about it
            protein = torch.load(path, weights_only=False)
            featurised = self.featuriser(protein, pdb_code=pdb_code)
            if featurised is None:
                logger.warning(f"Skipping train protein {pdb_code}: featuriser rejected it.")
                continue
            self.proteins.append(protein)
            self.fallback.append(featurised)
            self.pdb_codes.append(pdb_code)

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        featurised = self.featuriser(self.proteins[idx], pdb_code=self.pdb_codes[idx])
        return featurised if featurised is not None else self.fallback[idx]


class EarlyStopping:
    """Early stopping on val perplexity, with the best epoch's weights kept for test.

    The protocol every comparison model shares (comparisons/{MapDiff,PiFold} do the same):
    validate every epoch, keep the weights of the lowest per-protein val perplexity
    (`val_perp_mean`, lib.stats_utils.ResidueMetrics), stop once `patience` epochs pass
    without beating it, and score test once with the kept weights -- so the test split
    never influences either the weights or the choice of epoch. `patience` 0 disables the
    stopping but still selects the best epoch.
    """

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_perplexity, self.best_epoch, self.best_state = float("inf"), None, None

    def update(self, model: torch.nn.Module, perplexity: float, epoch: int) -> bool:
        """Record this epoch's val perplexity; True if it is a new best (weights kept)."""
        if not perplexity < self.best_perplexity:  # also rejects NaN
            return False
        self.best_perplexity, self.best_epoch = perplexity, epoch
        self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        logger.info(f"New best val perplexity {perplexity:.4f} at epoch {epoch}")
        return True

    def should_stop(self, epoch: int) -> bool:
        if self.patience <= 0 or self.best_epoch is None:
            return False
        epochs_since_best = epoch - self.best_epoch
        if epochs_since_best < self.patience:
            return False
        logger.info(f"Early stopping at epoch {epoch}: val perplexity has not improved in "
                    f"{epochs_since_best} epochs (best {self.best_perplexity:.4f} at epoch "
                    f"{self.best_epoch})")
        return True

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is None:
            logger.warning("No finite val perplexity was ever recorded; testing the final weights.")
            return
        model.load_state_dict(self.best_state)
        logger.info(f"Restored the best val epoch ({self.best_epoch}, perplexity "
                    f"{self.best_perplexity:.4f})")


def run_validation(model, val_loader, crit, val_pdb_order,
                   epoch, run, num_params=0, split_name="val"):
    """Score one held-out split and log scripts/train_residue_classifier.py's metric set.

    `lib.stats_utils.ResidueMetrics` owns every convention here (per-protein top-1/5/10,
    per-protein perplexity, per-residue precision/recall/f1, the confusion matrix and the
    `pred_seqs` table), so the keys this logs are the same keys the sheaf model logs.

    `split_name` prefixes those keys ("val"/"test"), so the per-epoch validation curve and the
    single end-of-training test score stay separate series in the same W&B run.
    """
    model.eval()

    metrics = ResidueMetrics(three_letter_codes(BASE_AMINO_ACIDS), val_name=split_name)
    # `crit` carries DynamicMPNN's label smoothing (and ignores GAP), which is right for the
    # gradient but would make the reported perplexity incomparable with the sheaf model's --
    # that one is exp() of a plain cross-entropy. Score with a plain one here.
    scoring_crit = torch.nn.CrossEntropyLoss()
    sample_offset = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"{split_name.capitalize()} eval"):
            batch = batch.to(DEVICE)
            logits, valid_mask = model(batch)
            target = batch.seq

            num_proteins_in_batch = batch.batch.max().item() + 1
            batch_pdb_ids = val_pdb_order[sample_offset:sample_offset + num_proteins_in_batch]
            sample_offset += num_proteins_in_batch

            for p_idx, pdb_id in zip(range(num_proteins_in_batch), batch_pdb_ids):
                protein_mask = (batch.batch == p_idx) & valid_mask
                if not protein_mask.any():
                    continue

                masked_logits = logits[protein_mask].cpu()
                # just save masked logits?
                masked_seq = target[protein_mask].cpu()

                pred_idx = masked_logits.argmax(dim=-1)
                design = ''.join(BASE_AMINO_ACIDS[i] for i in pred_idx.tolist())

                # GAP/UNKNOWN targets have no column in the 20-class confusion matrix, so
                # they are dropped from the scoring (the design above still spans every
                # residue).
                keep = masked_seq < len(BASE_AMINO_ACIDS)
                if keep.any():
                    kept_logits, kept_seq = masked_logits[keep], masked_seq[keep]
                    metrics.add_protein(
                        kept_logits, kept_seq, pdb_id=pdb_id,
                        nll=scoring_crit(kept_logits, kept_seq).item(),
                        sequence=design,
                    )

    log_dict = metrics.to_log_dict(
        cm_title=f"{split_name.capitalize()} Amino Acid Confusion Matrix", epoch=epoch)
    if run:
        run.log(log_dict)
    print(metrics.summary_line(num_params))
    return log_dict


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
                         help="ATLAS only: cross_val fold used as the validation split. "
                              "mdCATH ignores it -- its split column is categorical.")
    parser.add_argument("--test-fold", type=int, default=dataset_splits.DEFAULT_TEST_FOLD,
                         help="ATLAS only: cross_val fold held out as the test split -- never trained "
                              "on, scored once with the best val epoch's weights. mdCATH ignores it.")
    parser.add_argument("--processed-dir", type=Path, default=None,
                         help="Where the .pt conformer ensembles live (default: ./processed_data_<ds>)")
    parser.add_argument("--max-length", type=int, default=dataset_splits.MAX_LENGTH,
                         help="Drop proteins longer than this many residues, the cutoff MapDiff and "
                              "PiFold also apply (lib/dataset_splits.MAX_LENGTH). Pass 0 to disable.")
    parser.add_argument("--k", type=int, default=2, help="conformers sampled per protein at train/val time")
    parser.add_argument("--pool-size", type=int, default=10, help="conformer pool saved per protein's .pt file")
    parser.add_argument("--course-grain", type=int, default=None,
                        help="time subsampling for the pairwise-TM matrix (default: "
                             f"{DEFAULT_COURSE_GRAIN}, keyed on --ds-name)")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Proteins per forward pass -- upstream's per-GPU batch. Activation "
                             "memory scales with k * batch-size, so lower this for large --k")
    parser.add_argument("--effective-batch-size", type=int, default=32,
                        help="Proteins per optimizer step (the paper's batch size); reached by "
                             "accumulating gradients over effective/--batch-size forward passes")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Upper bound on epochs (upstream's trainer max_epochs); early "
                             "stopping normally ends the run well before it")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Stop once val perplexity has not improved on its best for this many "
                             "epochs; 0 disables stopping (the best epoch is still restored)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=dataset_splits.SEED)
    parser.add_argument("--from-scratch", action="store_true", help="skip loading a pretrained checkpoint")
    parser.add_argument("--ckpt", type=Path, default=None, help="override the default single_chain_k{k}.ckpt")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=Path, default=THIS_DIR / "weights",
                        help="trained weights land here as <run_name>_<wandb run id>.ckpt")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.effective_batch_size % args.batch_size:
        raise ValueError(f"--effective-batch-size {args.effective_batch_size} is not a multiple "
                         f"of --batch-size {args.batch_size}")
    accum_steps = args.effective_batch_size // args.batch_size
    if args.pool_size < args.k:
        raise ValueError(f"--pool-size {args.pool_size} < --k {args.k}: the featurizer would "
                         f"pad every ensemble with duplicated frames (see check_pool_sizes)")
    # Per-dataset default (see DEFAULT_COURSE_GRAIN): ATLAS's 1001-frame trajectories and
    # mdCATH's 130..501-frame ones cannot share one subsampling stride.
    if args.course_grain is None:
        args.course_grain = DEFAULT_COURSE_GRAIN[args.ds_name]
        logger.info(f"--course-grain defaulted to {args.course_grain} for {args.ds_name}")
    # `random` and `np.random`, not just torch: DynamicMPNN's featurizer picks which k of each
    # protein's saved conformer pool to use with `random.sample` / `np.random.choice`
    # (ProteinGraphFeaturiserSingleChain.get_entries). The val/test datasets below are built
    # with in_memory=True, so their draw happens once and then holds for every epoch -- leaving
    # these unseeded would mean the *validation set itself* differed between runs. (The
    # training draws are redone every epoch; DataLoader seeds each worker from the torch seed.)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    traj_h5 = args.traj_h5 or Path(dataset_splits.default_h5(args.ds_name))
    index_csv = args.index_csv or Path(dataset_splits.default_index_csv(args.ds_name))
    # Keyed by the dataset build (mdcath320 vs mdcath450, ...), not just ds_name: the cached
    # ensembles come from the trajectory store, so builds must not share a cache.
    ds_tag = dataset_splits.dataset_tag(args.ds_name, traj_h5)
    processed_dir = args.processed_dir or (THIS_DIR / f"processed_data_{ds_tag}")
    run_name = args.run_name or f"DynamicMPNN_{ds_tag}"

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
            "effective_batch_size": args.effective_batch_size,
            "epochs": args.epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "lr": args.lr,
            "seed": args.seed,
            "from_scratch": args.from_scratch,
            "task": f"predicting residues from {args.ds_name} MD conformer ensembles (DynamicMPNN)",
        },
    )

    # WANDB artifact logging
    artifact = wandb.Artifact(name="scripts", type="model_file")
    artifact.add_file(os.path.abspath(__file__))
    for dependency in (REPO_ROOT / "lib" / "stats_utils.py",
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
    # A pool shorter than k is padded with duplicated frames, silently -- fail instead.
    check_pool_sizes(processed_dir, train_pdbs + val_pdbs + test_pdbs, args.k, args.course_grain)
    # build_processed_dataset only warns about ids with no trajectory group in the h5 and
    # writes no .pt for them; PTFileDataset torch.loads every id it is given, so drop them here.
    train_pdbs, val_pdbs, test_pdbs = (
        [p for p in ids if (processed_dir / f"{p}.pt").exists()]
        for ids in (train_pdbs, val_pdbs, test_pdbs))
    logger.info(f"{args.ds_name}: {len(train_pdbs)} train / {len(val_pdbs)} val / "
                f"{len(test_pdbs)} test proteins with a built ensemble")

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

    # Re-featurised on every draw: a fresh TM-weighted draw of k conformers from the pool and
    # fresh train-time coordinate noise each time a protein comes up, as upstream trained.
    # Val/test below stay in_memory=True -- featurised once (seeded), no noise -- so every
    # epoch is scored on the same inputs.
    train_dataset = ResamplingDataset(
        train_pdbs,
        hydra.utils.instantiate(features_cfg, split="train", device="cpu", distance_eps=DISTANCE_EPS),
        processed_dir,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training protein survived featurisation.")
    # Wrapped in PdbTrackingFeaturiser so run_validation can label each batch's samples with
    # the pdb_code they came from.
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

    test_tracker = PdbTrackingFeaturiser(
        hydra.utils.instantiate(features_cfg, split="test", device="cpu", distance_eps=DISTANCE_EPS))
    test_dataset = PTFileDataset(
        pdb_codes=test_pdbs,
        cfg_features=test_tracker,
        processed_dir=processed_dir,
        split="test",
        in_memory=True,
    )
    test_pdb_order = test_tracker.kept_pdb_codes

    # The proteins actually scored, after the featurizer dropped whatever it could not use.
    # MapDiff and PiFold log the same summary key, so the three held-out sets can be diffed
    # rather than assumed identical -- they are cut from the same split index, but each model
    # drops its own failures.
    for split, wanted, kept in (("val", val_pdbs, val_pdb_order), ("test", test_pdbs, test_pdb_order)):
        logger.info(f"{args.ds_name}: featurized {len(kept)}/{len(wanted)} {split} proteins; "
                    f"dropped {sorted(set(wanted) - set(kept))}")
    run.summary["split_counts"] = {"train": len(train_dataset), "val": len(val_dataset),
                                   "test": len(test_dataset)}
    run.summary["val_split_ids"] = list(val_pdb_order)
    run.summary["test_split_ids"] = list(test_pdb_order)

    # Upstream's own loader: `drop_last=True`, `batch_size` proteins per forward pass (their
    # per-GPU batch). The paper's batch of 32 is what their DDP ranks summed to per optimizer
    # step; on one GPU the same step is `accum_steps` forward passes with the gradient averaged
    # over them -- the same regime finetune.py trains under.
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
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=safe_collate,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=safe_collate,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=GAP_TOKEN)

    # The W&B run id keeps resubmitted jobs from overwriting each other once HTCondor copies
    # every job's weights/ back into the same submit-side directory.
    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.save_dir / f"{run_name}_{run.id}.ckpt"
    best_path = args.save_dir / f"{run_name}_{run.id}_best.ckpt"
    run.summary["weights_path"] = save_path.name
    run.summary["best_weights_path"] = best_path.name

    stopper = EarlyStopping(args.early_stopping_patience)
    stopped_epoch = args.epochs - 1
    step = 0
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
                                  num_params=pytorch_total_params)
        print(f"Val Perplexity: {metrics['val_perp_mean']:.4f} \\pm {metrics['val_perp_std']:.4f}")
        print(f"Top-1 Recovery: {metrics['val_top1_acc_mean']:.4f} \\pm {metrics['val_top1_acc_std']:.4f}")
        print(f"Top-5 Recovery: {metrics['val_top5_acc_mean']:.4f} \\pm {metrics['val_top5_acc_std']:.4f}")
        print(f"Top-10 Recovery: {metrics['val_top10_acc_mean']:.4f} \\pm {metrics['val_top10_acc_std']:.4f}")

        # Overwritten every epoch, so an evicted job still brings home its latest weights.
        save_weights(model, optimizer, save_path, epoch, step, model_cfg, features_cfg, args)
        logger.info(f"Saved epoch {epoch} weights to {save_path}")
        if stopper.update(model, metrics["val_perp_mean"], epoch):
            # On disk as well, so an evicted job still brings home its best weights.
            save_weights(model, optimizer, best_path, epoch, step, model_cfg, features_cfg, args)
            run.log({"best_val_perp_mean": stopper.best_perplexity,
                     "best_val_epoch": stopper.best_epoch, "epoch": epoch})
        if stopper.should_stop(epoch):
            run.log({"early_stopped_epoch": epoch, "epoch": epoch})
            stopped_epoch = epoch
            break

    run.summary.update({"best_val_perp_mean": stopper.best_perplexity,
                        "best_val_epoch": stopper.best_epoch, "stopped_epoch": stopped_epoch})

    # The test split is scored exactly once, with the best val epoch's weights (see
    # EarlyStopping) -- nothing the test split influenced.
    stopper.restore(model)
    logger.info(f"Scoring the held-out test split ({len(test_pdb_order)} proteins)")
    test_metrics = run_validation(model, test_loader, crit, test_pdb_order, stopper.best_epoch, run,
                                   num_params=pytorch_total_params, split_name="test")
    print(f"Test Perplexity: {test_metrics['test_perp_mean']:.4f} \\pm {test_metrics['test_perp_std']:.4f}")
    print(f"Test Top-1 Recovery: {test_metrics['test_top1_acc_mean']:.4f} \\pm {test_metrics['test_top1_acc_std']:.4f}")
    print(f"Test Top-5 Recovery: {test_metrics['test_top5_acc_mean']:.4f} \\pm {test_metrics['test_top5_acc_std']:.4f}")
    print(f"Test Top-10 Recovery: {test_metrics['test_top10_acc_mean']:.4f} \\pm {test_metrics['test_top10_acc_std']:.4f}")
    run.summary.update({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})

    run.finish()


if __name__ == "__main__":
    main()
