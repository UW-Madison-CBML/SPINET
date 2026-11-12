"""Per-protein conformer pools for ADFLIP training, built exactly as comparisons/DynamicMPNN does.

ADFLIP is trained here on the same ensemble input DynamicMPNN is benchmarked on, so the frames it
sees have to be chosen the same way. This module is a dependency-light port of the pool-building
half of `comparisons/DynamicMPNN/train.py` (that file imports the vendored `dynamicmpnn` package,
hydra and torch_geometric, none of which are in the ADFLIP image). The pool logic is copied, not
re-derived, so the same trajectory store, `--pool-size`, `--course-grain` and `--frame-selection`
give the *same frames* in both models:

* every replica of a protein is pooled (residue-sequence mismatches are left out with a warning);
* units are read off the store's own CA-CA spacing (`detect_coord_scale`): ATLAS is nm, mdCATH A;
* ``fps`` -- farthest-point sampling on 1 - TM over every `course_grain`-th frame, TM computed with
  DynamicMPNN's Kabsch/d0 TM-score (`_batched_tm`), pool sorted into (replica, time) order;
* ``uniform`` -- `pool_size` frames uniformly at random from all frames, seeded from (seed, id).

`draw_k` is the train/val-time k-of-pool draw, a port of DynamicMPNN's
`ProteinGraphFeaturiserSingleChain.get_entries` / `compute_sequential_probabilities`: sequential,
each next frame drawn with probability softmax((1 - mean TM to the frames already drawn) / 2),
floored at 0.01; uniform `random.sample` when no TM matrix was saved.

Pools are saved as one ``<id>.npz`` per protein (coordinates in Angstroms, `BACKBONE_ATOMS` order).
"""
import logging
import os
import zlib
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

import traj_frames
from relaxed_pdb import BACKBONE_ATOMS

log = logging.getLogger(__name__)

CA_ATOM_IDX = BACKBONE_ATOMS.index("CA")

# DynamicMPNN/train.py's DEFAULT_COURSE_GRAIN -- see the long comment there. ATLAS trajectories are
# 1001 frames, mdCATH 130..501, so they cannot share a stride; cg=4 keeps the shortest mdCATH
# trajectory at >= 32 coarse frames.
DEFAULT_COURSE_GRAIN = {"atlas": 25, "mdcath": 4}

FRAME_SELECTIONS = ("fps", "uniform")


# --------------------------------------------------------------------------------------
# TM-score and farthest-point sampling (ported from DynamicMPNN/train.py)
# --------------------------------------------------------------------------------------

def tm_score(reference, mobile):
    """Scalar TM-score after Kabsch superposition, CA only -- the reference implementation the
    batched version is checked against (same maths as `dynamicmpnn.eval.scoring.compute_tm_score`)."""
    return float(_batched_tm(reference, mobile[None])[0])


def _batched_tm(reference, mobiles):
    """TM-score of every ``mobiles[m]`` (M, R, 3) against ``reference`` (R, 3): Kabsch superposition
    with reflection correction and DynamicMPNN's d0, batched over M."""
    ref_c = reference - reference.mean(axis=0)
    mob_c = mobiles - mobiles.mean(axis=1, keepdims=True)
    covariance = np.einsum("mra,rb->mab", mob_c, ref_c)
    v_matrix, _, w_transpose = np.linalg.svd(covariance)
    correction = np.broadcast_to(np.eye(3), covariance.shape).copy()
    correction[np.linalg.det(v_matrix @ w_transpose) < 0, -1, -1] = -1.0
    rotation = v_matrix @ correction @ w_transpose
    aligned = mob_c @ rotation
    distances = np.linalg.norm(ref_c[None] - aligned, axis=-1)  # M, R
    length = reference.shape[0]
    d0 = max(1.24 * max(length - 15, 1) ** (1.0 / 3.0) - 1.8, 0.5)
    return np.mean(1.0 / (1.0 + np.square(distances / d0)), axis=1)


def compute_pairwise_tm(coords):
    """(T, T) TM-similarity matrix over ``coords`` (T, R, num_atoms, 3), Angstroms, BACKBONE_ATOMS
    order. Symmetric because every frame is the same chain (identity residue correspondence)."""
    ca = np.asarray(coords[:, :, CA_ATOM_IDX, :], dtype=np.float64)
    T = ca.shape[0]
    tm_matrix = np.eye(T, dtype=np.float32)
    for i in range(T - 1):
        row = _batched_tm(ca[i], ca[i + 1:]).astype(np.float32)
        tm_matrix[i, i + 1:] = row
        tm_matrix[i + 1:, i] = row
    return tm_matrix


def farthest_point_sample(dissimilarity, pool_size):
    """Greedy max-min selection over a (T, T) dissimilarity matrix; deterministic (pure argmax).
    Same tie-breaking as DynamicMPNN's torch version (first maximum in row-major order)."""
    T = dissimilarity.shape[0]
    pool_size = min(pool_size, T)
    i0, j0 = divmod(int(np.argmax(dissimilarity)), T)
    selected = [i0, j0] if i0 != j0 else [i0]
    while len(selected) < pool_size:
        min_dist = dissimilarity[selected].min(axis=0).copy()
        min_dist[selected] = -1.0
        selected.append(int(np.argmax(min_dist)))
    return selected[:pool_size]


# --------------------------------------------------------------------------------------
# Pool building
# --------------------------------------------------------------------------------------

def _scale_of(coords_ds):
    """Unit factor to Angstroms, read off frame 0 (`traj_frames.detect_coord_scale` takes (R, A, 3))."""
    return traj_frames.detect_coord_scale(np.asarray(coords_ds[:, 0], dtype=np.float64))


def _read_replicas(h5_file, group_names, protein_id):
    residues = h5_file[group_names[0] + "/residues"][:]
    replicas = []
    for group_name in group_names:
        group_residues = h5_file[group_name + "/residues"][:]
        if len(group_residues) != len(residues) or not np.array_equal(group_residues, residues):
            log.warning("%s: %s disagrees with %s on the residue sequence -- left out of the "
                        "conformer pool.", protein_id, group_name, group_names[0])
            continue
        coords_ds = h5_file[group_name + "/coordinates"]  # num_res, T, num_atoms, 3
        replicas.append((group_name, coords_ds, _scale_of(coords_ds)))
    return residues, replicas


def build_pool(h5_file, group_names, protein_id, pool_size, course_grain, selection="fps", seed=42):
    """One protein's pool as a dict of arrays (see `save_pool`), plus ``is_flat``."""
    if selection not in FRAME_SELECTIONS:
        raise ValueError("selection must be one of {}, got {!r}".format(FRAME_SELECTIONS, selection))
    residues, replicas = _read_replicas(h5_file, group_names, protein_id)
    used_groups = [g for g, _, _ in replicas]
    # Raises on a residue with no one-letter code: ADFLIP's 20-class scoring has no slot for it.
    resnames, letters = traj_frames.parse_residues(residues)

    if selection == "fps":
        candidates, blocks = [], []
        for g, (_, coords_ds, scale) in enumerate(replicas):
            candidates.extend((g, f) for f in range(0, coords_ds.shape[1], course_grain))
            block = np.asarray(coords_ds[:, ::course_grain], dtype=np.float32) * np.float32(scale)
            blocks.append(np.transpose(block, (1, 0, 2, 3)))
        coarse = np.concatenate(blocks)  # candidates, R, A, 3
        tm_matrix = compute_pairwise_tm(coarse)
        if tm_matrix.shape[0] > 1:  # the batched path must agree with the scalar reference
            ref = tm_score(coarse[0, :, CA_ATOM_IDX].astype(np.float64),
                           coarse[-1, :, CA_ATOM_IDX].astype(np.float64))
            if abs(ref - float(tm_matrix[0, -1])) > 1e-4:
                raise RuntimeError("batched TM {} != reference {}".format(tm_matrix[0, -1], ref))
        off = ~np.eye(tm_matrix.shape[0], dtype=bool)
        is_flat = bool(off.any() and tm_matrix[off].min() >= 0.9)
        order = sorted(farthest_point_sample(1.0 - tm_matrix, pool_size))
        pool_coords = coarse[order]
        pool_tm = tm_matrix[np.ix_(order, order)]
        sources = [candidates[c] for c in order]
    else:
        candidates = [(g, f) for g, (_, ds, _) in enumerate(replicas) for f in range(ds.shape[1])]
        rng = np.random.default_rng([seed, zlib.crc32(protein_id.encode())])
        picks = sorted(rng.choice(len(candidates), size=min(pool_size, len(candidates)),
                                  replace=False).tolist())
        sources = [candidates[c] for c in picks]
        frames_out = []
        for g, (_, coords_ds, scale) in enumerate(replicas):
            frames = [