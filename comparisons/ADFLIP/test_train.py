"""Checks for train.py's two pieces of hand-written glue.

Run from a job-shaped directory (the ADFLIP checkout extracted beside this file), as
train.sh/test_multi_state.py do::

    python -m pytest test_train.py -q

The one that matters is `test_structure_data_matches_pdb_parse`. `structure_data_from_frame`
builds ADFLIP's input arrays straight from a trajectory frame's coordinates instead of writing a
PDB and parsing it back, because the parse dominates the per-example cost. That is only safe while
it agrees *exactly* with upstream's parser, so this pins the two against each other field by
field: a change to `parse_mmcif_to_structure_data`, to the atom order `write_record_pdb` writes,
or to the residue vocabulary breaks this test rather than silently training on mislabelled input.
"""
import os
import os.path as osp
import sys

import numpy as np
import pytest
import torch

HERE = osp.dirname(osp.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import train  # noqa: E402

# Upstream reads data/misc/* relatively at import time and pdb2data does the same at call time.
os.chdir(train.ADFLIP_ROOT)

from data import all_atom_parse as aap  # noqa: E402
from data.all_atom_dataset import AllAtomDataset  # noqa: E402

from relaxed_pdb import BACKBONE_ATOMS  # noqa: E402

import dataset_splits  # noqa: E402

RESNAMES = ['MET', 'ALA', 'GLY', 'SER', 'TRP', 'LYS', 'PRO', 'CYS']


def synthetic_coords(num_residues, seed=0):
    """``(R, 4, 3)`` Angstrom backbone coordinates in `BACKBONE_ATOMS` (CA, N, C, O) order.

    A roughly extended chain with ~3.8 A CA-CA spacing, which is what `detect_coord_scale` keys
    on, plus a little jitter so no two atoms coincide.
    """
    rng = np.random.default_rng(seed)
    coords = np.zeros((num_residues, len(BACKBONE_ATOMS), 3), dtype=np.float32)
    for i in range(num_residues):
        ca = np.array([3.8 * i, 0.0, 0.0], dtype=np.float32)
        coords[i, BACKBONE_ATOMS.index('CA')] = ca
        coords[i, BACKBONE_ATOMS.index('N')] = ca + [-1.2, 0.7, 0.0]
        coords[i, BACKBONE_ATOMS.index('C')] = ca + [1.2, 0.7, 0.0]
        coords[i, BACKBONE_ATOMS.index('O')] = ca + [1.4, 1.9, 0.0]
    return coords + rng.normal(0.0, 0.02, coords.shape).astype(np.float32)


@pytest.mark.parametrize('num_residues', [1, 12, 57])
def test_structure_data_matches_pdb_parse(tmp_path, num_residues):
    """The direct build equals writing the frame as a PDB and running upstream's parser on it.

    Uses the same `compare_to_pdb_parse` the job itself runs at startup, so the test and the
    runtime self-check cannot drift apart in what they consider agreement.
    """
    resnames = [RESNAMES[i % len(RESNAMES)] for i in range(num_residues)]
    problems = train.compare_to_pdb_parse(
        resnames, synthetic_coords(num_residues), aap, str(tmp_path / 'frame.pdb'))
    assert problems == [], '\n'.join(problems)


def test_every_residue_is_designable():
    """`is_center & is_protein & backbone_mask` -- what the loss and the logits are taken over --
    must select exactly one position per residue, or frames and labels would not line up."""
    resnames = [RESNAMES[i % len(RESNAMES)] for i in range(7)]
    built = train.structure_data_from_frame(resnames, synthetic_coords(len(resnames)), aap)
    designable = built.is_center & built.is_protein & built.backbone_mask
    assert int(designable.sum()) == len(resnames)
    assert np.array_equal(built.residue_index[designable], np.arange(len(resnames)))


def test_native_classes_match_the_alphabet():
    built_classes = train.native_classes(['ALA', 'TRP', 'GLY'], aap)
    assert built_classes.tolist() == [train.ALPHABET.index(c) for c in 'AWG']


def test_corrupt_at_zero_masks_every_residue():
    """t = 0 is the one-shot evaluation setting: nothing of the native sequence may survive."""
    resnames = [RESNAMES[i % len(RESNAMES)] for i in range(9)]
    built = train.structure_data_from_frame(resnames, synthetic_coords(len(resnames)), aap)
    corrupted = train.corrupt(built, AllAtomDataset, time=np.array([0.0]))
    mask_token = aap.residue_tokens['<MASK>']
    assert set(np.unique(corrupted.noisy_residue_token).tolist()) == {mask_token}
    # The native tokens are kept alongside, which is what the loss is scored against.
    assert not np.array_equal(corrupted.residue_token, corrupted.noisy_residue_token)


def test_corrupt_at_one_masks_nothing():
    resnames = [RESNAMES[i % len(RESNAMES)] for i in range(9)]
    built = train.structure_data_from_frame(resnames, synthetic_coords(len(resnames)), aap)
    corrupted = train.corrupt(built, AllAtomDataset, time=np.array([1.0]))
    assert np.array_equal(corrupted.noisy_residue_token, corrupted.residue_token)


def _entry(groups):
    """An index entry whose frames span several trajectory groups, pooled as the store pools them."""
    return {'frames': [(name, i) for name, n in groups for i in range(n)]}


def test_train_frames_strides_within_each_group():
    """The stride must restart per group: applying it to the pooled list would make which frames a
    replica contributes depend on how long the replicas before it happened to be."""
    entry = _entry([('g1', 35), ('g2', 35)])
    frames = train.train_frames(entry, 16)
    assert frames == [('g1', 0), ('g1', 16), ('g1', 32), ('g2', 0), ('g2', 16), ('g2', 32)]


def test_train_frames_stride_one_keeps_every_sheaf_frame():
    entry = _entry([('g1', 10), ('g2', 7)])
    assert train.train_frames(entry, 1) == entry['frames']


def test_eval_windows_match_proteinmpnn_sampler():
    """Windows of WINDOW_LEN starting every WINDOW_STEP, trailing partial window dropped --
    comparisons/ProteinMPNN's AvgLogitsBatchSampler(subsequence=128, step=32)."""
    length = dataset_splits.WINDOW_LEN + 3 * dataset_splits.WINDOW_STEP + 7
    entry = _entry([('g1', length)])
    windows = train.eval_windows(entry, window_stride=1)

    assert len(windows) == 4, 'the trailing partial window was not dropped'
    for i, window in enumerate(windows):
        assert len(window) == dataset_splits.WINDOW_LEN
        start = i * dataset_splits.WINDOW_STEP
        assert window == [('g1', j) for j in range(start, start + dataset_splits.WINDOW_LEN)]


def test_eval_windows_never_span_two_groups():
    entry = _entry([('g1', dataset_splits.WINDOW_LEN), ('g2', dataset_splits.WINDOW_LEN)])
    windows = train.eval_windows(entry, window_stride=1)
    assert len(windows) == 2
    for window in windows:
        assert len({key[0] for key in window}) == 1, 'a window straddles two replicas'


def test_eval_window_stride_thins_windows_without_narrowing_them():
    """Thinning must drop whole windows, never shorten them: a narrower window would quietly
    change what a scored example means instead of just how many there are."""
    length = dataset_splits.WINDOW_LEN + 7 * dataset_splits.WINDOW_STEP
    entry = _entry([('g1', length)])
    every = train.eval_windows(entry, window_stride=1)
    thinned = train.eval_windows(entry, window_stride=4)

    assert thinned == every[::4]
    assert all(len(window) == dataset_splits.WINDOW_LEN for window in thinned)


def test_eval_windows_empty_when_the_group_is_shorter_than_one_window():
    entry = _entry([('g1', dataset_splits.WINDOW_LEN - 1)])
    assert train.eval_windows(entry, window_stride=1) == []


class _FakeBatch:
    """Stands in for a batched `StructureData`: `to_device` only looks at its tensor attributes."""

    def __init__(self, **tensors):
        self.__dict__.update(tensors)


class _FakeModel:
    """Returns a fixed per-frame logit block, so the only thing under test is how `score_window`
    accumulates chunks -- not what the denoiser would have said."""

    def __init__(self, table):
        self.table = table  # (num_frames, num_residues, num_classes)
        self.model = self._forward

    def _forward(self, data, time_step):
        block = self.table[data['frame_ids'].tolist()]
        return block.reshape(-1, block.shape[-1]), None

    def eval(self):
        return self


def _window_item(table, chunk_size, targets):
    num_frames, num_residues = table.shape[0], table.shape[1]
    chunks = []
    for start in range(0, num_frames, chunk_size):
        ids = torch.arange(start, min(start + chunk_size, num_frames))
        zeros = torch.zeros(len(ids) * num_residues, dtype=torch.long)
        chunks.append((_FakeBatch(frame_ids=ids, noisy_residue_token=zeros,
                                  residue_token=zeros.clone(), time_step=torch.zeros(1)),
                       len(ids)))
    return {'protein_id': 'a', 'window': 'g1@0', 'chunks': chunks,
            'num_frames': num_frames, 'target': targets}


@pytest.mark.parametrize('reduction', train.FRAME_REDUCTIONS)
@pytest.mark.parametrize('chunk_size', [1, 7, 32, 128])
def test_window_average_is_independent_of_the_chunk_size(reduction, chunk_size):
    """Chunking exists only to bound memory: a window's score must not depend on how many frames
    were put through the denoiser at a time. If it did, --eval-chunk-size would silently become a
    third thing that changes the reported number."""
    torch.manual_seed(0)
    num_frames, num_residues = 128, 6
    table = torch.randn(num_frames, num_residues, max(train.STANDARD_TOKENS) + 1)
    targets = torch.arange(num_residues) % 20
    model = _FakeModel(table)
    device = torch.device('cpu')

    whole, _ = train.score_window(model, _window_item(table, num_frames, targets),
                                  reduction, device)
    chunked, _ = train.score_window(model, _window_item(table, chunk_size, targets),
                                    reduction, device)
    torch.testing.assert_close(chunked, whole)


def test_window_reduction_matches_proteinmpnn_log_prob_mean():
    """ProteinMPNN reduces a window with `log_softmax(log_probs.mean(0))`; `--frame-reduction
    logit` averages raw logits and takes one softmax. Those are the same thing -- the per-frame
    log-normalisers differ by a per-residue constant that log_softmax removes -- and this pins it,
    because if they ever diverged the two rows would stop being the same measurement."""
    torch.manual_seed(0)
    num_frames, num_residues = 16, 5
    table = torch.randn(num_frames, num_residues, max(train.STANDARD_TOKENS) + 1)
    targets = torch.arange(num_residues) % 20

    ours, _ = train.score_window(_FakeModel(table), _window_item(table, 4, targets),
                                 'logit', torch.device('cpu'))
    # ProteinMPNN's run_val, restricted to the same 20 columns.
    theirs = torch.log_softmax(
        torch.log_softmax(table, dim=-1).mean(dim=0)[:, train.STANDARD_TOKENS], dim=-1)
    torch.testing.assert_close(ours, theirs)


def test_aggregate_by_protein_averages_a_proteins_windows():
    per_window = [
        {'protein': 'a', 'length': 10, 'perplexity': 4.0, 'recovery_top1': 0.4,
         'recovery_top5': 0.8, 'recovery_top10': 0.9},
        {'protein': 'a', 'length': 10, 'perplexity': 6.0, 'recovery_top1': 0.6,
         'recovery_top5': 0.9, 'recovery_top10': 1.0},
        {'protein': 'b', 'length': 7, 'perplexity': 5.0, 'recovery_top1': 0.5,
         'recovery_top5': 0.85, 'recovery_top10': 0.95},
    ]
    rows = {row['title']: row for row in train.aggregate_by_protein(per_window)}

    assert rows['a']['n_windows'] == 2 and rows['b']['n_windows'] == 1
    assert rows['a']['perplexity'] == pytest.approx(5.0)
    assert rows['a']['recovery_top1'] == pytest.approx(0.5)
    assert rows['b']['perplexity'] == pytest.approx(5.0)


@pytest.mark.parametrize('reduction', train.FRAME_REDUCTIONS)
def test_frame_reduction_of_identical_frames_is_a_no_op(reduction):
    """Averaging a protein's frames must leave a single repeated frame's distribution untouched,
    whichever reduction is used -- otherwise the ensemble is biased by its own size."""
    torch.manual_seed(0)
    single = torch.randn(5, len(train.STANDARD_TOKENS) + 3)
    stacked = single.repeat(4, 1)
    per_frame = stacked.view(4, -1, stacked.shape[-1])
    if reduction == 'prob':
        reduced = torch.softmax(per_frame, dim=-1).mean(0)
    else:
        reduced = torch.softmax(per_frame.mean(0), dim=-1)
    torch.testing.assert_close(reduced, torch.softmax(single, dim=-1))
