"""Train ADFLIP on MD trajectory frames and score the held-out splits.

The zero-shot row (`eval_pretrained.py`) scores the released ICML checkpoint, which was trained on
~149K all-atom PDB entries. It does poorly here, and the likeliest reason is distribution rather
than capacity: with `--structure-source frame` the inputs are backbone-only single chains (N, CA,
C, O -- see `lib/traj_frames.frame_record`), while ADFLIP was built around an all-atom context with
ligands. This script retrains it on exactly the inputs it is evaluated on.

Training regime: per frame, as comparisons/PiFold's `--structure_source traj` does
---------------------------------------------------------------------------------
Each (protein, frame) pair is one training example -- upstream's own regime, which feeds
`Cluster_AllAtomDataset` a single structure per item, masks a fraction of its residues at a random
flow time *t*, and takes the denoising loss on that one structure (`trainer.py`). ADFLIP's
multi-state path (Algorithm 1, `multi_state.py`) is an *inference-time* ensembling procedure in the
paper, not a training one, so training per frame reproduces the authors' setup and keeps this row
comparable with PiFold's and MapDiff's `traj` rows, which also train on every frame independently.
Treating frames as separate examples is additionally the physically-honest version of the
coordinate jitter the denoiser already trains with (`augment_eps`).

Frame sampling and evaluation follow comparisons/ProteinMPNN/train.py
--------------------------------------------------------------------
Both the frames trained on and the way they are scored are copied from comparisons/ProteinMPNN,
so the two rows are read off the same procedure rather than two different ones. Concretely:

*Training frames* -- **every** frame the sheaf model trains on (`lib/traj_frames.TrajectoryStore.index`
-> `sheaf_frames`), thinned by `--train-frame-stride` (16, ProteinMPNN's own ``coords[:, ::16]``
training stride) within each trajectory group. This replaces the previous ``--k 20`` uniform draw,
which froze 20 frames per protein for the whole run and so gave the model no trajectory
augmentation at all; on mdCATH 320 K that is ~105K frames per epoch drawn from all 1.68M rather
than the same 87K every epoch.

*Evaluation windows* -- ProteinMPNN scores **windows**, not proteins
(`AvgLogitsBatchSampler(subsequence=128, step=32)`): within one trajectory group, every run of
`lib/dataset_splits.WINDOW_LEN` (128) consecutive frames starting every `WINDOW_STEP` (32) frames
is one scored example, and a trailing partial window is dropped. These are exactly the windows
`scripts/train_residue_classifier.py` cuts for the sheaf model, which is why those two constants
live in `dataset_splits` and are imported here rather than restated. The per-residue outputs of a
window's 128 frames are averaged and the window is then scored as a single example, so **one
protein contributes several metric samples** and the reported mean +/- std is over windows. A
secondary per-protein aggregate is written to the summary json as well, since the DynamicMPNN /
PiFold / MapDiff rows are per protein.

`--eval-window-stride` (4) keeps every Nth window rather than all of them. Each kept window is
still a faithful 128-frame average -- only how many windows a protein contributes changes -- and it
is what makes a per-epoch validation pass affordable here: ADFLIP featurises and runs every frame
individually, so all 11 windows of all 799 val proteins is ~1.1M forward passes (~30 min) per
epoch. Set it to 1 to reproduce ProteinMPNN's sampler exactly.

*Conditioning* -- every designable residue is masked (``t = 0``), one denoiser pass per frame, as
before. This deliberately does **not** copy ProteinMPNN's conditioning: `mpnn_forward` passes the
true sequence, so ProteinMPNN's perplexity is teacher-forced autoregressive (under a random
decoding order each residue sees roughly a Uniform(0,1) fraction of the true sequence), while
DynamicMPNN and the sheaf model are one-shot and de novo. Scoring ADFLIP at ``t = 0`` keeps it
comparable with those; its perplexity is therefore not directly comparable with ProteinMPNN's,
and is also inflated relative to all of them by the label smoothing upstream trains with
(`config/train_v1.yaml: training.label_smoothing`, applied over 33 classes in
`model/discrete_flow_aa.compute_loss`).

The frame reduction (`--frame-reduction`, `logit` by default) is already ProteinMPNN's: averaging
log-probs and re-normalising, as `run_val` does, is identical to averaging raw logits and taking
one softmax, because the per-frame log-normalisers are a per-residue constant that log_softmax
removes. `prob` averages the distributions themselves, which is what ADFLIP's Algorithm 1 does.

None of this is the iterative adaptive sampler: that is what `eval_pretrained.py` runs, and a
checkpoint saved here can be handed straight back to it (`--ckpt weights/<run>.pt`) to get the
sampled multi-state row for the same weights.

Metrics are `lib/stats_utils.ResidueMetrics` under the same W&B keys comparisons/DynamicMPNN
logs -- per-protein top-1/5/10 recovery, perplexity, per-residue precision/recall/f1, the
amino-acid confusion matrix, a `pred_seqs` table, `params`, `split_counts`, `val_split_ids` and
`test_split_ids` -- so all of these models read off one dashboard. ADFLIP's vocabulary is wider
than 20 tokens, so outputs are restricted to the 20 standard residues and renormalised before
scoring, exactly as `eval_pretrained.py` does it; residues whose native token is not one of the 20
are dropped.

Model selection follows the protocol every baseline here shares: validate every epoch, keep the
weights with the lowest per-protein val perplexity, early-stop after `--early-stopping-patience`
epochs without improvement, and score test once with the kept weights.

Usage:
    python train.py --data-name MDCATH --traj-h5 ./mdcath_spinet_320_0_v2.h5
    python train.py --from-scratch --data-name ATLAS --traj-h5 ./atlas_data.h5
"""
import argparse
import contextlib
import io
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = osp.dirname(osp.abspath(__file__))
ADFLIP_ROOT = osp.join(HERE, 'ADFLIP')
if ADFLIP_ROOT not in sys.path:
    sys.path.insert(0, ADFLIP_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import dataset_splits
    import relaxed_pdb
    import stats_utils
    import traj_frames
    from relaxed_pdb import BACKBONE_ATOMS
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = osp.join(HERE, _up, 'lib')
        if osp.isdir(_cand):
            sys.path.insert(0, osp.abspath(_cand))
            break
    import dataset_splits
    import relaxed_pdb
    import stats_utils
    import traj_frames
    from relaxed_pdb import BACKBONE_ATOMS

# The vocabulary restriction, the 20-class order and the checkpoint loader are shared with the
# zero-shot script rather than duplicated -- importing it also runs its check that the released
# checkpoint's 20 standard residues are still contiguous in ADFLIP's token table.
import eval_pretrained as ev  # noqa: E402

ALPHABET = ev.ALPHABET
STANDARD_TOKENS = ev.STANDARD_TOKENS
TOKEN_TO_CLASS = ev.TOKEN_TO_CLASS
DATA_NAMES = ev.DATA_NAMES

# Atom order `parse_mmcif_to_structure_data` sees in a PDB written by
# `lib.relaxed_pdb.write_record_pdb` (which writes `traj_frames.PDB_ATOM_ORDER`), as distinct from
# BACKBONE_ATOMS, the order the trajectory store's atom axis is in.
PDB_ATOM_ORDER = traj_frames.PDB_ATOM_ORDER
_REORDER = [BACKBONE_ATOMS.index(atom) for atom in PDB_ATOM_ORDER]

FRAME_REDUCTIONS = ('logit', 'prob')


# --------------------------------------------------------------------------------------
# Trajectory frame -> ADFLIP StructureData
# --------------------------------------------------------------------------------------

def structure_data_from_frame(resnames, coords, aap):
    """One backbone-only frame as the `StructureData` ADFLIP's parser would produce for it.

    Built directly from coordinates rather than by writing a PDB and parsing it back: the parse is
    the dominant cost per example and a trajectory frame is a single chain of standard residues
    with all four backbone atoms present, so every field is known up front. `test_train.py`'s
    `test_structure_data_matches_pdb_parse` pins this against `aap.pdb2data` on a written PDB so
    the shortcut cannot drift from upstream's parser.

    :param resnames: R canonical three-letter residue names.
    :param coords: ``(R, 4, 3)`` Angstrom coordinates in `BACKBONE_ATOMS` order.
    """
    num_residues = len(resnames)
    position = np.asarray(coords, dtype=np.float32)[:, _REORDER, :].reshape(-1, 3)
    num_atoms = position.shape[0]

    residue_token = np.repeat(
        np.array([aap.get_token_index(name) for name in resnames], dtype=np.int64),
        len(PDB_ATOM_ORDER))
    residue_index = np.repeat(np.arange(num_residues, dtype=np.int64), len(PDB_ATOM_ORDER))
    residue_atom_index = np.tile(np.arange(len(PDB_ATOM_ORDER), dtype=np.int64), num_residues)
    # `_build_residues` writes element = atom_name[0], which is what Biopython reads back.
    element_index = np.tile(
        np.array([aap.get_element_index(atom[0]) for atom in PDB_ATOM_ORDER], dtype=np.int64),
        num_residues)
    is_center = np.tile(np.array([atom == 'CA' for atom in PDB_ATOM_ORDER]), num_residues)

    true_mask = np.ones(num_atoms, dtype=bool)
    false_mask = np.zeros(num_atoms, dtype=bool)
    data = aap.StructureData(
        residue_token=residue_token,
        residue_index=residue_index,
        residue_atom_index=residue_atom_index,
        # `write_record_pdb` writes occupancy 1.0 / bfactor 0.0 for every backbone atom.
        occupancy=np.ones(num_atoms, dtype=np.float32),
        bfactor=np.zeros(num_atoms, dtype=np.float32),
        chain_id=np.zeros(num_atoms, dtype=np.int64),
        position=position,
        element_index=element_index,
        is_ion=false_mask.copy(),
        is_protein=true_mask.copy(),
        is_nucleotide=false_mask.copy(),
        is_center=is_center,
        # Every atom here IS a backbone atom, and all four are present in every residue.
        is_backbone=true_mask.copy(),
        backbone_mask=true_mask.copy(),
        not_pad_mask=true_mask.copy(),
    )
    # Fills interact_{non_protein,ion,nucleotide,molecule}_res; all False for a ligand-free chain,
    # but the fields have to exist because the collate and the denoiser both read them.
    return aap.interact_residue(data)


#: Fields `structure_data_from_frame` must reproduce exactly. `position` is checked separately,
#: to a tolerance: a PDB stores coordinates to three decimals, so the round trip is lossy.
PARSE_CHECKED_FIELDS = (
    'residue_token', 'residue_index', 'residue_atom_index', 'chain_id', 'element_index',
    'is_ion', 'is_protein', 'is_nucleotide', 'is_center', 'is_backbone', 'backbone_mask',
    'not_pad_mask', 'interact_non_protein_res', 'interact_ion_res', 'interact_nucleotide_res',
    'interact_molecule_res',
)


def compare_to_pdb_parse(resnames, coords, aap, pdb_path):
    """``[]`` if `structure_data_from_frame` agrees with upstream's own parser on this frame.

    Writes the frame out as a PDB exactly as `eval_pretrained.py` does and runs `aap.pdb2data`
    over it, then diffs that against the direct build. Returns a list of human-readable
    mismatches, so both `test_train.py` and the start-of-run self-check can use one definition of
    "agrees".
    """
    record = traj_frames.frame_record(
        'builder_check', list(resnames), ['X'] * len(resnames), np.asarray(coords),
        frame_index=0, num_frames=1, path='builder-check')
    relaxed_pdb.write_record_pdb(record, pdb_path)
    parsed = aap.pdb2data(pdb_path, device='cpu')
    built = structure_data_from_frame(resnames, coords, aap)

    problems = []
    for field in PARSE_CHECKED_FIELDS:
        expected = parsed[field].cpu().numpy()
        actual = np.asarray(getattr(built, field))
        if actual.shape != expected.shape:
            problems.append('{}: shape {} but the parser gives {}'.format(
                field, actual.shape, expected.shape))
        elif not np.array_equal(actual, expected):
            problems.append('{}: {} of {} entries differ'.format(
                field, int((actual != expected).sum()), actual.size))
    position = np.asarray(built.position)
    parsed_position = parsed['position'].cpu().numpy()
    if position.shape != parsed_position.shape:
        problems.append('position: shape {} but the parser gives {}'.format(
            position.shape, parsed_position.shape))
    elif not np.allclose(position, parsed_position, atol=1e-3):
        problems.append('position: largest difference {:.4f} A exceeds the 1e-3 a PDB\'s three '
                        'decimals explain'.format(float(np.abs(position - parsed_position).max())))
    return problems


def check_structure_builder(entry, store, aap):
    """Fail the run if the direct build disagrees with upstream's parser on a real frame.

    `structure_data_from_frame` exists to skip a PDB write-and-reparse per training example, which
    is the dominant per-example cost. The saving is only safe while the two agree, and a
    disagreement would not crash -- it would train and score on mislabelled input and report a
    number. So it is checked once per run, here, against the job's own data, rather than trusted.
    """
    import tempfile
    coords = store.read_frames(entry, entry['frames'][:1])[0]
    with tempfile.TemporaryDirectory(prefix='adflip_builder_check_') as work_dir:
        problems = compare_to_pdb_parse(entry['resnames'], coords, aap,
                                        osp.join(work_dir, 'check.pdb'))
    if problems:
        raise RuntimeError(
            'structure_data_from_frame disagrees with ADFLIP\'s own PDB parser:\n  {}\nTraining '
            'would silently use mislabelled input. Reconcile it with '
            'data/all_atom_parse.parse_mmcif_to_structure_data (see test_train.py).'.format(
                '\n  '.join(problems)))
    print('>>> structure builder agrees with ADFLIP\'s PDB parser on {} fields'.format(
        len(PARSE_CHECKED_FIELDS) + 1))


def native_classes(resnames, aap):
    """``(R,)`` 0..19 class index per residue, -1 where the native residue is not one of the 20
    standard ones (dropped from the metrics, as in `eval_pretrained.score_protein`)."""
    return np.array([TOKEN_TO_CLASS.get(aap.get_token_index(name), -1) for name in resnames],
                    dtype=np.int64)


def corrupt(data, dataset_cls, time=None):
    """Upstream's masking, reused rather than reimplemented.

    `AllAtomDataset.corrupt` touches `self` only for the `debug_corrupt` flag, so it is called
    unbound with a stand-in. It mutates and rebuilds ``data``, so pass a freshly built one.
    ``time=None`` draws t ~ U(0, 1) (training); ``time=np.array([0.0])`` masks every designable
    residue, which is the one-shot evaluation setting.
    """
    class _Shim:
        cfg = None
    return dataset_cls.corrupt(_Shim(), data, time=time, mask_chain_id=[0])


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------

def group_frames(entry):
    """``{group path: [frame key, ...]}`` -- this protein's frames split by trajectory group.

    `TrajectoryStore.index` pools every replica into one ``frames`` list, group by group in store
    order. Both the training stride and the evaluation windows have to be cut *within* a group:
    a window spanning the boundary between two replicas would average frames from two unrelated
    simulations and call it a trajectory window.
    """
    groups = {}
    for key in entry['frames']:
        groups.setdefault(key[0], []).append(key)
    return groups


def train_frames(entry, stride):
    """Every sheaf frame, thinned by ``stride`` within each group (ProteinMPNN's train stride).

    comparisons/ProteinMPNN takes ``coords[:, ::16]`` for the training split and leaves val/test
    whole. The first frame of each group is always kept, so a group never contributes nothing.
    """
    if stride < 1:
        raise ValueError('--train-frame-stride must be >= 1, got {}'.format(stride))
    frames = []
    for keys in group_frames(entry).values():
        frames.extend(keys[::stride])
    return frames


def eval_windows(entry, window_stride, window_len=dataset_splits.WINDOW_LEN,
                 window_step=dataset_splits.WINDOW_STEP):
    """``[[frame key, ...], ...]`` -- ProteinMPNN's `AvgLogitsBatchSampler` windows.

    Within each group: runs of ``window_len`` consecutive frames starting at 0, ``window_step``,
    ..., with a trailing partial window dropped (upstream's
    ``range(0, len(batch) - subsequence + 1, step)``). ``window_stride`` then keeps every Nth of
    those windows -- 1 reproduces the sampler exactly, larger values thin how many windows a
    protein contributes without changing any window's width.

    A group shorter than one window yields none, as upstream's sampler does. `sheaf_frames`
    already guarantees at least ``WINDOW_LEN`` frames per usable group, so this only bites when
    `--frame-stride` has thinned them below that -- `main` checks for it and says so.
    """
    if window_stride < 1:
        raise ValueError('--eval-window-stride must be >= 1, got {}'.format(window_stride))
    windows = []
    for keys in group_frames(entry).values():
        starts = range(0, len(keys) - window_len + 1, window_step)
        windows.extend([keys[start:start + window_len] for start in starts][::window_stride])
    return windows


class FrameDataset(torch.utils.data.Dataset):
    """One (protein, frame) training example per item, corrupted at a random flow time."""

    def __init__(self, store, entries, frames_by_protein, aap, dataset_cls):
        self.store = store
        self.entries = entries
        self.aap = aap
        self.dataset_cls = dataset_cls
        self.items = [(protein_id, key)
                      for protein_id, keys in frames_by_protein.items() for key in keys]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        protein_id, frame_key = self.items[idx]
        entry = self.entries[protein_id]
        try:
            coords = self.store.read_frames(entry, [frame_key])[0]
            data = structure_data_from_frame(entry['resnames'], coords, self.aap)
            return corrupt(data, self.dataset_cls)
        except Exception as exc:  # noqa: BLE001 -- one bad frame must not lose the epoch
            print('>>> {} frame {} failed to featurise: {}'.format(protein_id, frame_key, exc),
                  file=sys.stderr)
            return None


class WindowDataset(torch.utils.data.Dataset):
    """One evaluation **window** per item: `WINDOW_LEN` consecutive frames of one trajectory
    group, each masked completely (t = 0).

    The window's frames are returned as several pre-batched chunks of `chunk_size` rather than one
    batch: ProteinMPNN can put 128 frames through at once because it runs dense (B, L, 4, 3)
    tensors, while ADFLIP expands every frame into per-atom `StructureData` and runs a GNN over
    it, so a 128-frame window of a 512-residue protein is ~260K atoms in one graph. Chunking
    bounds the peak without changing the result -- `run_validation` sums the chunks' outputs and
    divides by the window's frame count, which is the same average either way.
    """

    def __init__(self, store, entries, windows, aap, dataset_cls, chunk_size):
        self.store = store
        self.entries = entries
        self.aap = aap
        self.dataset_cls = dataset_cls
        self.chunk_size = chunk_size
        #: ``(protein_id, window label, [frame key, ...])``, proteins in split order.
        self.items = [(protein_id, '{}@{}'.format(keys[0][0], keys[0][1]), keys)
                      for protein_id, protein_windows in windows.items()
                      for keys in protein_windows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        protein_id, label, frame_keys = self.items[idx]
        entry = self.entries[protein_id]
        try:
            coords = self.store.read_frames(entry, frame_keys)
            chunks = []
            for start in range(0, len(coords), self.chunk_size):
                states = [corrupt(structure_data_from_frame(entry['resnames'], frame, self.aap),
                                  self.dataset_cls, time=np.array([0.0]))
                          for frame in coords[start:start + self.chunk_size]]
                # Every frame is the same chain, so no padding happens inside a chunk.
                batch = self.aap.batch_structure_data_list(states, to_torch=True)
                if batch is None:
                    raise ValueError('batch_structure_data_list returned None')
                chunks.append((batch, len(states)))
            return {'protein_id': protein_id, 'window': label, 'chunks': chunks,
                    'num_frames': len(coords),
                    'target': torch.from_numpy(native_classes(entry['resnames'], self.aap))}
        except Exception as exc:  # noqa: BLE001
            print('>>> {} window {} failed to featurise: {}'.format(protein_id, label, exc),
                  file=sys.stderr)
            return None


def make_collate(aap):
    def collate(items):
        items = [item for item in items if item is not None]
        if not items:
            return None
        return aap.batch_structure_data_list(items, to_torch=True)
    return collate


def _first(items):
    """Collate for `WindowDataset` (batch_size 1): the item itself, already batched in chunks."""
    return items[0]


def seed_worker(worker_id):
    """Give each DataLoader worker its own numpy/random stream.

    PyTorch reseeds torch per worker and per epoch but leaves `numpy` and `random` alone, and
    upstream's `corrupt` draws the flow time *t* and the masking pattern with `np.random`. Without
    this every worker would corrupt its frames with the *same* sequence of t's, so a batch would
    cover far less of the time axis than it looks like it does.
    """
    import random
    worker_seed = torch.initial_seed() % (2 ** 31)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------------------

def to_device(batch, device):
    data = {k: v.to(device) for k, v in batch.__dict__.items() if torch.is_tensor(v)}
    return {k: v.float() if v.dtype == torch.float64 else v for k, v in data.items()}


@contextlib.contextmanager
def quiet():
    """Swallow the denoiser's own stdout for one call.

    `Zoidberg_GNN.forward` prints "No non-protein residue, skip context block" every time it is
    handed a structure with no ligand -- which is every protein here, since MD frames are
    ligand-free. One line per protein per epoch buries the log. (It is the only print on that
    path; nothing else is being hidden.)
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield


def aggregate_by_protein(per_window, metric_keys=stats_utils.METRIC_KEYS):
    """Window rows -> one row per protein, each metric averaged over that protein's windows.

    The headline numbers here are per *window*, because that is what comparisons/ProteinMPNN
    reports. The DynamicMPNN / PiFold / MapDiff rows are per protein, so this gives the same run a
    per-protein column to line up against them. It is a plain mean over a protein's windows, which
    is not the same as scoring the protein's whole trajectory at once -- it weights every protein
    equally instead of weighting by how many windows its trajectory happens to yield.
    """
    by_protein = {}
    for row in per_window:
        by_protein.setdefault(row['protein'], []).append(row)
    out = []
    for protein_id, rows in by_protein.items():
        summary = {'title': protein_id, 'length': rows[0]['length'], 'n_windows': len(rows)}
        for key in metric_keys:
            summary[key] = float(np.mean([row[key] for row in rows]))
        out.append(summary)
    return out


@torch.no_grad()
def score_window(model, item, reduction, device):
    """``(log_probs, targets)`` over the 20 standard residues for one evaluation window.

    The window's frames arrive as chunks (`WindowDataset`); each is run separately and the
    per-residue outputs are summed, then divided by the window's frame count. Summing and dividing
    once is what makes the chunking invisible: it is the same mean the whole window in one batch
    would give.
    """
    total, num_frames = None, 0
    for batch, chunk_frames in item['chunks']:
        data = to_device(batch, device)
        # Swap in the corrupted tokens, exactly as `DiscreteFlow_AA.forward` / `.test` do.
        # Without this the denoiser would read the NATIVE residue_token field and the
        # "prediction" would be the answer handed back.
        noisy = dict(data)
        noisy['residue_token'] = data['noisy_residue_token']
        with quiet():
            logits, _ = model.model(noisy, data['time_step'])
        if logits.shape[0] % chunk_frames:
            raise RuntimeError('{} logit rows do not split into {} frames'.format(
                logits.shape[0], chunk_frames))
        per_frame = logits.reshape(chunk_frames, -1, logits.shape[-1]).float()

        # Frames are reduced before the 20-class restriction, so the reduction happens over
        # ADFLIP's own output space. 'logit' is the plain mean of the raw outputs, which is
        # exactly ProteinMPNN's `log_probs.mean(0)` followed by a log_softmax -- the per-frame
        # log-normalisers it drops are a per-residue constant. 'prob' averages the distributions
        # themselves, which is what Algorithm 1 does.
        chunk_total = (F.softmax(per_frame, dim=-1) if reduction == 'prob' else per_frame).sum(0)
        total = chunk_total if total is None else total + chunk_total
        num_frames += chunk_frames

    if num_frames != item['num_frames']:
        raise RuntimeError('scored {} frames but the window holds {}'.format(
            num_frames, item['num_frames']))
    mean = total / num_frames
    probs = (mean if reduction == 'prob' else F.softmax(mean, dim=-1))[:, STANDARD_TOKENS]
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)
    log_probs = probs.clamp_min(torch.finfo(probs.dtype).tiny).log().cpu()

    targets = item['target']
    if log_probs.shape[0] != targets.shape[0]:
        raise RuntimeError('{} residues scored but {} native labels'.format(
            log_probs.shape[0], targets.shape[0]))
    keep = targets >= 0
    if not bool(keep.any()):
        raise ValueError('no standard residues to score')
    return log_probs[keep], targets[keep]


@torch.no_grad()
def run_validation(model, loader, epoch, run, num_params=0, split_name='val',
                   reduction='logit', device=None):
    """Score one held-out split window by window, and log DynamicMPNN's key set.

    One row per *window* (`WindowDataset`), so a protein contributes as many samples as it has
    windows and the mean +/- std is over windows -- comparisons/ProteinMPNN's `run_val`
    convention. `aggregate_by_protein` turns the same rows into a per-protein view.
    """
    model.eval()
    metrics = stats_utils.ResidueMetrics(stats_utils.three_letter_codes(ALPHABET),
                                         val_name=split_name)
    per_window, failures = [], {}

    for item in tqdm(loader, total=len(loader), desc='{} eval'.format(split_name.capitalize())):
        if item is None:
            continue
        window_id = '{}|{}'.format(item['protein_id'], item['window'])
        try:
            log_probs, targets = score_window(model, item, reduction, device)
        except Exception as exc:  # noqa: BLE001 -- one bad window should not lose the pass
            failures[window_id] = '{}: {}'.format(type(exc).__name__, exc)
            print('>>> {} failed: {}'.format(window_id, failures[window_id]), file=sys.stderr)
            continue

        nll = F.cross_entropy(log_probs, targets).item()
        design = ''.join(ALPHABET[i] for i in log_probs.argmax(dim=-1).tolist())
        metrics.add_protein(log_probs, targets, pdb_id=window_id, nll=nll, sequence=design)
        per_window.append({
            'title': window_id,
            'protein': item['protein_id'],
            'length': int(targets.shape[0]),
            'n_frames': item['num_frames'],
            'perplexity': stats_utils.perplexity_from_nll(torch.tensor(nll)),
            'recovery_top1': stats_utils.top_k_acc(log_probs, targets, 1),
            'recovery_top5': stats_utils.top_k_acc(log_probs, targets, 5),
            'recovery_top10': stats_utils.top_k_acc(log_probs, targets, 10),
            'design': design,
        })

    if not per_window:
        raise RuntimeError('every {} window failed to score'.format(split_name))

    log_dict = metrics.to_log_dict(
        cm_title='{} Amino Acid Confusion Matrix'.format(split_name.capitalize()), epoch=epoch)
    if run:
        run.log(log_dict)
    print(metrics.summary_line(num_params))
    return log_dict, per_window, failures


class EarlyStopping:
    """Early stopping on val perplexity, keeping the best epoch's weights for test.

    The protocol comparisons/{DynamicMPNN,MapDiff,PiFold} share: validate every epoch, keep the
    lowest per-protein val perplexity (`val_perp_mean`), stop once `patience` epochs pass without
    beating it, and score test once with the kept weights, so the test split never influences
    either the weights or the choice of epoch. `patience` 0 disables stopping but still selects.
    """

    def __init__(self, patience=10):
        self.patience = patience
        self.best_perplexity, self.best_epoch, self.best_state = float('inf'), None, None

    def update(self, model, perplexity, epoch):
        if not perplexity < self.best_perplexity:  # also rejects NaN
            return False
        self.best_perplexity, self.best_epoch = perplexity, epoch
        self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print('>>> new best val perplexity {:.4f} at epoch {}'.format(perplexity, epoch))
        return True

    def should_stop(self, epoch):
        if self.patience <= 0 or self.best_epoch is None:
            return False
        if epoch - self.best_epoch < self.patience:
            return False
        print('>>> early stopping at epoch {}: val perplexity has not improved in {} epochs '
              '(best {:.4f} at epoch {})'.format(epoch, epoch - self.best_epoch,
                                                 self.best_perplexity, self.best_epoch))
        return True

    def restore(self, model):
        if self.best_state is None:
            print('>>> no finite val perplexity was ever recorded; testing the final weights')
            return
        model.load_state_dict(self.best_state)
        print('>>> restored the best val epoch ({}, perplexity {:.4f})'.format(
            self.best_epoch, self.best_perplexity))


def save_checkpoint(path, config, flow, ema, optimizer, step, epoch):
    """Upstream's checkpoint layout, so `eval_pretrained.py --ckpt <path>` rescores a trained run
    with the real adaptive sampler (`test/design.py:load_flow_model` reads exactly these keys)."""
    payload = {'config': config, 'step': step, 'epoch': epoch,
               'model': flow.state_dict(), 'opt': optimizer.state_dict()}
    if ema is not None:
        payload['ema'] = ema.state_dict()
    tmp = str(path) + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------

def create_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', default=osp.join(ADFLIP_ROOT, 'results/weights/ADFLIP_v1.pt'),
                        help='Checkpoint to initialise from (the released weights by default -- '
                             'fine-tuning, since zero-shot scores poorly on backbone-only input). '
                             'The model is built from this checkpoint\'s own config.')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Ignore --ckpt and train from a random init, built from --config')
    parser.add_argument('--config', default=osp.join(ADFLIP_ROOT, 'config/train_v1.yaml'),
                        help='--from-scratch only: upstream\'s training config (architecture + '
                             'optimizer defaults)')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', default='./weights',
                        help='Checkpoints land here as <run name>.pt / <run name>_best.pt')
    parser.add_argument('--res-dir', default='./results/train',
                        help='Where the summary json and per-protein metrics are written')

    # dataset -- same meaning and defaults as the other comparisons' scripts
    parser.add_argument('--data-name', default='MDCATH', choices=sorted(DATA_NAMES))
    parser.add_argument('--index-csv', default='', type=str)
    parser.add_argument('--traj-h5', default=None, type=str, required=False)
    parser.add_argument('--val-fold', default=dataset_splits.DEFAULT_VAL_FOLD, type=int)
    parser.add_argument('--test-fold', default=dataset_splits.DEFAULT_TEST_FOLD, type=int)
    parser.add_argument('--max-length', default=512, type=int,
                        help="Drop proteins longer than this many residues. Defaults to ADFLIP's "
                             'own max_num_residues (512) rather than the 3000 the other '
                             'comparisons use: upstream crops longer structures around a random '
                             'centre, which would score a different residue set per epoch. The '
                             'dropped ids are logged and written to the summary so the held-out '
                             'sets can be diffed against the other models.')
    parser.add_argument('--frame-stride', default=1, type=int,
                        help="Thin each trajectory group's sheaf frames before anything else "
                             'sees them. Leave at 1: it also thins the evaluation windows, and '
                             'below WINDOW_LEN frames a group stops yielding any')
    parser.add_argument('--train-frame-stride', default=16, type=int,
                        help="Keep every Nth training frame within each trajectory group. 16 is "
                             "comparisons/ProteinMPNN's own training stride (coords[:, ::16]); "
                             '1 trains on every sheaf frame')
    parser.add_argument('--eval-window-stride', default=4, type=int,
                        help='Keep every Nth evaluation window. 1 reproduces ProteinMPNN\'s '
                             'AvgLogitsBatchSampler exactly (every window of WINDOW_LEN frames '
                             'starting every WINDOW_STEP); larger values thin how many windows a '
                             'protein contributes without changing any window\'s width')
    parser.add_argument('--eval-chunk-size', default=32, type=int,
                        help='Frames per forward pass when scoring a window. Bounds peak memory; '
                             'the window average is identical whatever this is set to')
    parser.add_argument('--limit', default=0, type=int,
                        help='Use only the first N proteins of each split (0 = all), for smoke tests')

    # optimisation -- upstream's config/train_v1.yaml values
    parser.add_argument('--batch-size', default=8, type=int,
                        help='Frames per forward pass')
    parser.add_argument('--effective-batch-size', default=64, type=int,
                        help="Frames per optimizer step (upstream's batch_size), reached by "
                             'accumulating gradients over effective/--batch-size passes')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help="Adam lr. Upstream trains from scratch at 1e-3; the default here is a "
                             'tenth of that because the default run fine-tunes released weights '
                             '(our choice, not theirs). Pass --lr 1e-3 with --from-scratch.')
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--clip-grad-norm', default=1.0, type=float)
    parser.add_argument('--epochs', default=200, type=int,
                        help='Upper bound on epochs; early stopping normally ends the run first')
    parser.add_argument('--early-stopping-patience', default=10, type=int,
                        help='Stop once val perplexity has not improved for this many epochs; '
                             '0 disables stopping (the best epoch is still restored)')
    parser.add_argument('--max-hours', default=None, type=float,
                        help='Wall-clock budget from process start. Training stops after the last '
                             'epoch that still fits, judging by the previous one; the best val '
                             'epoch is then restored and scored on test as usual')
    parser.add_argument('--no-ema', dest='ema', action='store_false',
                        help="Disable the EMA copy upstream trains and evaluates with")
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--seed', default=dataset_splits.SEED, type=int)
    parser.add_argument('--frame-reduction', default='logit', choices=FRAME_REDUCTIONS,
                        help="How a protein's frames are combined at eval time: 'logit' averages "
                             "the raw outputs, 'prob' averages the distributions (Algorithm 1)")

    parser.add_argument('--wandb', default=1, type=int, help='Log to W&B (0 disables)')
    parser.add_argument('--wandb-key-file', default='./wandb_api.txt')
    parser.add_argument('--wandb-entity', default='jenslundsgaard7-uw-madison')
    parser.add_argument('--wandb-project', default='SheafProtein')
    parser.add_argument('--wandb-run-name', default='', type=str)
    return parser.parse_args()


PATH_ARGS = ('ckpt', 'config', 'traj_h5', 'index_csv', 'res_dir', 'save_dir', 'wandb_key_file')


def resolve_paths(args, ds_name, origin):
    """Pin every path argument to the directory the job started in: `main` chdirs into the
    checkout, because upstream reads `data/misc/*` and `PIPPack/model_weights/*` relatively."""
    for name in PATH_ARGS:
        value = getattr(args, name)
        if value:
            setattr(args, name, osp.abspath(osp.join(origin, value)))
    if not args.index_csv:
        args.index_csv = osp.abspath(osp.join(origin, dataset_splits.default_index_csv(ds_name)))
    return args


def build_model(args, device, design):
    """``(flow, config, checkpoint)`` -- from the checkpoint's own config, or upstream's yaml.

    Built **without** `sidechain_packing`, unlike `test/design.py:load_flow_model`: repacking is
    part of the iterative sampler, and training never samples. That keeps PIPPack out of the
    training job entirely. It does not change which tensors are trained either way -- upstream
    keeps its PIPPack models in a plain list, so they are never registered submodules -- and
    `eval_pretrained.py` rebuilds the model with packing when it scores a checkpoint.
    """
    from model.discrete_flow_aa import DiscreteFlow_AA
    from data.residue_config import configure as configure_residues

    if args.from_scratch:
        import yaml
        with open(args.config) as handle:
            config = design.Config(yaml.safe_load(handle))
        configure_residues(include_nonstd_amino_acids=config.data.include_nonstd_amino_acids)
        flow = DiscreteFlow_AA(config, design.build_denoiser(config, device), min_t=0.0)
        print('>>> built a fresh model from {}'.format(args.config))
        return flow.to(device), config, None

    # `load_design_module` has registered the names the pickled config resolves against.
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    configure_residues(include_nonstd_amino_acids=config.data.include_nonstd_amino_acids)
    flow = DiscreteFlow_AA(config, design.build_denoiser(config, device), min_t=0.0)
    flow.load_state_dict(checkpoint['model'])
    print('>>> initialised from {} (step {})'.format(args.ckpt, checkpoint.get('step')))
    return flow.to(device), config, checkpoint


def main():
    start_time = time.monotonic()
    args = create_parser()
    if args.effective_batch_size % args.batch_size:
        raise ValueError('--effective-batch-size {} is not a multiple of --batch-size {}'.format(
            args.effective_batch_size, args.batch_size))
    accum_steps = args.effective_batch_size // args.batch_size
    device = torch.device(args.device)
    ds_name = DATA_NAMES[args.data_name]

    origin = os.getcwd()
    args = resolve_paths(args, ds_name, origin)
    if not args.traj_h5:
        raise ValueError('--traj-h5 is required: training reads MD frames from the store')
    os.chdir(ADFLIP_ROOT)
    print('>>> working directory: {} (upstream resolves its data/ files relative to it)'.format(
        ADFLIP_ROOT))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from data import all_atom_parse as aap
    from data.all_atom_dataset import AllAtomDataset
    design = ev.load_design_module()

    flow, config, checkpoint = build_model(args, device, design)
    num_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print('>>> ADFLIP: {} trainable parameters'.format(num_params))

    # ----------------------------------------------------------------------------------
    # Splits and frames
    # ----------------------------------------------------------------------------------
    train_ids, val_ids, test_ids = dataset_splits.get_splits(
        ds_name, index_csv=args.index_csv or None,
        val_fold=args.val_fold, test_fold=args.test_fold)
    if args.limit:
        train_ids, val_ids, test_ids = (ids[:args.limit] for ids in (train_ids, val_ids, test_ids))

    store = traj_frames.TrajectoryStore(args.traj_h5)
    entries, dropped = store.index(train_ids + val_ids + test_ids, ds_name,
                                   stride=args.frame_stride, max_length=args.max_length)
    train_by_protein = {pid: train_frames(entry, args.train_frame_stride)
                        for pid, entry in entries.items()}
    windows_by_protein = {pid: eval_windows(entry, args.eval_window_stride)
                          for pid, entry in entries.items()}

    if not entries:
        raise RuntimeError('no usable proteins in {} -- {} were dropped, e.g. {}'.format(
            args.traj_h5, len(dropped), dict(list(dropped.items())[:3])))
    # Before anything is trained on, confirm the fast featuriser still matches upstream's parser.
    check_structure_builder(entries[next(iter(entries))], store, aap)

    splits = {}
    for name, ids in (('train', train_ids), ('val', val_ids), ('test', test_ids)):
        splits[name] = [pid for pid in ids if pid in entries]
        print('>>> {}: {}/{} proteins usable'.format(name, len(splits[name]), len(ids)))
    for name in ('train', 'val', 'test'):
        if not splits[name]:
            raise RuntimeError(
                'no usable {} proteins -- every id in that split was dropped while indexing the '
                'trajectory store (too long for --max-length {}, or no trajectory group with all '
                'of {}). First few reasons: {}'.format(
                    name, args.max_length, dataset_splits.SHEAF_REQUIRED_DATASETS,
                    dict(list(dropped.items())[:3])))

    # A trajectory group shorter than one window yields none, as ProteinMPNN's sampler does.
    # `sheaf_frames` rules that out, so it can only happen once --frame-stride has thinned a group
    # below WINDOW_LEN -- which would drop those proteins from the scored splits silently, while
    # the "N/M proteins usable" counts above still looked right. Train needs no windows.
    no_windows = sorted(pid for name in ('val', 'test') for pid in splits[name]
                        if not windows_by_protein[pid])
    if no_windows:
        raise RuntimeError(
            '{} val/test protein(s) yield no {}-frame evaluation window, e.g. {} -- --frame-stride '
            '{} has thinned their trajectory groups below one window. Leave it at 1 (the sheaf '
            'frames whole) and thin the evaluation with --eval-window-stride instead.'.format(
                len(no_windows), dataset_splits.WINDOW_LEN, no_windows[:3], args.frame_stride))

    def windows_of(ids):
        return {pid: windows_by_protein[pid] for pid in ids}

    train_dataset = FrameDataset(
        store, entries, {pid: train_by_protein[pid] for pid in splits['train']}, aap, AllAtomDataset)
    val_dataset = WindowDataset(store, entries, windows_of(splits['val']), aap, AllAtomDataset,
                                args.eval_chunk_size)
    test_dataset = WindowDataset(store, entries, windows_of(splits['test']), aap, AllAtomDataset,
                                 args.eval_chunk_size)
    print('>>> {} training frames from {} proteins (every {} sheaf frame)'.format(
        len(train_dataset), len(splits['train']), args.train_frame_stride))
    for name, dataset in (('val', val_dataset), ('test', test_dataset)):
        print('>>> {}: {} windows of {} frames from {} proteins (every {} window)'.format(
            name, len(dataset), dataset_splits.WINDOW_LEN, len(splits[name]),
            args.eval_window_stride))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, collate_fn=make_collate(aap),
        worker_init_fn=seed_worker, persistent_workers=args.num_workers > 0)
    if len(train_loader) == 0:
        raise RuntimeError('{} training frames is less than one --batch-size {} batch'.format(
            len(train_dataset), args.batch_size))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=_first)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=_first)

    # ----------------------------------------------------------------------------------
    # W&B -- the same keys comparisons/DynamicMPNN logs
    # ----------------------------------------------------------------------------------
    ds_tag = dataset_splits.dataset_tag(ds_name, args.traj_h5)
    run_name = args.wandb_run_name or 'ADFLIP_{}{}'.format(
        ds_tag, '_scratch' if args.from_scratch else '')
    run = None
    if args.wandb:
        run = stats_utils.init_wandb(
            args.wandb_key_file, args.wandb_entity, args.wandb_project, run_name,
            config=dict(vars(args), params=num_params, ds_tag=ds_tag,
                        train_frames=len(train_dataset)))
        run.summary['split_counts'] = {name: len(ids) for name, ids in splits.items()}
        run.summary['val_split_ids'] = splits['val']
        run.summary['test_split_ids'] = splits['test']
        run.summary['dropped_proteins'] = dropped
        run.log({'params': num_params})

    # ----------------------------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = None
    if args.ema:
        from ema_pytorch import EMA
        ema = EMA(flow, beta=config.training.ema_beta,
                  update_every=config.training.ema_update_every).to(device)
        # Restore the checkpoint's EMA copy alongside its online weights, as upstream's
        # `Trainer.load` does -- the released model's quality lives in the EMA copy, and it is
        # what the first validation pass (and `eval_pretrained.py`) scores.
        if checkpoint is not None and 'ema' in checkpoint:
            ema.load_state_dict(checkpoint['ema'])
            print('>>> restored the EMA copy from {}'.format(args.ckpt))
    # Upstream evaluates (and ships) the EMA copy when it has one.
    eval_model = ema.ema_model if ema is not None else flow

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, '{}.pt'.format(run_name))
    best_path = osp.join(save_dir, '{}_best.pt'.format(run_name))
    if run:
        run.summary['weights_path'] = osp.basename(save_path)
        run.summary['best_weights_path'] = osp.basename(best_path)

    stopper = EarlyStopping(args.early_stopping_patience)
    deadline = start_time + args.max_hours * 3600 if args.max_hours else None
    stopped_epoch, step = args.epochs - 1, 0

    # Score the starting weights before any update, as comparisons/DynamicMPNN's finetune.py does.
    # It is the zero-shot number on exactly these proteins under exactly this metric, so the
    # fine-tuning curve starts from a comparable point -- and it enters model selection, so if no
    # epoch ever beats it, test scores the released weights rather than something worse.
    if not args.from_scratch:
        initial, _, _ = run_validation(eval_model, val_loader, -1, run,
                                       num_params=num_params, split_name='val',
                                       reduction=args.frame_reduction, device=device)
        print('>>> epoch -1 (released weights): val perplexity {:.4f}, top-1 {:.4f}'.format(
            initial['val_perp_mean'], initial['val_top1_acc_mean']))
        stopper.update(eval_model, initial['val_perp_mean'], -1)

    for epoch in range(args.epochs):
        epoch_start = time.monotonic()
        flow.train()
        total_loss, window_loss, num_passes = 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        window_passes = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc='Epoch {} train'.format(epoch))):
            # The epoch's last window can be short; average over the passes it actually has.
            window_start = batch_idx - batch_idx % accum_steps
            window_size = min(accum_steps, len(train_loader) - window_start)

            # A batch is skipped when every frame in it failed to featurise, or when the loss
            # comes back non-finite. The window still closes on schedule below, so a skip never
            # strands gradients in the next window.
            if batch is not None:
                data = to_device(batch, device)
                _, loss = flow(data)
                if torch.isfinite(loss):
                    (loss / window_size).backward()
                    total_loss += loss.item()
                    window_loss += loss.item() / window_size
                    num_passes += 1
                    window_passes += 1
                else:
                    print('>>> non-finite loss in epoch {} batch {}, skipping the pass'.format(
                        epoch, batch_idx), file=sys.stderr)

            if batch_idx - window_start + 1 == window_size:
                if window_passes:
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    if ema is not None:
                        ema.update()
                    step += 1
                    if run:
                        run.log({'train_loss': window_loss, 'epoch': epoch, 'step': step})
                optimizer.zero_grad(set_to_none=True)
                window_loss, window_passes = 0.0, 0

        if num_passes == 0:
            raise RuntimeError('epoch {} had no usable training batch'.format(epoch))
        print('>>> epoch {}: train loss {:.4f}'.format(epoch, total_loss / num_passes))

        metrics, _, _ = run_validation(eval_model, val_loader, epoch, run,
                                       num_params=num_params, split_name='val',
                                       reduction=args.frame_reduction, device=device)
        print('Val Perplexity: {:.4f} \\pm {:.4f}'.format(
            metrics['val_perp_mean'], metrics['val_perp_std']))
        print('Top-1 Recovery: {:.4f} \\pm {:.4f}'.format(
            metrics['val_top1_acc_mean'], metrics['val_top1_acc_std']))

        # Rewritten every epoch, so an evicted job still brings home its latest weights.
        save_checkpoint(save_path, config, flow, ema, optimizer, step, epoch)
        if stopper.update(eval_model, metrics['val_perp_mean'], epoch):
            save_checkpoint(best_path, config, flow, ema, optimizer, step, epoch)
            if run:
                run.log({'best_val_perp_mean': stopper.best_perplexity,
                         'best_val_epoch': stopper.best_epoch, 'epoch': epoch})
        if stopper.should_stop(epoch):
            if run:
                run.log({'early_stopped_epoch': epoch, 'epoch': epoch})
            stopped_epoch = epoch
            break
        if deadline is not None:
            now = time.monotonic()
            if now + (now - epoch_start) > deadline:
                print('>>> stopping at epoch {}: another {:.1f}-min epoch would pass --max-hours '
                      '{} ({:.2f} h elapsed)'.format(epoch, (now - epoch_start) / 60,
                                                     args.max_hours, (now - start_time) / 3600))
                if run:
                    run.log({'time_stopped_epoch': epoch, 'epoch': epoch})
                stopped_epoch = epoch
                break

    if run:
        run.summary.update({'best_val_perp_mean': stopper.best_perplexity,
                            'best_val_epoch': stopper.best_epoch, 'stopped_epoch': stopped_epoch})

    # ----------------------------------------------------------------------------------
    # Test, once, with the best val epoch's weights
    # ----------------------------------------------------------------------------------
    stopper.restore(eval_model)
    print('>>> scoring the held-out test split ({} proteins)'.format(len(splits['test'])))
    test_metrics, per_window, failures = run_validation(
        eval_model, test_loader, stopper.best_epoch, run, num_params=num_params,
        split_name='test', reduction=args.frame_reduction, device=device)
    print('Test Perplexity: {:.4f} \\pm {:.4f}'.format(
        test_metrics['test_perp_mean'], test_metrics['test_perp_std']))
    print('Test Top-1 Recovery: {:.4f} \\pm {:.4f}'.format(
        test_metrics['test_top1_acc_mean'], test_metrics['test_top1_acc_std']))

    # Both views of the same windows: per window is ProteinMPNN's convention and is what the
    # printed/W&B numbers above are; per protein is what the DynamicMPNN/PiFold/MapDiff rows use.
    metric_keys = stats_utils.METRIC_KEYS
    per_protein = aggregate_by_protein(per_window, metric_keys)
    summary = stats_utils.summarize_per_protein(per_window, metric_keys)
    summary['n_windows'] = len(per_window)
    protein_summary = stats_utils.summarize_per_protein(per_protein, metric_keys)
    protein_summary['n_proteins'] = len(per_protein)
    print(json.dumps({'per_window': summary, 'per_protein': protein_summary}, indent=2))

    os.makedirs(args.res_dir, exist_ok=True)
    with open(osp.join(args.res_dir, 'summary.json'), 'w') as handle:
        json.dump({'args': vars(args), 'summary': summary, 'protein_summary': protein_summary,
                   'split_counts': {name: len(ids) for name, ids in splits.items()},
                   'window_counts': {name: len(ds) for name, ds in
                                     (('val', val_dataset), ('test', test_dataset))},
                   'best_val_epoch': stopper.best_epoch,
                   'best_val_perp_mean': stopper.best_perplexity,
                   'dropped': dropped, 'failed': failures}, handle, indent=2)
    with open(osp.join(args.res_dir, 'per_window.json'), 'w') as handle:
        json.dump(per_window, handle, indent=2)
    with open(osp.join(args.res_dir, 'per_protein.json'), 'w') as handle:
        json.dump(per_protein, handle, indent=2)

    if run:
        run.summary.update({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})
        run.summary.update({'test_protein_{}'.format(k): v for k, v in protein_summary.items()})
        stats_utils.log_per_protein_table(per_window, metric_keys)
        run.finish()
    store.close()


if __name__ == '__main__':
    main()
