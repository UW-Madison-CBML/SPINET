"""PiFold datasets over *every* MD frame of each protein's trajectory.

`relaxed_dataset.RelaxedStructures` gives PiFold one structure per protein (the deposited entry,
or one frame with ``structure_source='frame'``). These are the all-frames counterparts: train on
every frame -- each frame is just another structure with the protein's sequence -- and evaluate
per protein by running PiFold on all its frames and averaging the logits across frames.

* `TrajectoryFrames` -- one (protein, frame) entry per item, for **training**. Items are shaped
  exactly like `RelaxedStructures`' (``title``/``seq``/``N``/``CA``/``C``/``O``/``score``), so
  the stock `featurize_GTrans` collates them. Laid out protein by protein, so `.sampler()` can
  draw a fixed number of frames per protein per epoch (lib/traj_frames.FramesPerProteinSampler).
* `TrajectoryEnsembles` -- all of one protein's frames per item, for **evaluation**. Collate
  with `EnsembleFeaturizer` (one protein at a time): it featurizes the frames in chunks, and the
  model's per-chunk ``log_probs`` concatenate frame-major into what
  `average_frame_logits` reduces to one ``(R, 20)`` prediction.

Coordinates are read lazily from the trajectory store (`lib/traj_frames.TrajectoryStore`, one h5
handle per DataLoader worker) -- holding every frame of every protein in memory is tens of GB.
Splits come from lib/dataset_splits.py; frames are exactly the ones the sheaf model trains on
(`lib/traj_frames.sheaf_frames`: scripts/train_residue_classifier.py's 128-frame windows every
32 frames, first 1000 frames for ATLAS), optionally thinned by ``frame_stride``/``max_frames`` --
the same reader MapDiff's data/graphs_from_traj.py uses, so all models see the same frames. Proteins that cannot be used are recorded in ``dropped``.

`load_traj_data` builds the three loaders (main.py: `--structure_source traj`). Evaluation sketch::

    _, _, test_loader = load_traj_data('MDCATH', batch_size=8, data_root='./')
    ensembles, featurize = test_loader.ensembles, test_loader.ensemble_featurizer
    for item in (featurize([ensembles[i]]) for i in range(len(ensembles))):
        chunk_log_probs = []
        for X, S, score, mask, lengths in item['chunks']:
            X, S, score, mask = cuda((X, S, score, mask), device=device)
            X, S, score, h_V, h_E, E_idx, batch_id, *_ = model._get_features(S, score, X=X, mask=mask)
            chunk_log_probs.append(model(h_V, h_E, E_idx, batch_id).cpu())
        log_probs = average_frame_logits(torch.cat(chunk_log_probs), item['num_frames'])  # (R, 20)
        target = item['target']                                                              # (R,)

PiFold emits log-probabilities, so this averages log-probs (a renormalisation away from
averaging logits; re-apply ``log_softmax`` before an NLL).
"""
import os
import sys

import torch
import torch.utils.data as data

from .dataloader_gtrans import DataLoader_GTrans
from .featurizer import featurize_GTrans

try:
    import dataset_splits
    import traj_frames
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import dataset_splits
    import traj_frames

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
ATOM_INDEX = {atom: i for i, atom in enumerate(traj_frames.BACKBONE_ATOMS)}
SPLITS = ('train', 'valid', 'test')

average_frame_logits = traj_frames.average_frame_logits


def _entry(protein_id, seq, coords, frame_key, category):
    """One frame as a `featurize_GTrans` entry (same keys as `RelaxedStructures`)."""
    return {
        'title': protein_id,
        'seq': seq,
        'N': coords[:, ATOM_INDEX['N']],
        'CA': coords[:, ATOM_INDEX['CA']],
        'C': coords[:, ATOM_INDEX['C']],
        'O': coords[:, ATOM_INDEX['O']],
        'score': 100.0,
        'category': category,
        # (trajectory group, frame index): a protein's frames are pooled across every trajectory
        # group the sheaf model uses for it (e.g. replicas).
        'frame_key': (str(frame_key[0]), int(frame_key[1])),
    }


class _TrajectorySplits:
    """Index of one dataset's trajectories, shared by the per-split datasets so the h5 metadata
    is read once."""

    def __init__(self, path='./', ds_name='atlas', index_csv=None, traj_h5=None,
                 max_length=dataset_splits.MAX_LENGTH, frame_stride=1, max_frames=None,
                 val_fold=dataset_splits.DEFAULT_VAL_FOLD, test_fold=dataset_splits.DEFAULT_TEST_FOLD,
                 splits=SPLITS):
        self.ds_name = ds_name.lower()
        index_csv = index_csv or os.path.join(path, dataset_splits.default_index_csv(self.ds_name))
        traj_h5 = traj_h5 or os.path.join(path, dataset_splits.default_h5(self.ds_name))
        for f in (index_csv, traj_h5):
            if not os.path.exists(f):
                raise FileNotFoundError('no such file: {} !!!'.format(f))
        self.store = traj_frames.TrajectoryStore(traj_h5)

        train_ids, val_ids, test_ids = dataset_splits.get_splits(
            self.ds_name, index_csv, val_fold=val_fold, test_fold=test_fold)
        split_ids = dict(zip(SPLITS, (train_ids, val_ids, test_ids)))
        wanted = [p for s in splits for p in split_ids[s]]

        entries, failures = self.store.index(wanted, self.ds_name, stride=frame_stride, max_frames=max_frames,
                                             max_length=max_length)
        # Opened in the parent only to index; workers reopen their own handle.
        self.store.close()
        alphabet = set(ALPHABET)
        self.dropped = {'unusable_trajectory': {}, 'nonstandard_residue': []}
        for protein_id, reason in failures.items():
            self.dropped['unusable_trajectory'][protein_id] = reason
        self.split_entries = {}
        for split in SPLITS:
            kept = {}
            for protein_id in (split_ids[split] if split in splits else []):
                entry = entries.get(protein_id)
                if entry is None:
                    continue
                if set(entry['seq']) - alphabet:
                    self.dropped['nonstandard_residue'].append(protein_id)
                    continue
                kept[protein_id] = entry
            self.split_entries[split] = kept

        print('{}: {} train / {} valid / {} test trajectories ({} / {} / {} frames)'.format(
            self.ds_name, *(len(self.split_entries[s]) for s in SPLITS),
            *(sum(len(e['frames']) for e in self.split_entries[s].values()) for s in SPLITS)))
        if failures:
            print('{}: dropped {} protein(s) with unusable trajectories: {}'.format(
                self.ds_name, len(failures), failures))


class TrajectoryFrames(data.Dataset):
    """One MD frame per item; see the module docstring. Build with `from_splits` or
    `load_traj_data` rather than directly.

    ``max_frames`` thins each protein's frames further (evenly spaced, `select_frames`) for this
    view only -- ``max_frames=1`` gives one frame per protein (the first), which is what the
    per-frame valid/test loaders use for upstream's frame-level loss.
    """

    def __init__(self, index, mode, max_frames=None):
        self.index = index
        self.store = index.store
        self.mode = mode
        self.entries = index.split_entries[mode]
        self.protein_ids = list(self.entries)
        self.frames_of = {}
        for p in self.protein_ids:
            frames = self.entries[p]['frames']
            keep = traj_frames.select_frames(len(frames), max_frames=max_frames)
            self.frames_of[p] = [frames[i] for i in keep]
        self.frame_counts = [len(self.frames_of[p]) for p in self.protein_ids]
        self.items = [(p, f) for p in self.protein_ids for f in self.frames_of[p]]
        self.dropped = index.dropped
        self.category = index.ds_name.upper()

    @classmethod
    def from_splits(cls, mode='train', **kwargs):
        return cls(_TrajectorySplits(**kwargs), mode)

    def split_ids(self, mode=None):
        """Protein ids in this split, in split order (main.py's manifest hook)."""
        return list(self.protein_ids)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        protein_id, frame_key = self.items[index]
        entry = self.entries[protein_id]
        coords = self.store.read_frames(entry, [frame_key])[0]
        return _entry(protein_id, entry['seq'], coords, frame_key, self.category)

    def sampler(self, frames_per_protein=None, shuffle=True, seed=dataset_splits.SEED):
        """``frames_per_protein`` random frames of each protein per epoch (``None`` = all)."""
        return traj_frames.FramesPerProteinSampler(self.frame_counts, frames_per_protein, shuffle, seed)


class TrajectoryEnsembles(TrajectoryFrames):
    """All of one protein's frames per item: ``(protein_id, [entry, ...])``. Same proteins and
    frames as a `TrajectoryFrames` over the same split. Reads the whole frame set in one h5
    request."""

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, index):
        protein_id = self.protein_ids[index]
        entry = self.entries[protein_id]
        frames = self.frames_of[protein_id]
        coords = self.store.read_frames(entry, frames)
        return protein_id, [_entry(protein_id, entry['seq'], c, f, self.category)
                            for f, c in zip(frames, coords)]


class EnsembleFeaturizer:
    """Collate one `TrajectoryEnsembles` item (``batch_size=1``).

    Returns ``{'name', 'frame_keys', 'num_frames', 'entries', 'chunks', 'target'}``:
    ``chunks`` is a list of `featurize_GTrans` outputs ``(X, S, score, mask, lengths)``, each
    over at most ``chunk_size`` consecutive frames (``None`` = one chunk), and ``target`` the
    ``(R,)`` native labels in PiFold's alphabet. A trajectory frame has no missing backbone
    atoms, so featurize_GTrans keeps every residue and the concatenated per-chunk log-probs are
    frame-major ``(num_frames * R, 20)``.
    """

    def __init__(self, chunk_size=None):
        self.chunk_size = chunk_size

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError('EnsembleFeaturizer collates one protein at a time -- use batch_size=1')
        protein_id, entries = batch[0]
        size = self.chunk_size or len(entries)
        return {
            'name': protein_id,
            'frame_keys': [e['frame_key'] for e in entries],
            'num_frames': len(entries),
            'entries': entries,
            'chunks': [featurize_GTrans(entries[i:i + size]) for i in range(0, len(entries), size)],
            'target': torch.tensor([ALPHABET.index(a) for a in entries[0]['seq']], dtype=torch.long),
        }


def load_traj_data(data_name, batch_size, data_root='./', num_workers=8, frames_per_protein=None,
                   eval_chunk_size=None, val_max_frames=None, seed=dataset_splits.SEED, **kwargs):
    """``(train_loader, valid_loader, test_loader)`` for all-frames training/evaluation.

    All three yield `featurize_GTrans` batches of single frames, so upstream's
    ``train_one_epoch`` / ``valid_one_epoch`` / ``test_one_epoch`` run unchanged:

    * train draws ``frames_per_protein`` frames of each protein per epoch (``None`` = all);
    * valid/test hold one frame (the first) per protein -- upstream's frame-level loss only.

    The frame-*averaged* evaluation lives on the valid/test loaders as ``.ensembles`` (a
    `TrajectoryEnsembles` over every selected frame) and ``.ensemble_featurizer`` (an
    `EnsembleFeaturizer`); main.py's `residue_metrics` uses those when present.
    ``val_max_frames`` thins the *validation* ensembles to that many evenly spaced frames per
    protein (``None`` = every frame) -- it only drives early stopping, and test always averages
    every selected frame so reported numbers stay comparable across models. Remaining
    kwargs go to `_TrajectorySplits` (``index_csv``, ``traj_h5``, ``max_length``,
    ``frame_stride``, ``max_frames``, ``val_fold``, ``test_fold``, ``splits``).
    """
    ds_names = {'ATLAS': 'atlas', 'MDCATH': 'mdcath'}
    if data_name not in ds_names:
        raise ValueError('unknown dataset {}'.format(data_name))
    index = _TrajectorySplits(path=data_root, ds_name=ds_names[data_name], **kwargs)

    train_set = TrajectoryFrames(index, 'train')
    train_loader = DataLoader_GTrans(
        train_set, batch_size=batch_size, num_workers=num_workers, collate_fn=featurize_GTrans,
        sampler=train_set.sampler(frames_per_protein, shuffle=True, seed=seed))

    eval_loaders = []
    for mode in ('valid', 'test'):
        loader = DataLoader_GTrans(TrajectoryFrames(index, mode, max_frames=1), batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers, collate_fn=featurize_GTrans)
        loader.ensembles = TrajectoryEnsembles(
            index, mode, max_frames=val_max_frames if mode == 'valid' else None)
        loader.ensemble_featurizer = EnsembleFeaturizer(eval_chunk_size)
        eval_loaders.append(loader)
    return (train_loader, *eval_loaders)
