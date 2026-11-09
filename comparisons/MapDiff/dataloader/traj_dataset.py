"""MapDiff datasets over *every* MD frame, as featurized by data/graphs_from_traj.py.

Two views of the same ``<split_dir>/<protein_id>/<group>__<frame:05d>.pt`` graphs (a protein's
frames are pooled across every trajectory group the sheaf model uses for it, e.g. replicas):

* `TrajFrames` -- one (protein, frame) graph per item, for **training**: each frame is just
  another structure with the protein's sequence. Items are laid out protein by protein, so
  `TrajFrames.sampler` can draw a fixed number of frames per protein per epoch (see
  `lib/traj_frames.FramesPerProteinSampler`) instead of making one epoch = every frame.
  Batches with the stock `CollatorDiff` / `CollatorIPAPretrain`, so it drops into train.py's
  loaders unchanged.
* `TrajEnsembles` -- one protein per item, all its frames, for **evaluation**: run the model on
  every frame and average the logits across frames (`lib/traj_frames.average_frame_logits`).
  Batch with `EnsembleCollator` at ``batch_size=1``; it hands back the frames pre-collated in
  chunks so a 200-frame protein does not have to fit on the GPU at once.

Like train.py's `split_file_ids`, the protein list comes from the split index
(lib/dataset_splits.py), not from whatever directories exist, and proteins with no featurized
frame are reported rather than silently skipped.

Training::

    train_set = TrajFrames(f'{root}/train/', train_ids, max_length=cfg.dataset.max_length)
    train_loader = DataLoader(train_set, batch_size=B, num_workers=16, collate_fn=CollatorDiff(),
                              sampler=train_set.sampler(frames_per_protein=4, seed=dataset_splits.SEED))

`OnlineTrajFrames` / `OnlineTrajEnsembles` are the same two views without the .pt files: each
frame is featurized (PDB -> DSSP -> `pdb2graph`) inside the DataLoader workers when it is drawn
and then discarded, so every frame of every trajectory is available without the disk cost
(~4 KB/residue/frame -- terabytes for mdCATH). See their docstrings.

Evaluation (logits averaged across frames, then scored like eval_relaxed.py)::

    test_set = TrajEnsembles(f'{root}/test/', test_ids, max_length=cfg.dataset.max_length)
    loader = DataLoader(test_set, batch_size=1, collate_fn=EnsembleCollator(chunk_size=16))
    for item in loader:
        frame_logits = []
        for g_batch, ipa_batch in item['chunks']:
            logits, _ = model.mc_ddim_sample(g_batch.to(dev), ipa_batch.to(dev), diverse=True, step=ddim_steps)
            frame_logits.append(logits.cpu())
        mean_logits = average_frame_logits(torch.cat(frame_logits), item['num_frames'])  # (R, 20)
        target = item['target']                                                           # (R,)
"""
import bisect
import os
import sys
import tempfile
import warnings

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataloader.collator import CollatorDiff
from dataloader.large_dataset import to_model_graph

try:
    import traj_frames
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import traj_frames

import relaxed_pdb  # lib/, on sys.path by now (flat copy in the job, or found above)

average_frame_logits = traj_frames.average_frame_logits


def list_frame_files(protein_id, protein_dir, max_frames=None):
    """Sorted ``(frame_key, path)`` pairs for one protein's featurized frames -- ``frame_key``
    being ``(trajectory group, frame index)`` -- optionally thinned to ``max_frames`` evenly
    spaced ones across all of its trajectories."""
    if not os.path.isdir(protein_dir):
        return []
    keyed = sorted((traj_frames.parse_frame_filename(protein_id, f), f)
                   for f in os.listdir(protein_dir) if f.endswith('.pt'))
    if max_frames is not None and 0 < max_frames < len(keyed):
        keyed = [keyed[i] for i in traj_frames.select_frames(len(keyed), max_frames=max_frames)]
    return [(key, os.path.join(protein_dir, f)) for key, f in keyed]


class TrajFrames(Dataset):
    """One featurized MD frame per item.

    ``protein_ids`` is the split (from lib/dataset_splits.py); ``max_length`` drops proteins
    longer than that many residues, same as `large_dataset.Cath`; ``max_frames`` caps the
    frames used per protein (evenly spaced) without re-featurizing.

    Attributes: ``protein_ids`` (kept proteins, split order), ``frame_counts`` (frames per kept
    protein, aligned), ``items`` (flat ``(protein_id, frame_key, path)``), ``list_IDs``
    (``<protein_id>.pt`` per kept protein, the attribute train.py logs as the split manifest),
    ``missing`` / ``too_long`` (dropped protein ids).
    """

    def __init__(self, split_dir, protein_ids, max_length=None, max_frames=None, verbose=True):
        self.split_dir = split_dir
        self.protein_ids, self.frame_counts, self.items = [], [], []
        self.missing, self.too_long = [], []

        iterator = tqdm(protein_ids, desc=f'indexing frames in {split_dir}') if verbose else protein_ids
        for protein_id in iterator:
            frames = list_frame_files(protein_id, os.path.join(split_dir, protein_id), max_frames)
            if not frames:
                self.missing.append(protein_id)
                continue
            if max_length is not None and torch.load(frames[0][1]).x.shape[0] > max_length:
                self.too_long.append(protein_id)
                continue
            self.protein_ids.append(protein_id)
            self.frame_counts.append(len(frames))
            self.items.extend((protein_id, frame_key, path) for frame_key, path in frames)

        self.list_IDs = [f'{p}.pt' for p in self.protein_ids]
        if verbose:
            print(f'{split_dir}: {len(self.protein_ids)} proteins, {len(self.items)} frames')
            if self.missing:
                print(f'{split_dir}: {len(self.missing)}/{len(protein_ids)} proteins have no featurized '
                      f'frame and are skipped: {self.missing}')
            if self.too_long:
                print(f'{split_dir}: dropped {len(self.too_long)} proteins over max_length={max_length}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        protein_id, (group_name, frame_index), path = self.items[index]
        graph = to_model_graph(torch.load(path), protein_id)
        graph.frame_index = frame_index
        graph.traj_group = group_name
        return graph

    def sampler(self, frames_per_protein=None, shuffle=True, seed=0):
        """A `FramesPerProteinSampler` over this dataset: ``frames_per_protein`` random frames
        of each protein per epoch (``None`` = every frame). Pass as ``DataLoader(sampler=...)``."""
        return traj_frames.FramesPerProteinSampler(self.frame_counts, frames_per_protein, shuffle, seed)


class TrajEnsembles(Dataset):
    """All of one protein's featurized frames per item: ``(protein_id, [graph, ...])``.

    Same arguments and drop rules as `TrajFrames` (it is built on one), so a `TrajEnsembles`
    and a `TrajFrames` over the same split agree on the proteins and frames.
    """

    def __init__(self, split_dir, protein_ids, max_length=None, max_frames=None, verbose=True):
        self.frames = TrajFrames(split_dir, protein_ids, max_length, max_frames, verbose)
        self.protein_ids = self.frames.protein_ids
        self.frame_counts = self.frames.frame_counts
        self.list_IDs = self.frames.list_IDs
        offsets, total = [], 0
        for n in self.frame_counts:
            offsets.append(total)
            total += n
        self.offsets = offsets

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, index):
        start = self.offsets[index]
        graphs = [self.frames[i] for i in range(start, start + self.frame_counts[index])]
        return self.protein_ids[index], graphs


class EnsembleCollator:
    """Collate one `TrajEnsembles` item (use ``batch_size=1``) into model-ready chunks.

    Returns a dict:

    * ``name`` -- protein id
    * ``frame_keys`` -- ``(trajectory group, frame index)`` of each graph, in order
    * ``num_frames`` -- how many frames
    * ``chunks`` -- list of ``(g_batch, ipa_batch)`` from `CollatorDiff`, each over at most
      ``chunk_size`` consecutive frames (``None`` = one chunk). Concatenating the per-chunk
      logits gives frame-major ``(num_frames * R, 20)`` rows, which is what
      `average_frame_logits` expects.
    * ``target`` -- ``(R,)`` native residue labels (identical for every frame)
    """

    def __init__(self, chunk_size=None):
        self.chunk_size = chunk_size
        self.collate = CollatorDiff()

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError('EnsembleCollator collates one protein at a time -- use batch_size=1')
        protein_id, graphs = batch[0]
        lengths = {g.x.shape[0] for g in graphs}
        if len(lengths) != 1:
            raise ValueError(f'{protein_id}: frames disagree on residue count {sorted(lengths)}')
        size = self.chunk_size or len(graphs)
        return {
            'name': protein_id,
            'frame_keys': [(g.traj_group, int(g.frame_index)) for g in graphs],
            'num_frames': len(graphs),
            'chunks': [self.collate(graphs[i:i + size]) for i in range(0, len(graphs), size)],
            'target': graphs[0].x.argmax(dim=-1),
        }


# ---------------------------------------------------------------------------
# On-the-fly featurization: no graphs on disk
# ---------------------------------------------------------------------------

def featurize_record(record):
    """One `traj_frames.frame_record` -> raw `pdb2graph` graph (``None`` if it gives none).

    Writes the frame to a temporary PDB, runs MapDiff's featurizer (DSSP included) on it and
    deletes the PDB -- the same steps data/graphs_from_traj.py saves to disk."""
    from data.generate_graph_cath import pdb2graph  # imported lazily: pulls in the CATH dataset code

    fd, tmp_path = tempfile.mkstemp(suffix='.pdb')
    os.close(fd)
    try:
        relaxed_pdb.write_record_pdb(record, tmp_path)
        return pdb2graph(tmp_path)
    finally:
        os.remove(tmp_path)


class OnlineTrajFrames(Dataset):
    """`TrajFrames` without the .pt files: item ``i`` is featurized from the trajectory store
    when it is requested (in a DataLoader worker) and not kept.

    Proteins and frames come from `traj_frames.TrajectoryStore.index` -- the sheaf model's frames
    pooled across every trajectory group, thinned by ``stride`` / ``max_frames`` -- the same
    set data/graphs_from_traj.py would write, so the two modes train on identical frames.

    A frame DSSP/featurization fails on: with ``fallback=True`` (training) the next frames of
    the same protein are tried instead, so a batch never comes back short; with
    ``fallback=False`` (evaluation) the item is ``None`` and the ensemble just leaves that frame
    out, as the on-disk pipeline skips it.

    Attributes match `TrajFrames`: ``protein_ids``, ``frame_counts``, ``list_IDs``, ``missing``
    (protein id -> reason it was dropped), plus ``offsets`` (first item of each protein).
    """

    MAX_FALLBACK = 8

    def __init__(self, h5_path, protein_ids, ds_name, max_length=None, max_frames=None, stride=1,
                 fallback=True, verbose=True):
        self.store = traj_frames.TrajectoryStore(h5_path)
        self.entries, self.missing = self.store.index(protein_ids, ds_name, stride=stride, max_frames=max_frames,
                                                      max_length=max_length, verbose=verbose)
        self.store.close()  # reopened lazily in each worker
        self.protein_ids = [p for p in dict.fromkeys(protein_ids) if p in self.entries]
        self.frame_counts = [len(self.entries[p]['frames']) for p in self.protein_ids]
        self.offsets, total = [], 0
        for n in self.frame_counts:
            self.offsets.append(total)
            total += n
        self._len = total
        self.list_IDs = [f'{p}.pt' for p in self.protein_ids]
        self.fallback = fallback
        if verbose:
            print(f'{h5_path}: {len(self.protein_ids)} proteins, {total} frames (featurized on the fly)')
            if self.missing:
                print(f'{len(self.missing)}/{len(protein_ids)} proteins dropped: {self.missing}')

    def __len__(self):
        return self._len

    def locate(self, index):
        """``(protein position, frame position within that protein)`` of flat item ``index``."""
        if not 0 <= index < self._len:
            raise IndexError(index)
        position = bisect.bisect_right(self.offsets, index) - 1
        return position, index - self.offsets[position]

    def featurize(self, protein_id, frame_key):
        entry = self.entries[protein_id]
        record = next(self.store.frame_records(protein_id, entry, [frame_key]))
        raw = featurize_record(record)
        if raw is None:
            raise ValueError('pdb2graph returned None')
        graph = to_model_graph(raw, protein_id)
        graph.frame_index = int(frame_key[1])
        graph.traj_group = frame_key[0]
        return graph

    def __getitem__(self, index):
        position, j = self.locate(index)
        protein_id = self.protein_ids[position]
        frames = self.entries[protein_id]['frames']
        attempts = min(len(frames), self.MAX_FALLBACK) if self.fallback else 1
        errors = []
        for t in range(attempts):
            frame_key = frames[(j + t) % len(frames)]
            try:
                return self.featurize(protein_id, frame_key)
            except Exception as exc:  # noqa: BLE001 -- one bad frame skips just that frame
                errors.append((frame_key, str(exc)))
        if not self.fallback:
            warnings.warn(f'{protein_id}: skipping frame {errors[0][0]} ({errors[0][1]})')
            return None
        raise RuntimeError(f'{protein_id}: {attempts} consecutive frames failed to featurize: {errors}')

    def sampler(self, frames_per_protein=None, shuffle=True, seed=0):
        """Same as `TrajFrames.sampler`."""
        return traj_frames.FramesPerProteinSampler(self.frame_counts, frames_per_protein, shuffle, seed)


class OnlineTrajEnsembles:
    """Per-protein frame ensembles for evaluation, featurized on the fly -- an iterable that
    stands in for ``DataLoader(TrajEnsembles(...), batch_size=1, collate_fn=EnsembleCollator(...))``.

    Iterating yields one dict per protein, like `EnsembleCollator`'s, except that ``chunks`` is a
    *generator*: each ``(g_batch, ipa_batch)`` of up to ``chunk_size`` frames is featurized by
    ``num_workers`` DataLoader workers (chunks in parallel, in order) and only one protein's
    chunks are ever held in memory. ``num_frames``, ``frame_keys`` and ``target`` are filled in
    as the chunks are consumed, so read them *after* exhausting ``chunks`` (trainer.py's
    `_eval_batches` and eval_relaxed.py do). Frames that failed to featurize are left out of
    ``num_frames``; a protein with none left has ``num_frames == 0`` and no ``target``.
    """

    def __init__(self, frames, chunk_size=8, num_workers=8):
        if frames.fallback:
            raise ValueError('build the evaluation OnlineTrajFrames with fallback=False')
        self.frames = frames
        self.chunk_size = chunk_size or max(frames.frame_counts, default=1)
        self.num_workers = num_workers
        self.collate = CollatorDiff()
        self.protein_ids = frames.protein_ids
        self.frame_counts = frames.frame_counts
        self.list_IDs = frames.list_IDs
        self.chunks_per_protein = [-(-n // self.chunk_size) for n in self.frame_counts]

    def __len__(self):
        return len(self.protein_ids)

    def _batches(self):
        for offset, n in zip(self.frames.offsets, self.frame_counts):
            for start in range(0, n, self.chunk_size):
                yield list(range(offset + start, offset + min(start + self.chunk_size, n)))

    def _collate(self, graphs):
        graphs = [g for g in graphs if g is not None]
        if not graphs:
            return None
        return {'keys': [(g.traj_group, int(g.frame_index)) for g in graphs], 'n': len(graphs),
                'target': graphs[0].x.argmax(dim=-1), 'batch': self.collate(graphs)}

    def _protein_chunks(self, stream, n_chunks, item):
        for _ in range(n_chunks):
            chunk = next(stream)
            if chunk is None:
                continue
            if 'target' in item and item['target'].shape != chunk['target'].shape:
                raise ValueError(f"{item['name']}: frames disagree on residue count")
            item['target'] = chunk['target']
            item['num_frames'] += chunk['n']
            item['frame_keys'].extend(chunk['keys'])
            yield chunk['batch']

    def __iter__(self):
        loader = DataLoader(self.frames, batch_sampler=list(self._batches()), num_workers=self.num_workers,
                            collate_fn=self._collate)
        stream = iter(loader)
        for protein_id, n_chunks in zip(self.protein_ids, self.chunks_per_protein):
            item = {'name': protein_id, 'num_frames': 0, 'frame_keys': []}
            chunks = self._protein_chunks(stream, n_chunks, item)
            item['chunks'] = chunks
            yield item
            for _ in chunks:  # the consumer stopped early: keep the stream aligned
                pass


def compute_marginal(frames, out_path, num_workers=8):
    """Amino-acid marginal over ``frames``' proteins, one frame each -- what
    data/graphs_from_traj.py's `compute_marginal` writes, without the saved graphs."""
    first_frames = torch.utils.data.Subset(frames, frames.offsets)
    loader = DataLoader(first_frames, batch_size=1, num_workers=num_workers, collate_fn=lambda b: b[0])
    counts, used = torch.zeros(20), 0
    for graph in tqdm(loader, desc='computing amino-acid marginal'):
        if graph is None:
            continue
        counts += graph.x[:, :20].sum(dim=0)
        used += 1
    marginal = counts / counts.sum()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    torch.save(marginal, out_path)
    print(f'Saved amino-acid marginal over {used} proteins ({int(counts.sum())} residues) to {out_path}')
    return marginal
