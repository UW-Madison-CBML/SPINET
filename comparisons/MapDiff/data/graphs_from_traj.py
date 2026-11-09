"""Featurize *every* frame of each protein's MD trajectory into MapDiff graphs.

`data/generate_graph_relaxed.py` gives MapDiff one structure per protein (the deposited entry,
or one MD frame with `--structure-source frame`). This script is the all-frames counterpart:
each (selected) frame of each trajectory becomes its own MapDiff graph, so the static model
can be trained on every frame -- a frame is just another structure with the same sequence --
and evaluated per protein by running it on all of a protein's frames and averaging the
logits. The matching datasets are `dataloader/traj_dataset.py`'s `TrajFrames` (one graph per
item, for training) and `TrajEnsembles` (all of a protein's frames per item, for evaluation).

Output layout, one directory per protein so a protein's frames can be found without an index:

    <save-root>/<split>/<protein_id>/<group>__<frame:05d>.pt     split in train/validation/test

``<group>`` names the trajectory the frame came from (`lib/traj_frames.frame_key_filename`):
like the sheaf model, every usable trajectory group under a protein (e.g. each replica) is used,
and its frames are pooled into that protein's set.

Frames are exactly the ones the sheaf model trains on (scripts/train_residue_classifier.py:
128-frame windows every 32 frames, first 1000 frames for ATLAS -- `lib/traj_frames.sheaf_frames`),
with the same protein filter (trajectories shorter than one window, or missing the sheaf
model's datasets, are skipped). `--frame-stride` / `--max-frames` thin that set; leave them unset
to match the sheaf model. The PiFold loader uses the same reader, so all models see the same
frames. Featurization is
per-frame-resumable: a frame whose .pt exists is skipped, so a preempted job picks up where it
stopped. A frame DSSP/featurization fails on is skipped and reported; the rest of the protein is
kept.

Splits, units and the backbone-only caveat are as for `generate_graph_relaxed.py
--structure-source frame` (see lib/traj_frames.py): coordinates are converted to Angstroms per
store, DSSP SASA is computed on a backbone-only chain. The amino-acid marginal is computed from
one frame per training protein -- every frame carries the same sequence, so this is the same
distribution the relaxed pipeline computes, not one weighted by trajectory length.

DSSP runs once per frame, which dominates the cost (~200 frames/protein on ATLAS); use
`--workers` to run proteins in parallel and `--frame-stride`/`--max-frames` to thin. Saved graphs
take ~4 KB per residue per frame, so every frame of mdCATH runs to terabytes: train.py's default
(`dataset.online_featurize=true`) skips this script and featurizes frames on the fly instead.

Usage:
    python data/graphs_from_traj.py --ds-name mdcath \
        --index-csv ./mdcath_320_0_topology_split.csv --traj-h5 ./mdcath_spinet_320_0.h5 \
        --save-root ./surffold_data/mdcath320_traj_process/ \
        --marginal-out ./surffold_data/train_marginal_x_mdcath320_traj.pt --workers 8
"""
import argparse
import os
import sys
from multiprocessing import Pool

import torch
from tqdm import tqdm

# data/generate_graph_cath.py imports `dataloader.*`/`utils` assuming the repo root is on
# sys.path; make that true regardless of the caller's CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.traj_dataset import featurize_record
from dataloader.pyg_safe_globals import allow_pyg_data_pickles

# lib/ modules are copied in flat next to this file (see ../train.sh) -- fall back to walking
# up to a lib/ directory for local/dev runs from inside the source tree.
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


allow_pyg_data_pickles()

SPLIT_DIRS = ('train', 'validation', 'test')


def frame_filename(protein_id, frame_key):
    return traj_frames.frame_key_filename(protein_id, frame_key)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ds-name', default='atlas', choices=dataset_splits.DATASETS)
    parser.add_argument('--index-csv', default=None,
                        help="Split index csv (default: the dataset's standard file name)")
    parser.add_argument('--traj-h5', default=None,
                        help="Trajectory store (default: the dataset's standard file name)")
    parser.add_argument('--frame-stride', type=int, default=1,
                        help="Use every N-th frame of each trajectory")
    parser.add_argument('--max-frames', type=int, default=None,
                        help="After striding, thin to at most this many evenly spaced frames per protein")
    parser.add_argument('--max-length', type=int, default=dataset_splits.MAX_LENGTH,
                        help="Skip proteins longer than this many residues (no point running DSSP on "
                             "frames the dataloader would drop)")
    parser.add_argument('--val-fold', type=int, default=dataset_splits.DEFAULT_VAL_FOLD,
                        help="ATLAS only: cross_val fold used as the validation split")
    parser.add_argument('--test-fold', type=int, default=dataset_splits.DEFAULT_TEST_FOLD,
                        help="ATLAS only: cross_val fold held out as the test split")
    parser.add_argument('--splits', nargs='+', default=list(SPLIT_DIRS), choices=SPLIT_DIRS,
                        help="Which splits to featurize (e.g. just `test` for an eval-only job)")
    parser.add_argument('--workers', type=int, default=1,
                        help="Proteins featurized in parallel (each runs its own DSSP calls)")
    parser.add_argument('--save-root', default=None,
                        help="Default: ./surffold_data/<dataset tag>_traj_process/")
    parser.add_argument('--marginal-out', default=None,
                        help="Default: ./surffold_data/train_marginal_x_<dataset tag>_traj.pt")
    return parser.parse_args()


# Per-worker trajectory reader, set by `_init_worker` (one h5 handle per process).
_STORE = None


def _init_worker(h5_path):
    global _STORE
    _STORE = traj_frames.TrajectoryStore(h5_path)


def featurize_protein(job):
    """Featurize every pending frame of one protein. Returns ``(protein_id, written, errors)``."""
    protein_id, entry, save_dir = job
    os.makedirs(save_dir, exist_ok=True)
    pending = [k for k in entry['frames']
               if not os.path.exists(os.path.join(save_dir, frame_filename(protein_id, k)))]
    if not pending:
        return protein_id, 0, []

    written, errors = 0, []
    try:
        records = _STORE.frame_records(protein_id, entry, pending)
        for record in records:
            frame_key = (record['group'], record['frame_index'])
            try:
                graph = featurize_record(record)
            except Exception as exc:  # noqa: BLE001 -- one bad frame skips just that frame
                errors.append((frame_key, str(exc)))
                continue
            if graph is None:
                errors.append((frame_key, 'pdb2graph returned None'))
                continue
            graph.frame_index = int(frame_key[1])
            graph.traj_group = frame_key[0]
            # Write-then-rename so a preempted job never leaves a truncated .pt that the
            # resume check would mistake for a finished frame.
            out_path = os.path.join(save_dir, frame_filename(protein_id, frame_key))
            torch.save(graph, out_path + '.tmp')
            os.replace(out_path + '.tmp', out_path)
            written += 1
    except Exception as exc:  # noqa: BLE001 -- e.g. unreadable coordinates
        errors.append((None, str(exc)))
    return protein_id, written, errors


def process_split(entries, split_name, split_dir, workers, h5_path):
    jobs = [(p, entry, os.path.join(split_dir, p)) for p, entry in entries.items()]
    total_frames = sum(len(e['frames']) for e in entries.values())
    print(f'{split_name}: {len(jobs)} proteins, {total_frames} frames')

    failed = {}
    if workers > 1:
        pool = Pool(workers, initializer=_init_worker, initargs=(h5_path,))
        results = pool.imap_unordered(featurize_protein, jobs)
    else:
        pool = None
        _init_worker(h5_path)
        results = map(featurize_protein, jobs)
    try:
        for protein_id, _, errors in tqdm(results, total=len(jobs), desc=f'featurizing {split_name}'):
            if errors:
                failed[protein_id] = errors
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return failed


def compute_marginal(split_dir, out_path, train_ids):
    """Amino-acid marginal over the training proteins, one frame each (see module docstring)."""
    counts = torch.zeros(20)
    used = 0
    for protein_id in tqdm(train_ids, desc='computing amino-acid marginal'):
        protein_dir = os.path.join(split_dir, protein_id)
        frames = sorted(f for f in os.listdir(protein_dir) if f.endswith('.pt')) if os.path.isdir(protein_dir) else []
        if not frames:
            continue
        graph = torch.load(os.path.join(protein_dir, frames[0]))
        counts += graph.x[:, :20].sum(dim=0)
        used += 1
    marginal = counts / counts.sum()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    torch.save(marginal, out_path)
    print(f'Saved amino-acid marginal over {used} training proteins ({int(counts.sum())} residues) to {out_path}')


def main():
    args = create_parser()
    traj_h5 = args.traj_h5 or dataset_splits.default_h5(args.ds_name)
    tag = dataset_splits.dataset_tag(args.ds_name, traj_h5)
    save_root = args.save_root or f'./surffold_data/{tag}_traj_process/'
    marginal_out = args.marginal_out or f'./surffold_data/train_marginal_x_{tag}_traj.pt'

    train_ids, val_ids, test_ids = dataset_splits.get_splits(
        args.ds_name, args.index_csv, val_fold=args.val_fold, test_fold=args.test_fold)
    split_ids = dict(zip(SPLIT_DIRS, (train_ids, val_ids, test_ids)))
    print(f'{args.ds_name}: {len(train_ids)} train / {len(val_ids)} validation / {len(test_ids)} test proteins')

    store = traj_frames.TrajectoryStore(traj_h5)
    wanted = [p for split in args.splits for p in split_ids[split]]
    entries, failures = store.index(wanted, args.ds_name, stride=args.frame_stride, max_frames=args.max_frames,
                                    max_length=args.max_length)
    store.close()
    for protein_id, reason in failures.items():
        print(f'skip {protein_id}: {reason}')

    for split in args.splits:
        split_entries = {p: entries[p] for p in split_ids[split] if p in entries}
        failed = process_split(split_entries, split, os.path.join(save_root, split), args.workers, traj_h5)
        n_bad = sum(len(v) for v in failed.values())
        print(f'{split}: {n_bad} frames across {len(failed)} proteins failed to featurize')
        for protein_id, errors in failed.items():
            print(f'  {protein_id}: {errors[:3]}{" ..." if len(errors) > 3 else ""}')

    if 'train' in args.splits:
        compute_marginal(os.path.join(save_root, 'train'), marginal_out, train_ids)


if __name__ == '__main__':
    main()
