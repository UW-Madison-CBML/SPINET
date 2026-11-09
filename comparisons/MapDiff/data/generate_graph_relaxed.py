"""Turn structures into MapDiff's per-residue graph format.

MapDiff is a static-structure inverse-folding model, so by default it is trained and
evaluated on the deposited PDB entry for each protein in our datasets rather than on a frame
pulled out of the MD trajectory. `lib/relaxed_pdb.py` downloads the entry from RCSB, picks
the chain the dataset id names, and crops it to the residues the trajectory covers (using the
trajectory's own residue list as the reference sequence), so MapDiff scores the same residues
as every other model in the comparison. DynamicMPNN deliberately still trains on MD
ensembles -- that ensemble input is the thing being benchmarked.

`--structure-source frame` switches the input to one random frame per protein's trajectory
(`lib/traj_frames.py`) instead, with everything downstream -- splits, featurization, the
marginal -- unchanged. That makes "relaxed vs. a single MD frame" a one-flag ablation on an
otherwise identical pipeline. The frame is chosen deterministically from
``(--frame-seed, protein id)``; pass `--frame-index` to pin one index for every protein.
Write it to a *different* `--save-root`, since the .pt files are the only record of which
source a graph came from.

Both shared datasets are supported, trained/evaluated separately (`--ds-name`), matching
scripts/train_residue_classifier.py:

* ``atlas``  -- `atlas_cross_val_index.csv`'s `cross_val` column: fold `--val-fold` is the
  validation split, fold `--test-fold` the held-out test split, the remaining folds train.
* ``mdcath`` -- `mdcath_320_0_topology_split.csv`'s train/validation/test rows, one split
  each.

Each split is featurized into its own directory (`train/`, `validation/`, `test/` under
`--save-root`), which is what ../conf/dataset/*.yaml's `train_dir`/`val_dir`/`test_dir`
point at.

The deposited chain is written back out as a PDB (side chains and residue numbering intact)
so DSSP secondary-structure assignment (`data.generate_graph_cath.pdb2graph`) works
unchanged, then deleted. Also computes the train-split amino-acid marginal distribution that
`model.prior_diff.Prior_Diff` needs for its `marginal` noise model.

Requires DSSP (`mkdssp` on PATH), network access to files.rcsb.org, and MapDiff's full
dependency stack -- run inside the training Docker image, or a local env built per the
README.

Usage:
    python data/generate_graph_relaxed.py --ds-name atlas \
        --index-csv ./atlas_cross_val_index.csv --traj-h5 ./atlas_data.h5 \
        --save-root ./surffold_data/atlas_process/ \
        --marginal-out ./surffold_data/train_marginal_x_atlas.pt
"""
import argparse
import os
import sys
import tempfile

import torch
from tqdm import tqdm

# data/generate_graph_cath.py imports `dataloader.*`/`utils` assuming the repo root is on
# sys.path; make that true regardless of the caller's CWD (train.sh invokes this as
# `python data/generate_graph_relaxed.py` from the job's scratch root, not from inside data/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_graph_cath import pdb2graph
from dataloader.pyg_safe_globals import allow_pyg_data_pickles

# lib/relaxed_pdb.py + lib/dataset_splits.py are copied in flat next to this file (see
# ../README.md and ../train.sh) -- fall back to walking up to a lib/ directory for local/dev
# runs from inside the source tree.
try:
    import dataset_splits
    import relaxed_pdb
    import traj_frames
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import dataset_splits
    import relaxed_pdb
    import traj_frames


allow_pyg_data_pickles()


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ds-name', default='atlas', choices=dataset_splits.DATASETS,
                         help="Which shared dataset to featurize; trained/evaluated separately")
    parser.add_argument('--index-csv', default=None,
                         help="Split index csv (default: the dataset's standard file name)")
    parser.add_argument('--traj-h5', default=None,
                         help="Trajectory store (default: the dataset's standard file name). With "
                              "--structure-source relaxed it is read only for each protein's "
                              "reference residue sequence, and '' featurizes whole deposited "
                              "chains uncropped; with --structure-source frame it is the "
                              "coordinate source and is required.")
    parser.add_argument('--structure-source', default='relaxed', choices=('relaxed', 'frame'),
                         help="Where each protein's coordinates come from: its deposited RCSB "
                              "entry ('relaxed', the default) or one frame of its MD trajectory "
                              "('frame'). Use a separate --save-root per source.")
    parser.add_argument('--frame-seed', type=int, default=dataset_splits.SEED,
                         help="--structure-source frame only: seeds the per-protein frame draw")
    parser.add_argument('--frame-index', type=int, default=None,
                         help="--structure-source frame only: use this frame index for every "
                              "protein instead of drawing one at random (0 = first frame)")
    parser.add_argument('--val-fold', type=int, default=dataset_splits.DEFAULT_VAL_FOLD,
                         help="ATLAS only: cross_val fold used as the validation split")
    parser.add_argument('--test-fold', type=int, default=dataset_splits.DEFAULT_TEST_FOLD,
                         help="ATLAS only: cross_val fold held out as the test split (never trained on)")
    parser.add_argument('--pdb-cache', default='./pdb_cache',
                         help="Where downloaded RCSB entries are cached")
    parser.add_argument('--save-root', default=None,
                         help="Where to write the processed per-protein graph .pt files "
                              "(default: ./surffold_data/<ds_name>_process/)")
    parser.add_argument('--marginal-out', default=None,
                         help="Where to save the train-split amino-acid marginal distribution "
                              "(default: ./surffold_data/train_marginal_x_<ds_name>.pt)")
    return parser.parse_args()


def process_split(records, split_name, save_dir):
    """Featurize each loaded structure into a MapDiff graph .pt. Returns the ids that failed.

    Source-agnostic: `lib/traj_frames.py` hands back records in the same shape
    `lib/relaxed_pdb.py` does, so `write_record_pdb` and `pdb2graph` are unchanged. A frame's
    PDB is backbone-only, which MapDiff's featurizer already tolerates (`place_missing_cb` /
    `place_missing_o` fill CB and O); only DSSP's SASA is affected, so SASA is not comparable
    across the two sources.
    """
    os.makedirs(save_dir, exist_ok=True)
    pending = [p for p in records if not os.path.exists(os.path.join(save_dir, f'{p}.pt'))]
    if not pending:
        return []

    errors = []
    for protein_id in tqdm(pending, desc=f'processing {split_name}'):
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pdb')
        os.close(tmp_fd)
        try:
            relaxed_pdb.write_record_pdb(records[protein_id], tmp_path)
            graph = pdb2graph(tmp_path)
        except Exception as exc:
            print(f'skip {protein_id}: {exc}')
            errors.append(protein_id)
            continue
        finally:
            os.remove(tmp_path)

        if graph is None:
            errors.append(protein_id)
            continue
        torch.save(graph, os.path.join(save_dir, f'{protein_id}.pt'))
    return errors


def compute_marginal(save_dir, out_path, train_ids):
    """Amino-acid marginal over the *training* split only.

    Driven by `train_ids` rather than `os.listdir(save_dir)`: this distribution is the
    diffusion model's `marginal` noise prior, so folding in a stale graph left over from a
    run with a different --val-fold would leak held-out statistics into training.
    """
    counts = torch.zeros(20)
    filenames = [f'{p}.pt' for p in train_ids if os.path.exists(os.path.join(save_dir, f'{p}.pt'))]
    for filename in tqdm(filenames, desc='computing amino-acid marginal'):
        graph = torch.load(os.path.join(save_dir, filename))
        counts += graph.x[:, :20].sum(dim=0)
    marginal = counts / counts.sum()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(marginal, out_path)
    print(f'Saved amino-acid marginal over {len(filenames)} training proteins '
          f'({int(counts.sum())} residues) to {out_path}')


def main():
    args = create_parser()
    # The source is part of the default path: a relaxed run and a frame run must not write
    # their graphs into the same directory.
    suffix = '' if args.structure_source == 'relaxed' else f'_{args.structure_source}'
    save_root = args.save_root or f'./surffold_data/{args.ds_name}{suffix}_process/'
    marginal_out = args.marginal_out or f'./surffold_data/train_marginal_x_{args.ds_name}{suffix}.pt'
    traj_h5 = dataset_splits.default_h5(args.ds_name) if args.traj_h5 is None else args.traj_h5

    train_ids, val_ids, test_ids = dataset_splits.get_splits(
        args.ds_name, args.index_csv, val_fold=args.val_fold, test_fold=args.test_fold)
    all_ids = train_ids + val_ids + test_ids
    print(f'{args.ds_name}: {len(train_ids)} train / {len(val_ids)} validation / '
          f'{len(test_ids)} test proteins')

    if args.structure_source == 'frame':
        # A trajectory frame already spans exactly the simulated residues (for mdCATH, exactly
        # the domain), so there is nothing to crop and no RCSB download.
        if not traj_h5:
            raise SystemExit('--structure-source frame needs --traj-h5')
        records, failures = traj_frames.load_frame_structures(
            all_ids, traj_h5, seed=args.frame_seed, frame_index=args.frame_index)
        print(f'Loaded {len(records)} trajectory frames from {traj_h5}, {len(failures)} unavailable')
    else:
        # Crop each deposited chain to the residues the trajectory covers, so MapDiff scores the
        # same residue set as the sheaf model. For mdCATH this is also what reduces a deposited
        # chain to the single CATH *domain* the dataset id names.
        reference_seqs = {}
        if traj_h5:
            reference_seqs = relaxed_pdb.reference_seqs_from_h5(traj_h5, all_ids)
            print(f'Reference residue sequences found for {len(reference_seqs)}/'
                  f'{len(all_ids)} proteins in {traj_h5}')

        records, failures = relaxed_pdb.load_relaxed_structures(
            all_ids, cache_dir=args.pdb_cache, reference_seqs=reference_seqs)
        print(f'Loaded {len(records)} relaxed structures, {len(failures)} unavailable')

    all_errors = {
        'train': process_split({p: records[p] for p in train_ids if p in records},
                                'train', os.path.join(save_root, 'train')),
        'validation': process_split({p: records[p] for p in val_ids if p in records},
                                     'validation', os.path.join(save_root, 'validation')),
        # The test split gets its own directory: dataset.test_dir points here, and train.py
        # drives the file list from the split index, so a graph can never leak across splits.
        'test': process_split({p: records[p] for p in test_ids if p in records},
                               'test', os.path.join(save_root, 'test')),
    }

    for split, errors in all_errors.items():
        print(f'{split}: {len(errors)} structures failed to process')
        if errors:
            print(errors)

    train_save_dir = os.path.join(save_root, 'train')
    if os.path.isdir(train_save_dir) and len(os.listdir(train_save_dir)) > 0:
        compute_marginal(train_save_dir, marginal_out, train_ids)


if __name__ == '__main__':
    main()
