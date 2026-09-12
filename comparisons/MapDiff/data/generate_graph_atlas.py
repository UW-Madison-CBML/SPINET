"""Turn ATLAS structures into MapDiff's per-residue graph format.

Reads one representative frame per protein directly from the shared `atlas_data.h5` +
`atlas_cross_val_index.csv` -- the same store and split used by
scripts/train_residue_classifier.py, comparisons/gvp-pytorch/train.py, and
comparisons/DynamicMPNN/train.py -- instead of pre-split raw-PDB folders. Fold 0 of the
`cross_val` column is held out as validation, *and reused as the test split too* (MapDiff
wants train/val/test, but ATLAS only has one held-out fold; see ../conf/dataset/atlas.yaml,
where `val_dir`/`test_dir` both point at this script's "validation" output).

Each protein's representative frame (the `random_indices` column, the same frame
scripts/train_residue_classifier.py / comparisons/gvp-pytorch/train.py pick) is written out
as a minimal single-model PDB (lib/atlas_frame_pdb.write_frame_pdb) so DSSP secondary
structure assignment (data.generate_graph_cath.pdb2graph) works unchanged, then deleted.
Also computes the train-split amino-acid marginal distribution that `model.prior_diff.
Prior_Diff` needs for its `marginal` noise model.

Requires DSSP (`mkdssp` on PATH) and MapDiff's full dependency stack -- run inside the
training Docker image, or a local env built per the README.

Usage:
    python data/generate_graph_atlas.py \
        --atlas-h5 ./atlas_data.h5 \
        --cross-val-csv ./atlas_cross_val_index.csv \
        --save-root ./surffold_data/atlas_process/ \
        --marginal-out ./surffold_data/train_marginal_x_atlas.pt
"""
import argparse
import os
import sys
import tempfile

import h5py
import pandas as pd
import torch
from tqdm import tqdm

# data/generate_graph_cath.py imports `dataloader.*`/`utils` assuming the repo root is on
# sys.path; make that true regardless of the caller's CWD (train.sh invokes this as
# `python data/generate_graph_atlas.py` from the job's scratch root, not from inside data/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_graph_cath import pdb2graph

# lib/atlas_frame_pdb.py is copied in flat next to this file (see ../README.md and
# ../train.sh) -- fall back to walking up to a lib/ directory for local/dev runs from
# inside the source tree.
try:
    from atlas_frame_pdb import get_atlas_splits, get_frame_index, index_h5_groups, extract_frame, write_frame_pdb
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    from atlas_frame_pdb import get_atlas_splits, get_frame_index, index_h5_groups, extract_frame, write_frame_pdb


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--atlas-h5', default='./atlas_data.h5', help="Path to the shared ATLAS hdf5 store")
    parser.add_argument('--cross-val-csv', default='./atlas_cross_val_index.csv',
                         help="Path to the shared ATLAS cross-validation index")
    parser.add_argument('--val-fold', type=int, default=0,
                         help="cross_val fold held out for validation/test (folds != this train)")
    parser.add_argument('--save-root', default='./surffold_data/atlas_process/',
                         help="Where to write the processed per-protein graph .pt files")
    parser.add_argument('--marginal-out', default='./surffold_data/train_marginal_x_atlas.pt',
                         help="Where to save the train-split amino-acid marginal distribution")
    return parser.parse_args()


def process_split(pdb_ids, split_name, save_dir, atlas_h5_path, index_df):
    os.makedirs(save_dir, exist_ok=True)
    pending = [p for p in pdb_ids if not os.path.exists(os.path.join(save_dir, f'{p}.pt'))]
    if not pending:
        return []

    errors = []
    with h5py.File(atlas_h5_path, 'r') as h5_file:
        group_lookup = index_h5_groups(h5_file)

        for pdb_id in tqdm(pending, desc=f'processing {split_name}'):
            group_name = group_lookup.get(pdb_id)
            if group_name is None:
                print(f'skip {pdb_id}: no matching trajectory group in {atlas_h5_path}')
                errors.append(pdb_id)
                continue

            frame_idx = get_frame_index(index_df, pdb_id)
            frame = extract_frame(h5_file, group_name, frame_idx)

            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pdb')
            os.close(tmp_fd)
            try:
                write_frame_pdb(tmp_path, frame)
                graph = pdb2graph(tmp_path)
            except Exception as exc:
                print(f'skip {pdb_id}: {exc}')
                errors.append(pdb_id)
                continue
            finally:
                os.remove(tmp_path)

            if graph is None:
                errors.append(pdb_id)
                continue
            torch.save(graph, os.path.join(save_dir, f'{pdb_id}.pt'))
    return errors


def compute_marginal(save_dir, out_path):
    counts = torch.zeros(20)
    for filename in tqdm(os.listdir(save_dir), desc='computing amino-acid marginal'):
        graph = torch.load(os.path.join(save_dir, filename))
        counts += graph.x[:, :20].sum(dim=0)
    marginal = counts / counts.sum()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(marginal, out_path)
    print(f'Saved amino-acid marginal ({int(counts.sum())} residues) to {out_path}')


def main():
    args = create_parser()
    index_df = pd.read_csv(args.cross_val_csv)
    train_pdbs, val_pdbs = get_atlas_splits(args.cross_val_csv, val_fold=args.val_fold)

    all_errors = {
        'train': process_split(train_pdbs, 'train', os.path.join(args.save_root, 'train'),
                                args.atlas_h5, index_df),
        # Fold 0 doubles as both validation and test -- one processed copy, referenced by
        # both dataset.val_dir and dataset.test_dir (see ../conf/dataset/atlas.yaml).
        'validation': process_split(val_pdbs, 'validation', os.path.join(args.save_root, 'validation'),
                                     args.atlas_h5, index_df),
    }

    for split, errors in all_errors.items():
        print(f'{split}: {len(errors)} structures failed to process')
        if errors:
            print(errors)

    train_save_dir = os.path.join(args.save_root, 'train')
    if os.path.isdir(train_save_dir) and len(os.listdir(train_save_dir)) > 0:
        compute_marginal(train_save_dir, args.marginal_out)


if __name__ == '__main__':
    main()
