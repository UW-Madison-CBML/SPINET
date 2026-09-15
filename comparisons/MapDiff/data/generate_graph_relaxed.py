"""Turn *relaxed* (deposited) structures into MapDiff's per-residue graph format.

MapDiff is a static-structure inverse-folding model, so it is trained and evaluated on the
deposited PDB entry for each protein in our datasets -- not on a frame pulled out of the MD
trajectory, which is what this script used to do. `lib/relaxed_pdb.py` downloads the entry
from RCSB, picks the chain the dataset id names, and crops it to the residues the trajectory
covers (using the trajectory's own residue list as the reference sequence), so MapDiff scores
the same residues as every other model in the comparison. DynamicMPNN deliberately still
trains on MD ensembles -- that ensemble input is the thing being benchmarked.

Both shared datasets are supported, trained/evaluated separately (`--ds-name`), matching
scripts/train_residue_classifier.py:

* ``atlas``  -- fold 0 of `atlas_cross_val_index.csv`'s `cross_val` column is held out and,
  because ATLAS has no further held-out set, reused as the test split too (see
  ../conf/dataset/atlas.yaml, where `val_dir`/`test_dir` both point at this script's
  "validation" output).
* ``mdcath`` -- `mdcath_320_0_topology_split.csv`'s train+test rows train, and **evaluation
  is on the `validation` rows only**.

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
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import dataset_splits
    import relaxed_pdb


allow_pyg_data_pickles()


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ds-name', default='atlas', choices=dataset_splits.DATASETS,
                         help="Which shared dataset to featurize; trained/evaluated separately")
    parser.add_argument('--index-csv', default=None,
                         help="Split index csv (default: the dataset's standard file name)")
    parser.add_argument('--traj-h5', default=None,
                         help="Trajectory store, read only for each protein's reference residue "
                              "sequence (default: the dataset's standard file name). Pass '' to "
                              "featurize whole deposited chains uncropped.")
    parser.add_argument('--val-fold', type=int, default=dataset_splits.DEFAULT_VAL_FOLD,
                         help="ATLAS only: cross_val fold held out for validation/test (folds != this train)")
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
    """Featurize each loaded relaxed structure into a MapDiff graph .pt. Returns the ids that failed."""
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
    save_root = args.save_root or f'./surffold_data/{args.ds_name}_process/'
    marginal_out = args.marginal_out or f'./surffold_data/train_marginal_x_{args.ds_name}.pt'
    traj_h5 = dataset_splits.default_h5(args.ds_name) if args.traj_h5 is None else args.traj_h5

    train_ids, val_ids = dataset_splits.get_splits(args.ds_name, args.index_csv, val_fold=args.val_fold)
    print(f'{args.ds_name}: {len(train_ids)} train / {len(val_ids)} validation proteins')

    # Crop each deposited chain to the residues the trajectory covers, so MapDiff scores the
    # same residue set as the sheaf model. For mdCATH this is also what reduces a deposited
    # chain to the single CATH *domain* the dataset id names.
    reference_seqs = {}
    if traj_h5:
        reference_seqs = relaxed_pdb.reference_seqs_from_h5(traj_h5, train_ids + val_ids)
        print(f'Reference residue sequences found for {len(reference_seqs)}/'
              f'{len(train_ids) + len(val_ids)} proteins in {traj_h5}')

    records, failures = relaxed_pdb.load_relaxed_structures(
        train_ids + val_ids, cache_dir=args.pdb_cache, reference_seqs=reference_seqs)
    print(f'Loaded {len(records)} relaxed structures, {len(failures)} unavailable')

    all_errors = {
        'train': process_split({p: records[p] for p in train_ids if p in records},
                                'train', os.path.join(save_root, 'train')),
        # The held-out split doubles as both validation and test -- one processed copy,
        # referenced by both dataset.val_dir and dataset.test_dir (see ../conf/dataset/).
        'validation': process_split({p: records[p] for p in val_ids if p in records},
                                     'validation', os.path.join(save_root, 'validation')),
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
