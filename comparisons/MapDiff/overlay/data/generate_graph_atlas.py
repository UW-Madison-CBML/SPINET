"""Turn raw ATLAS structures into MapDiff's per-residue graph format.

The ATLAS-derived dataset (the initial static structure of each MD
simulation, pre-split into train/validation/test folders of `.pdb` files)
is featurized the same way as CATH: `data.generate_graph_cath.pdb2graph`
runs DSSP for secondary structure and builds the k-NN residue graph used
by `dataloader.large_dataset.Cath`. This script just points that same
featurizer at the ATLAS split folders instead of `cath_download/`, and
additionally computes the train-split amino-acid marginal distribution
that `model.prior_diff.Prior_Diff` needs for its `marginal` noise model.

Requires DSSP (`mkdssp` on PATH) and MapDiff's full dependency stack --
run inside the CHTC docker image, or a local env built per the README.

Usage:
    python data/generate_graph_atlas.py \
        --pdb_root ./surffold_data/ \
        --save_root ./data/atlas/atlas_process/ \
        --marginal_out data/source/train_marginal_x_atlas.pt
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

# data/generate_graph_cath.py imports `dataloader.*`/`utils` assuming the repo
# root is on sys.path; make that true regardless of the caller's CWD (e.g. the
# CHTC run scripts invoke this as `python data/generate_graph_atlas.py` from
# the job's scratch root, not from inside data/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_graph_cath import pdb2graph


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pdb_root', default='./surffold_data/',
                         help="Root containing train/validation/test folders of raw ATLAS .pdb files")
    parser.add_argument('--save_root', default='./data/atlas/atlas_process/',
                         help="Where to write the processed per-protein graph .pt files")
    parser.add_argument('--splits', default='train,validation,test',
                         help="Comma-separated split folder names to process")
    parser.add_argument('--marginal_out', default='data/source/train_marginal_x_atlas.pt',
                         help="Where to save the train-split amino-acid marginal distribution")
    return parser.parse_args()


def process_split(pdb_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filenames = sorted(f for f in os.listdir(pdb_dir) if f.endswith('.pdb'))
    error_pdbs = []
    for filename in tqdm(filenames, desc=f'processing {pdb_dir}'):
        save_path = os.path.join(save_dir, filename.replace('.pdb', '.pt'))
        if os.path.exists(save_path):
            continue
        try:
            graph = pdb2graph(os.path.join(pdb_dir, filename))
        except Exception as exc:
            print(f'skip {filename}: {exc}')
            error_pdbs.append(filename)
            continue
        if graph is None:
            error_pdbs.append(filename)
            continue
        torch.save(graph, save_path)
    return error_pdbs


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
    splits = args.splits.split(',')

    all_errors = {}
    for split in splits:
        pdb_dir = os.path.join(args.pdb_root, split)
        save_dir = os.path.join(args.save_root, split)
        if not os.path.isdir(pdb_dir):
            raise FileNotFoundError(f"Expected an ATLAS split folder at {pdb_dir}")
        all_errors[split] = process_split(pdb_dir, save_dir)

    for split, errors in all_errors.items():
        print(f'{split}: {len(errors)} structures failed to process')
        if errors:
            print(errors)

    train_save_dir = os.path.join(args.save_root, 'train')
    if os.path.isdir(train_save_dir) and len(os.listdir(train_save_dir)) > 0:
        compute_marginal(train_save_dir, args.marginal_out)


if __name__ == '__main__':
    main()
