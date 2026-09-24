"""Overlaid onto a pristine PiFold checkout (see run_pifold.sh).

Adds the dataset-selection flags our two datasets need (`--data_name ATLAS|MDCATH`, plus the
split index / trajectory store / PDB cache locations) and the Weights & Biases wiring
`main.py` uses to log recovery and perplexity.

The defaults that have to agree with the other comparison models (held-out fold, max protein
length, RNG seed) are read straight out of `lib/dataset_splits.py` rather than spelled out
here, so there is exactly one place to change them. That module is copied in flat next to
this file (see ../README.md and ../run_pifold.sh).
"""
import argparse
import os
import sys

try:
    import dataset_splits
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import dataset_splits


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    # Shared with MapDiff/DynamicMPNN (lib/dataset_splits.SEED) so every comparison run
    # draws the same stream -- weight init, batch order, dropout.
    parser.add_argument('--seed', default=dataset_splits.SEED, type=int)

    # dataset parameters. ATLAS and MDCATH are trained/evaluated separately, one run each --
    # see API/relaxed_dataset.py and lib/dataset_splits.py.
    parser.add_argument('--data_name', default='CATH', choices=['CATH', 'TS50', 'ATLAS', 'MDCATH'])
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--index_csv', default='', type=str,
                         help='Split index csv (default: the dataset standard name under --data_root)')
    parser.add_argument('--traj_h5', default=None, type=str,
                         help="Trajectory store, read only for each protein's reference residue sequence "
                              "(default: the dataset standard name under --data_root). Pass '' to featurize "
                              "whole deposited chains uncropped.")
    parser.add_argument('--pdb_cache', default='', type=str,
                         help='Where downloaded RCSB entries are cached (default: <data_root>/pdb_cache)')
    parser.add_argument('--cath_dir', default='', type=str,
                         help='MDCATH relaxed only: directory holding the extracted CATH domain files '
                              '(mdcath_pdbs.tar.gz), used whole with no cropping (default: <data_root>/pdbs)')
    parser.add_argument('--structure_source', default='relaxed', choices=['relaxed', 'frame'],
                         help="What each protein's coordinates come from: its deposited RCSB entry "
                              "('relaxed', the default) or one frame of its MD trajectory ('frame')")
    parser.add_argument('--frame_seed', default=dataset_splits.SEED, type=int,
                         help='--structure_source frame only: seeds the per-protein frame draw')
    parser.add_argument('--frame_index', default=None, type=int,
                         help='--structure_source frame only: use this frame index for every protein '
                              'instead of drawing one at random (0 = first frame)')
    parser.add_argument('--val_fold', default=dataset_splits.DEFAULT_VAL_FOLD, type=int,
                         help='ATLAS only: cross_val fold used as the validation split')
    parser.add_argument('--test_fold', default=dataset_splits.DEFAULT_TEST_FOLD, type=int,
                         help='ATLAS only: cross_val fold held out as the test split -- never '
                              'trained on and never used for model selection')
    parser.add_argument('--max_length', default=dataset_splits.MAX_LENGTH, type=int,
                         help='Proteins longer than this (in residues) are dropped from every split. '
                              'Shared with MapDiff (lib/dataset_splits.MAX_LENGTH) -- raising it here '
                              'alone would score PiFold on proteins MapDiff never sees.')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    # method parameters
    parser.add_argument('--method', default='ProDesign', choices=['ProDesign'])
    parser.add_argument('--config_file', '-c', default=None, type=str)
    parser.add_argument('--hidden_dim',  default=128, type=int)
    parser.add_argument('--node_features',  default=128, type=int)
    parser.add_argument('--edge_features',  default=128, type=int)
    parser.add_argument('--k_neighbors',  default=30, type=int)
    parser.add_argument('--dropout',  default=0.1, type=int)
    parser.add_argument('--num_encoder_layers', default=10, type=int)

    # Training parameters
    parser.add_argument('--epoch', default=100, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--patience', default=10, type=int,
                        help='Epochs without a new best val perplexity before stopping')

    # ProDesign parameters
    parser.add_argument('--updating_edges', default=4, type=int)
    parser.add_argument('--node_dist', default=1, type=int)
    parser.add_argument('--node_angle', default=1, type=int)
    parser.add_argument('--node_direct', default=1, type=int)
    parser.add_argument('--edge_dist', default=1, type=int)
    parser.add_argument('--edge_angle', default=1, type=int)
    parser.add_argument('--edge_direct', default=1, type=int)
    parser.add_argument('--virtual_num', default=3, type=int)

    # Weights & Biases
    parser.add_argument('--wandb', default=1, type=int, help='Log to W&B (0 disables)')
    parser.add_argument('--wandb_key_file', default='./wandb_api.txt')
    parser.add_argument('--wandb_entity', default='jenslundsgaard7-uw-madison')
    parser.add_argument('--wandb_project', default='SheafProtein')
    parser.add_argument('--wandb_run_name', default='', type=str)

    return parser.parse_args()
