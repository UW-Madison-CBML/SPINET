"""Overlaid onto a pristine PiFold checkout (see run_pifold.sh).

Adds the dataset-selection flags our two datasets need (`--data_name ATLAS|MDCATH`, plus the
split index / trajectory store / PDB cache locations) and the Weights & Biases wiring
`main.py` uses to log recovery, perplexity and scRMSD.
"""
import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=111, type=int)

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
    parser.add_argument('--val_fold', default=0, type=int, help='ATLAS only: cross_val fold held out')
    parser.add_argument('--max_length', default=500, type=int, help='Max sequence length')
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
    parser.add_argument('--patience', default=100, type=int)

    # ProDesign parameters
    parser.add_argument('--updating_edges', default=4, type=int)
    parser.add_argument('--node_dist', default=1, type=int)
    parser.add_argument('--node_angle', default=1, type=int)
    parser.add_argument('--node_direct', default=1, type=int)
    parser.add_argument('--edge_dist', default=1, type=int)
    parser.add_argument('--edge_angle', default=1, type=int)
    parser.add_argument('--edge_direct', default=1, type=int)
    parser.add_argument('--virtual_num', default=3, type=int)

    # Self-consistency RMSD (ESMFold) -- see lib/scrmsd.py
    parser.add_argument('--scrmsd', default=1, type=int,
                         help='Compute scRMSD against the relaxed structures on the held-out split (0 disables)')
    parser.add_argument('--scrmsd_batch_size', default=4, type=int,
                         help='How many designs ESMFold folds per call')

    # Weights & Biases
    parser.add_argument('--wandb', default=1, type=int, help='Log to W&B (0 disables)')
    parser.add_argument('--wandb_key_file', default='./wandb_api.txt')
    parser.add_argument('--wandb_entity', default='jenslundsgaard7-uw-madison')
    parser.add_argument('--wandb_project', default='SheafProtein')
    parser.add_argument('--wandb_run_name', default='', type=str)

    return parser.parse_args()
