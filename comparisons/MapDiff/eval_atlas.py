"""Evaluate a trained MapDiff (Prior_Diff) checkpoint on the ATLAS validation/test split.

Loads an EGNN+IPA Prior_Diff checkpoint produced by `train.py`, runs MC-DDIM inference over
every protein in fold 0 of `atlas_cross_val_index.csv` one at a time (the same held-out
fold scripts/train_residue_classifier.py and comparisons/{gvp-pytorch,DynamicMPNN}/train.py
use), featurizing each protein's representative frame directly from `atlas_data.h5` (via
lib/atlas_frame_pdb + `data.generate_graph_cath.pdb2graph`, same as train.py's
data/generate_graph_atlas.py), and computes per-protein perplexity, top-1/top-5/top-10
sequence recovery, and self-consistency RMSD (fold the predicted sequence with ESMFold and
Kabsch-RMSD it against the protein's ground-truth backbone). Reports the mean +/- standard
deviation (over proteins) for each metric and logs everything to Weights & Biases.

Must be run in an environment with MapDiff's full dependencies (torch,
torch_geometric, torch_scatter/torch_cluster, biopython, DSSP, wandb,
transformers) -- e.g. the training Docker image. See eval.sub / train.sh to
run this against the staged ATLAS data on CHTC.

Usage:
    python eval_atlas.py \
        --run_dir outputs/MapDiff_atlas_run \
        --atlas-h5 ./atlas_data.h5 --cross-val-csv ./atlas_cross_val_index.csv \
        --wandb_key_file ./wandb_api.txt
"""
import argparse
import json
import os
import sys
import tempfile

import h5py
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from data.generate_graph_cath import pdb2graph, get_processed_graph
from dataloader.collator import CollatorDiff
from model.egnn_pytorch.egnn_net import EGNN_NET
from model.ipa.ipa_net import IPANetPredictor
from model.prior_diff import Prior_Diff
from utils import enable_dropout

# lib/stats_utils.py + lib/scrmsd.py + lib/atlas_frame_pdb.py are copied in flat next to
# this file (see README.md and eval.sh) -- fall back to walking up to a lib/ directory for
# local/dev runs from inside the source tree.
try:
    import stats_utils
    from scrmsd import load_esmfold, evaluate_batch_rmsd
    from atlas_frame_pdb import get_atlas_splits, get_frame_index, index_h5_groups, extract_frame, write_frame_pdb
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import stats_utils
    from scrmsd import load_esmfold, evaluate_batch_rmsd
    from atlas_frame_pdb import get_atlas_splits, get_frame_index, index_h5_groups, extract_frame, write_frame_pdb

METRIC_KEYS = stats_utils.METRIC_KEYS + ('scrmsd',)

# Fixed one-letter amino-acid order MapDiff's 20-dim one-hot (`data.x[:, :20]`)
# is built in -- see (upstream) data/generate_graph_cath.py's `amino_acids_type`.
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
# `atom_pos` stacks [N, CA, C, CB, O] -- reorder to lib/scrmsd.py's expected CA, N, C, O.
ATOM_POS_TO_BACKBONE_IDX = [1, 0, 2, 4]


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', default='outputs/MapDiff_atlas_run',
                         help="Hydra output dir from train.py, used to default --checkpoint/--config")
    parser.add_argument('--checkpoint', default=None,
                         help="Path to the trained checkpoint .pt (default: newest *_best_*.pt under "
                              "<run_dir>/model/)")
    parser.add_argument('--config', default=None,
                         help="Path to the run's resolved Hydra config.yaml (default: <run_dir>/configs/config.yaml)")
    parser.add_argument('--atlas-h5', default='./atlas_data.h5', help="Path to the shared ATLAS hdf5 store")
    parser.add_argument('--cross-val-csv', default='./atlas_cross_val_index.csv',
                         help="Path to the shared ATLAS cross-validation index")
    parser.add_argument('--val-fold', type=int, default=0,
                         help="cross_val fold to evaluate on (held out from training)")
    parser.add_argument('--max_length', default=None, type=int,
                         help="Skip proteins longer than this many residues")
    parser.add_argument('--ensemble_num', default=None, type=int,
                         help="Override cfg.diffusion.ensemble_num (number of MC-dropout DDIM samples averaged "
                              "per protein)")
    parser.add_argument('--ddim_steps', default=None, type=int, help="Override cfg.diffusion.ddim_steps")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb_key_file', default='./wandb_api.txt')
    parser.add_argument('--wandb_entity', default='jenslundsgaard7-uw-madison')
    parser.add_argument('--wandb_project', default='SheafProtein')
    parser.add_argument('--wandb_run_name', default='Eval_ATLAS_MapDiff')
    return parser.parse_args()


def find_checkpoint(run_dir):
    model_dir = os.path.join(run_dir, 'model')
    candidates = sorted(f for f in os.listdir(model_dir) if f.endswith('.pt'))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint .pt files found under {model_dir}")
    best = [f for f in candidates if '_best_' in f]
    chosen = best[-1] if best else candidates[-1]
    return os.path.join(model_dir, chosen)


def load_validation_entries(atlas_h5_path, cross_val_csv, val_fold, max_length):
    """Featurize fold 0's representative frame for every held-out ATLAS protein, directly
    from atlas_data.h5 -- the same store/split scripts/train_residue_classifier.py and
    comparisons/{gvp-pytorch,DynamicMPNN}/train.py use."""
    index_df = pd.read_csv(cross_val_csv)
    _, val_pdbs = get_atlas_splits(cross_val_csv, val_fold=val_fold)

    entries = []
    with h5py.File(atlas_h5_path, 'r') as h5_file:
        group_lookup = index_h5_groups(h5_file)

        for pdb_id in tqdm(val_pdbs, desc='featurizing ATLAS fold-{} proteins'.format(val_fold)):
            group_name = group_lookup.get(pdb_id)
            if group_name is None:
                print(f'skip {pdb_id}: no matching trajectory group in {atlas_h5_path}')
                continue

            frame_idx = get_frame_index(index_df, pdb_id)
            frame = extract_frame(h5_file, group_name, frame_idx)

            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pdb')
            os.close(tmp_fd)
            try:
                write_frame_pdb(tmp_path, frame)
                graph = get_processed_graph(pdb2graph(tmp_path))
            except Exception as exc:
                print(f'skip {pdb_id}: {exc}')
                continue
            finally:
                os.remove(tmp_path)

            if graph is None:
                continue
            if max_length is not None and graph.x.shape[0] > max_length:
                continue
            entries.append({'title': pdb_id, 'graph': graph})
    return entries


@torch.no_grad()
def evaluate(model, entries, collator, ensemble_num, ddim_steps, device, esmfold_tokenizer, esmfold_model):
    per_protein = []
    for entry in tqdm(entries, desc='evaluating ATLAS validation set'):
        g_batch, ipa_batch = collator([entry['graph']])
        g_batch, ipa_batch = g_batch.to(device), ipa_batch.to(device)

        ens_logits = []
        for _ in range(ensemble_num):
            logits, _ = model.mc_ddim_sample(g_batch, ipa_batch, diverse=True, step=ddim_steps)
            ens_logits.append(logits)
        mean_logits = torch.stack(ens_logits).mean(dim=0).cpu()

        target_onehot = g_batch.x.cpu()
        target = target_onehot.argmax(dim=1)

        pred_seq = ''.join(AMINO_ACIDS[i.item()] for i in mean_logits.argmax(dim=1))
        gt_backbone = entry['graph'].atom_pos[:, ATOM_POS_TO_BACKBONE_IDX, :]  # (R, 4, 3)
        scrmsd = evaluate_batch_rmsd(
            [pred_seq], gt_backbone[None, None, :, :, :],
            torch.ones(1, 1, gt_backbone.shape[0], dtype=torch.bool),
            esmfold_tokenizer, esmfold_model, device=device,
        ).item()

        per_protein.append({
            'title': entry['title'],
            'length': target.shape[0],
            'perplexity': stats_utils.perplexity_from_nll(F.cross_entropy(mean_logits, target_onehot, reduction='mean')),
            'recovery_top1': stats_utils.top_k_acc(mean_logits, target, 1),
            'recovery_top5': stats_utils.top_k_acc(mean_logits, target, 5),
            'recovery_top10': stats_utils.top_k_acc(mean_logits, target, 10),
            'scrmsd': scrmsd,
        })
    return per_protein


def main():
    args = create_parser()

    checkpoint_path = args.checkpoint or find_checkpoint(args.run_dir)
    config_path = args.config or os.path.join(args.run_dir, 'configs', 'config.yaml')

    cfg = OmegaConf.load(config_path)
    device = torch.device(args.device)

    egnn = EGNN_NET(input_feat_dim=cfg.model.input_feat_dim, hidden_channels=cfg.model.hidden_dim,
                     edge_attr_dim=cfg.model.edge_attr_dim, dropout=cfg.model.drop_out, n_layers=cfg.model.depth,
                     update_edge=cfg.model.update_edge, norm_coors=cfg.model.norm_coors,
                     update_coors=cfg.model.update_coors, update_global=cfg.model.update_global,
                     embedding=cfg.model.embedding, embedding_dim=cfg.model.embedding_dim,
                     norm_feat=cfg.model.norm_feat, embed_ss=cfg.model.embed_ss)
    ipa = IPANetPredictor(dropout=cfg.model.ipa_drop_out, max_length=cfg.model.get('ipa_pe_max_len', 1200))
    model = Prior_Diff(egnn, ipa, timesteps=cfg.diffusion.timesteps, objective=cfg.diffusion.objective,
                        noise_type=cfg.diffusion.noise_type, sample_method=cfg.diffusion.sample_method,
                        min_mask_ratio=cfg.mask_prior.min_mask_ratio, dev_mask_ratio=cfg.mask_prior.dev_mask_ratio,
                        marginal_dist_path=cfg.dataset.marginal_train_dir).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    # MC-dropout ensembling drives mc_ddim_sample's uncertainty estimate, so dropout must stay
    # active even in eval mode -- matches trainer.py's test() and model_inference.ipynb.
    enable_dropout(model)

    ensemble_num = args.ensemble_num or cfg.diffusion.ensemble_num
    ddim_steps = args.ddim_steps or cfg.diffusion.ddim_steps

    entries = load_validation_entries(args.atlas_h5, args.cross_val_csv, args.val_fold, args.max_length)
    if len(entries) == 0:
        raise RuntimeError("No usable validation proteins found in fold {} of {}".format(
            args.val_fold, args.cross_val_csv))
    print("Loaded {} ATLAS fold-{} proteins from {}".format(len(entries), args.val_fold, args.atlas_h5))

    esmfold_tokenizer, esmfold_model = load_esmfold(device=device)

    collator = CollatorDiff()
    per_protein = evaluate(model, entries, collator, ensemble_num, ddim_steps, device,
                            esmfold_tokenizer, esmfold_model)
    # ddof=1 (sample std) matches this script's previous hand-rolled aggregation.
    summary = stats_utils.summarize_per_protein(per_protein, METRIC_KEYS, ddof=1)

    print(json.dumps(summary, indent=2))

    run = stats_utils.init_wandb(
        args.wandb_key_file, args.wandb_entity, args.wandb_project, args.wandb_run_name,
        config={
            'checkpoint': checkpoint_path,
            'config_path': config_path,
            'atlas_h5': args.atlas_h5,
            'cross_val_csv': args.cross_val_csv,
            'val_fold': args.val_fold,
            'max_length': args.max_length,
            'ensemble_num': ensemble_num,
            'ddim_steps': ddim_steps,
            'n_validation_proteins': len(entries),
        },
    )
    run.log(summary)
    stats_utils.log_per_protein_table(per_protein, METRIC_KEYS)
    run.finish()


if __name__ == '__main__':
    main()
