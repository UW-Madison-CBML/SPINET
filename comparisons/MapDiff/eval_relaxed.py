"""Evaluate a trained MapDiff (Prior_Diff) checkpoint on a dataset's held-out split.

Loads an EGNN+IPA Prior_Diff checkpoint produced by `train.py`, runs MC-DDIM inference over
every protein in the held-out split one at a time, and computes per-protein perplexity,
top-1/top-5/top-10 sequence recovery, and self-consistency RMSD (fold the predicted sequence
with ESMFold and Kabsch-RMSD it against the protein's ground-truth relaxed backbone). Reports
the mean +/- standard deviation (over proteins) for each metric and logs everything to
Weights & Biases.

Both shared datasets are supported and evaluated separately (`--ds-name`), the same splits
scripts/train_residue_classifier.py uses: ATLAS's held-out `cross_val` fold, or mdCATH's
`validation` rows. Each protein is featurized on the fly from its *relaxed* (deposited)
structure -- `lib/relaxed_pdb.py` downloads the RCSB entry, picks the chain the dataset id
names and crops it to the residues the trajectory covers -- then run through
`data.generate_graph_cath.pdb2graph`, exactly as train.py's data/generate_graph_relaxed.py
does. No precomputed graph directory is needed.

Must be run in an environment with MapDiff's full dependencies (torch, torch_geometric,
torch_scatter/torch_cluster, biopython, DSSP, wandb, transformers) -- e.g. the training
Docker image. See eval.sub / eval.sh to run this on CHTC.

Usage:
    python eval_relaxed.py --run_dir outputs/MapDiff_atlas_run --ds-name atlas \
        --index-csv ./atlas_cross_val_index.csv --traj-h5 ./atlas_data.h5 \
        --wandb_key_file ./wandb_api.txt
"""
import argparse
import json
import os
import sys
import tempfile

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

# lib/{stats_utils,scrmsd,relaxed_pdb,dataset_splits}.py are copied in flat next to this file
# (see README.md and eval.sh) -- fall back to walking up to a lib/ directory for local/dev
# runs from inside the source tree.
try:
    import dataset_splits
    import relaxed_pdb
    import stats_utils
    from scrmsd import load_esmfold, evaluate_scrmsd
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import dataset_splits
    import relaxed_pdb
    import stats_utils
    from scrmsd import load_esmfold, evaluate_scrmsd

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
    parser.add_argument('--ds-name', default='atlas', choices=dataset_splits.DATASETS,
                         help="Which shared dataset's held-out split to evaluate on")
    parser.add_argument('--index-csv', default=None,
                         help="Split index csv (default: the dataset's standard file name)")
    parser.add_argument('--traj-h5', default=None,
                         help="Trajectory store, read only for each protein's reference residue "
                              "sequence (default: the dataset's standard file name)")
    parser.add_argument('--pdb-cache', default='./pdb_cache', help="Where downloaded RCSB entries are cached")
    parser.add_argument('--val-fold', type=int, default=dataset_splits.DEFAULT_VAL_FOLD,
                         help="ATLAS only: cross_val fold to evaluate on (held out from training)")
    parser.add_argument('--max_length', default=dataset_splits.MAX_LENGTH, type=int,
                         help="Skip proteins longer than this many residues. Defaults to the cutoff "
                              "every model trains under (lib/dataset_splits.MAX_LENGTH) -- evaluating "
                              "without it would score MapDiff on proteins it never saw and that PiFold "
                              "dropped too. Pass 0 to disable.")
    parser.add_argument('--ensemble_num', default=None, type=int,
                         help="Override cfg.diffusion.ensemble_num (number of MC-dropout DDIM samples averaged "
                              "per protein)")
    parser.add_argument('--ddim_steps', default=None, type=int, help="Override cfg.diffusion.ddim_steps")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb_key_file', default='./wandb_api.txt')
    parser.add_argument('--wandb_entity', default='jenslundsgaard7-uw-madison')
    parser.add_argument('--wandb_project', default='SheafProtein')
    parser.add_argument('--wandb_run_name', default=None)
    return parser.parse_args()


def find_checkpoint(run_dir):
    model_dir = os.path.join(run_dir, 'model')
    candidates = sorted(f for f in os.listdir(model_dir) if f.endswith('.pt'))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint .pt files found under {model_dir}")
    best = [f for f in candidates if '_best_' in f]
    chosen = best[-1] if best else candidates[-1]
    return os.path.join(model_dir, chosen)


def load_validation_entries(ds_name, index_csv, traj_h5, pdb_cache, val_fold, max_length):
    """Featurize the relaxed (deposited) structure of every held-out protein."""
    _, val_ids = dataset_splits.get_splits(ds_name, index_csv, val_fold=val_fold)
    traj_h5 = dataset_splits.default_h5(ds_name) if traj_h5 is None else traj_h5

    reference_seqs = relaxed_pdb.reference_seqs_from_h5(traj_h5, val_ids) if traj_h5 else {}
    records, failures = relaxed_pdb.load_relaxed_structures(
        val_ids, cache_dir=pdb_cache, reference_seqs=reference_seqs)
    if failures:
        print(f'{len(failures)}/{len(val_ids)} held-out proteins had no usable deposited structure')

    entries = []
    for protein_id in tqdm(val_ids, desc=f'featurizing {ds_name} held-out proteins'):
        record = records.get(protein_id)
        if record is None:
            continue

        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pdb')
        os.close(tmp_fd)
        try:
            relaxed_pdb.write_record_pdb(record, tmp_path)
            graph = get_processed_graph(pdb2graph(tmp_path))
        except Exception as exc:
            print(f'skip {protein_id}: {exc}')
            continue
        finally:
            os.remove(tmp_path)

        if graph is None:
            continue
        if max_length and graph.x.shape[0] > max_length:
            continue
        entries.append({'title': protein_id, 'graph': graph})
    return entries


@torch.no_grad()
def evaluate(model, entries, collator, ensemble_num, ddim_steps, device, esmfold_tokenizer, esmfold_model):
    per_protein = []
    for entry in tqdm(entries, desc='evaluating held-out set'):
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
        # The graph was featurized from the deposited structure, so `atom_pos` is the relaxed
        # reference, residue-for-residue aligned with the prediction.
        gt_backbone = entry['graph'].atom_pos[:, ATOM_POS_TO_BACKBONE_IDX, :]  # (R, 4, 3)
        scrmsd = evaluate_scrmsd([pred_seq], [gt_backbone], esmfold_tokenizer, esmfold_model,
                                  device=device).item()

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

    entries = load_validation_entries(args.ds_name, args.index_csv, args.traj_h5, args.pdb_cache,
                                       args.val_fold, args.max_length)
    if len(entries) == 0:
        raise RuntimeError("No usable held-out proteins found for dataset {}".format(args.ds_name))
    print("Loaded {} held-out {} proteins".format(len(entries), args.ds_name))

    esmfold_tokenizer, esmfold_model = load_esmfold(device=device)

    collator = CollatorDiff()
    per_protein = evaluate(model, entries, collator, ensemble_num, ddim_steps, device,
                            esmfold_tokenizer, esmfold_model)
    # ddof=1 (sample std) matches this script's previous hand-rolled aggregation.
    summary = stats_utils.summarize_per_protein(per_protein, METRIC_KEYS, ddof=1)

    print(json.dumps(summary, indent=2))

    run = stats_utils.init_wandb(
        args.wandb_key_file, args.wandb_entity, args.wandb_project,
        args.wandb_run_name or 'Eval_{}_MapDiff'.format(args.ds_name.upper()),
        config={
            'checkpoint': checkpoint_path,
            'config_path': config_path,
            'ds_name': args.ds_name,
            'index_csv': args.index_csv or dataset_splits.default_index_csv(args.ds_name),
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
