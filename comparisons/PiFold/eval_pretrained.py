"""Zero-shot evaluation of PRE-TRAINED PiFold weights on a dataset's held-out split -- no training.

Overlaid onto a pristine PiFold checkout (see eval_pretrained.sh). train.sub trains PiFold from
scratch on our data; this is the other half of the comparison: published CATH4.2 weights, our
splits, our metrics.

* The checkpoint (`--checkpoint`, e.g. model_zoom/CATH4.2/PiFold/results.pt) is loaded as a
  state dict into upstream's `ProDesign_Model`, built with the released CATH4.2 hyperparameters
  (this script's defaults, the same as parser.py's). Wrapper dicts (`state_dict` / `model` /
  `model_state_dict`) and `model.` / `module.` key prefixes (Lightning, DataParallel) are
  unwrapped first, and the load is strict -- a mismatch fails loudly instead of scoring a
  half-random model.
* Proteins come from `API.relaxed_dataset.RelaxedStructures` exactly as in training (relaxed
  deposited structure cropped to the trajectory's residues, or `--structure_source frame`),
  but only the scored split is loaded, so no training protein is downloaded.
* Metrics are the per-protein pass main.py's `residue_metrics` logs (lib/stats_utils.ResidueMetrics:
  top-1/5/10 recovery, perplexity, precision/recall/f1, confusion matrix, `test_pred_seqs`), plus
  self-consistency RMSD (ESMFold-fold the design, Kabsch-RMSD against the input backbone), the
  same scRMSD MapDiff's eval_relaxed.py reports. `--scrmsd 0` skips ESMFold.
"""
import argparse
import json
import os
import os.path as osp
import sys

import torch
from tqdm import tqdm

from API.relaxed_dataset import RelaxedStructures
from API.featurizer import featurize_GTrans
from methods.prodesign_model import ProDesign_Model
from methods.utils import cuda

try:
    import dataset_splits
    import stats_utils
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = osp.join(osp.dirname(osp.abspath(__file__)), _up, 'lib')
        if osp.isdir(_cand):
            sys.path.insert(0, osp.abspath(_cand))
            break
    import dataset_splits
    import stats_utils

# PiFold's fixed label order (API.featurizer.featurize_GTrans).
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
# featurize_GTrans stacks [N, CA, C, O]; lib/scrmsd.py wants relaxed_pdb.BACKBONE_ATOMS order
# (CA, N, C, O).
FEATURIZER_TO_BACKBONE_IDX = [1, 0, 2, 3]
METRIC_KEYS = stats_utils.METRIC_KEYS + ('scrmsd',)
DATA_NAMES = {'ATLAS': 'atlas', 'MDCATH': 'mdcath'}


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='Pre-trained PiFold weights (.pt/.pth)')
    parser.add_argument('--weights_name', default='CATH4.2',
                         help='What the weights were trained on -- only used to name the W&B run')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--esmfold_device', default=None, help='Where ESMFold runs (default: --device)')
    parser.add_argument('--scrmsd', default=1, type=int, help='Compute ESMFold scRMSD (0 disables)')
    parser.add_argument('--res_dir', default='./results/pretrained_eval',
                         help='Where the summary json and per-protein metrics are written')

    # dataset -- same meaning and defaults as parser.py
    parser.add_argument('--data_name', default='MDCATH', choices=sorted(DATA_NAMES))
    parser.add_argument('--split', default='test', choices=('test', 'valid'))
    parser.add_argument('--data_root', default='./')
    parser.add_argument('--index_csv', default='', type=str)
    parser.add_argument('--traj_h5', default=None, type=str)
    parser.add_argument('--pdb_cache', default='', type=str)
    parser.add_argument('--structure_source', default='relaxed', choices=['relaxed', 'frame'])
    parser.add_argument('--frame_seed', default=dataset_splits.SEED, type=int)
    parser.add_argument('--frame_index', default=None, type=int)
    parser.add_argument('--val_fold', default=dataset_splits.DEFAULT_VAL_FOLD, type=int)
    parser.add_argument('--test_fold', default=dataset_splits.DEFAULT_TEST_FOLD, type=int)
    parser.add_argument('--max_length', default=dataset_splits.MAX_LENGTH, type=int)

    # ProDesign_Model hyperparameters -- the released CATH4.2 configuration (= parser.py's defaults)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--node_features', default=128, type=int)
    parser.add_argument('--edge_features', default=128, type=int)
    parser.add_argument('--k_neighbors', default=30, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_encoder_layers', default=10, type=int)
    parser.add_argument('--updating_edges', default=4, type=int)
    parser.add_argument('--node_dist', default=1, type=int)
    parser.add_argument('--node_angle', default=1, type=int)
    parser.add_argument('--node_direct', default=1, type=int)
    parser.add_argument('--edge_dist', default=1, type=int)
    parser.add_argument('--edge_angle', default=1, type=int)
    parser.add_argument('--edge_direct', default=1, type=int)
    parser.add_argument('--virtual_num', default=3, type=int)

    parser.add_argument('--wandb', default=1, type=int, help='Log to W&B (0 disables)')
    parser.add_argument('--wandb_key_file', default='./wandb_api.txt')
    parser.add_argument('--wandb_entity', default='jenslundsgaard7-uw-madison')
    parser.add_argument('--wandb_project', default='SheafProtein')
    parser.add_argument('--wandb_run_name', default='', type=str)
    return parser.parse_args()


def load_pretrained(model, checkpoint_path, device):
    """Load a released PiFold checkpoint into `model`, tolerating the usual wrappers."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model', 'net'):
            if isinstance(checkpoint.get(key), dict):
                print('>>> checkpoint: using its {!r} entry'.format(key))
                checkpoint = checkpoint[key]
                break
    state = dict(checkpoint)

    own_keys = set(model.state_dict())
    for prefix in ('module.', 'model.'):
        if not (set(state) & own_keys) and all(k.startswith(prefix) for k in state):
            print('>>> checkpoint: stripping {!r} key prefix'.format(prefix))
            state = {k[len(prefix):]: v for k, v in state.items()}

    missing = sorted(own_keys - set(state))
    unexpected = sorted(set(state) - own_keys)
    if missing or unexpected:
        raise RuntimeError('checkpoint {} does not match ProDesign_Model -- check the hyperparameter '
                           'flags.\n  missing ({}): {}\n  unexpected ({}): {}'.format(
                               checkpoint_path, len(missing), missing[:20], len(unexpected), unexpected[:20]))
    model.load_state_dict(state, strict=True)
    print('>>> loaded {} tensors from {}'.format(len(state), checkpoint_path))


@torch.no_grad()
def evaluate(model, dataset, device, split_name, esmfold=None, esmfold_device=None):
    """One protein at a time, the same pass as main.py's `residue_metrics` plus scRMSD."""
    model.eval()
    nll_loss = torch.nn.NLLLoss()
    accumulator = stats_utils.ResidueMetrics(stats_utils.three_letter_codes(ALPHABET), val_name=split_name)
    per_protein = []

    for entry in tqdm(dataset, desc='designing {} split'.format(split_name)):
        X, S, score, mask, lengths = cuda(featurize_GTrans([entry]), device=device)
        # featurize_GTrans compacts away residues with missing backbone atoms, so the valid
        # rows of X are residue-for-residue aligned with the masked S the model predicts.
        reference = X[0][mask[0] == 1][:, FEATURIZER_TO_BACKBONE_IDX].cpu()  # (R, 4, 3)
        X, S, score, h_V, h_E, E_idx, batch_id, _, _, _ = model._get_features(S, score, X=X, mask=mask)
        log_probs = model(h_V, h_E, E_idx, batch_id).cpu()
        S = S.cpu()

        design = ''.join(ALPHABET[i] for i in log_probs.argmax(dim=1).tolist())
        nll = nll_loss(log_probs, S).item()
        accumulator.add_protein(log_probs, S, pdb_id=entry['title'], nll=nll, sequence=design)

        row = {
            'title': entry['title'],
            'length': int(S.shape[0]),
            'perplexity': stats_utils.perplexity_from_nll(torch.tensor(nll)),
            'recovery_top1': stats_utils.top_k_acc(log_probs, S, 1),
            'recovery_top5': stats_utils.top_k_acc(log_probs, S, 5),
            'recovery_top10': stats_utils.top_k_acc(log_probs, S, 10),
            'design': design,
        }
        if esmfold is not None:
            from scrmsd import evaluate_scrmsd
            tokenizer, esm_model = esmfold
            row['scrmsd'] = evaluate_scrmsd([design], [reference], tokenizer, esm_model,
                                             device=esmfold_device).item()
            accumulator.add_scrmsd([row['scrmsd']])
        per_protein.append(row)

    return accumulator, per_protein


def main():
    args = create_parser()
    device = torch.device(args.device)
    ds_name = DATA_NAMES[args.data_name]

    model = ProDesign_Model(args).to(device)
    load_pretrained(model, args.checkpoint, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dataset = RelaxedStructures(
        args.data_root, mode=args.split, max_length=args.max_length, ds_name=ds_name,
        index_csv=args.index_csv or None, traj_h5=args.traj_h5, pdb_cache=args.pdb_cache or None,
        val_fold=args.val_fold, test_fold=args.test_fold, structure_source=args.structure_source,
        frame_seed=args.frame_seed, frame_index=args.frame_index, splits=(args.split,))
    if len(dataset) == 0:
        raise RuntimeError('no usable {} proteins for {}'.format(args.split, ds_name))
    print('>>> scoring {} {} {} proteins'.format(len(dataset), ds_name, args.split))

    esmfold = None
    esmfold_device = args.esmfold_device or args.device
    if args.scrmsd:
        from scrmsd import load_esmfold
        esmfold = load_esmfold(device=str(esmfold_device))

    split_name = 'test' if args.split == 'test' else 'val'
    accumulator, per_protein = evaluate(model, dataset, device, split_name, esmfold, esmfold_device)

    metric_keys = METRIC_KEYS if args.scrmsd else stats_utils.METRIC_KEYS
    summary = stats_utils.summarize_per_protein(per_protein, metric_keys)
    summary['n_proteins'] = len(per_protein)
    print(json.dumps(summary, indent=2))
    print(accumulator.summary_line(num_params))

    os.makedirs(args.res_dir, exist_ok=True)
    with open(osp.join(args.res_dir, 'summary.json'), 'w') as f:
        json.dump({'args': vars(args), 'summary': summary,
                   'dropped': getattr(dataset, 'dropped', {})}, f, indent=2)
    with open(osp.join(args.res_dir, 'per_protein.json'), 'w') as f:
        json.dump(per_protein, f, indent=2)

    if args.wandb:
        run = stats_utils.init_wandb(
            args.wandb_key_file, args.wandb_entity, args.wandb_project,
            args.wandb_run_name or 'Eval_PiFold_pretrained{}_{}_{}'.format(
                args.weights_name, dataset_splits.dataset_tag(ds_name, args.traj_h5), args.structure_source),
            config=dict(vars(args), n_eval_proteins=len(per_protein), params=num_params))
        run.summary['{}_split_ids'.format(split_name)] = [p['title'] for p in per_protein]
        run.summary['dropped_proteins'] = getattr(dataset, 'dropped', {})
        run.log({'params': num_params, **summary})
        run.log(accumulator.to_log_dict(
            cm_title='{} Amino Acid Confusion Matrix'.format(split_name.capitalize())))
        stats_utils.log_per_protein_table(per_protein, metric_keys)
        run.finish()


if __name__ == '__main__':
    main()
