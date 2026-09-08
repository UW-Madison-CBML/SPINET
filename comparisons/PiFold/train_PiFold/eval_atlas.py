"""Evaluate a trained PiFold checkpoint on the ATLAS validation split.

Loads a ProDesign_Model checkpoint, runs inference over every protein in
the ATLAS `validation/` folder one at a time, and computes per-protein
perplexity and top-1/top-5/top-10 sequence recovery. Reports the mean
+/- standard deviation (over proteins) for each metric and logs everything
to Weights & Biases.

Must be run in an environment with PiFold's full dependencies (torch,
torch_scatter, biopython, wandb) -- e.g. the CHTC docker image used for
training -- since `methods/__init__.py` pulls in torch_scatter. See
chtc/eval.sub / chtc/run_eval.sh to run this against the staged ATLAS
validation data on CHTC.

Usage:
    python eval_atlas.py \
        --run_dir results/6133697_0/chtc_run \
        --data_root ./surffold_data/ \
        --wandb_key_file ./wandb_api.txt
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from API.atlas_dataset import ALPHABET, parse_pdb_backbone
from API.featurizer import featurize_GTrans
from methods.prodesign_model import ProDesign_Model

# lib/stats_utils.py is transferred/copied in flat next to this file (see
# README.md and chtc/run_eval.sh) -- fall back to walking up to a lib/
# directory for local/dev runs from inside the source tree.
try:
    import stats_utils
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import stats_utils

METRIC_KEYS = stats_utils.METRIC_KEYS


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', default='results/6133697_0/chtc_run',
                         help="Directory containing checkpoint.pth and model_param.json")
    parser.add_argument('--checkpoint', default=None,
                         help="Path to checkpoint .pth (default: <run_dir>/checkpoint.pth)")
    parser.add_argument('--model_param', default=None,
                         help="Path to model_param.json (default: <run_dir>/model_param.json)")
    parser.add_argument('--data_root', default='./surffold_data/',
                         help="ATLAS data root containing a validation/ split")
    parser.add_argument('--max_length', default=None, type=int,
                         help="Override the max sequence length filter (default: value from model_param.json)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb_key_file', default='./wandb_api.txt')
    parser.add_argument('--wandb_entity', default='jenslundsgaard7-uw-madison')
    parser.add_argument('--wandb_project', default='SheafProtein')
    parser.add_argument('--wandb_run_name', default='Eval_Test_PiFold')
    return parser.parse_args()


def load_validation_entries(data_root, max_length):
    """Load ATLAS validation-split proteins directly from `{data_root}/validation/`."""
    split_dir = os.path.join(data_root, 'validation')
    if not os.path.isdir(split_dir):
        raise FileNotFoundError("Could not find a 'validation' split folder under {}".format(data_root))

    alphabet_set = set(ALPHABET)
    entries = []
    files = sorted(f for f in os.listdir(split_dir) if f.endswith('.pdb') or f.endswith('.pdb.gz'))
    for fname in tqdm(files, desc='loading ATLAS/validation'):
        fpath = os.path.join(split_dir, fname)
        parsed = parse_pdb_backbone(fpath)
        if parsed is None:
            continue

        seq = parsed['seq']
        if set(seq).difference(alphabet_set):
            continue
        if max_length is not None and len(seq) > max_length:
            continue

        title = fname
        for suffix in ('.pdb.gz', '.pdb'):
            if title.endswith(suffix):
                title = title[:-len(suffix)]
                break

        entries.append({
            'title': title,
            'seq': seq,
            'CA': parsed['CA'],
            'C': parsed['C'],
            'O': parsed['O'],
            'N': parsed['N'],
        })
    return entries


@torch.no_grad()
def evaluate(model, entries, device):
    """Per-protein perplexity + top-k recovery.

    `model(...)` returns log-softmax scores, so NLLLoss on them gives the
    true per-residue negative log-likelihood -- equivalent to running
    CrossEntropyLoss on raw logits (as the reference GVP training/eval loop
    does), and matching ProDesign.test_one_epoch's loss_nll_flatten rather
    than the double-softmaxed loss used by train/valid_one_epoch. Since each
    `entry` here is evaluated on its own (one PDB == one "batch" == one
    protein), grouping per protein and averaging per residue within it is
    already exactly what the reference script's per-batch protein loop does.
    """
    per_protein = []
    for entry in tqdm(entries, desc='evaluating ATLAS validation set'):
        X, S, score, mask, lengths = featurize_GTrans([entry])
        X, S, score, mask = X.to(device), S.to(device), score.to(device), mask.to(device)

        X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = \
            model._get_features(S, score, X=X, mask=mask)
        log_probs = model(h_V, h_E, E_idx, batch_id)

        raw_loss = F.nll_loss(log_probs, S, reduction='none')

        per_protein.append({
            'title': entry['title'],
            'length': len(entry['seq']),
            'perplexity': stats_utils.perplexity_from_nll(raw_loss.mean()),
            'recovery_top1': stats_utils.top_k_acc(log_probs, S, 1),
            'recovery_top5': stats_utils.top_k_acc(log_probs, S, 5),
            'recovery_top10': stats_utils.top_k_acc(log_probs, S, 10),
        })
    return per_protein


def main():
    args = create_parser()

    checkpoint_path = args.checkpoint or os.path.join(args.run_dir, 'checkpoint.pth')
    model_param_path = args.model_param or os.path.join(args.run_dir, 'model_param.json')

    with open(model_param_path) as f:
        model_args = SimpleNamespace(**json.load(f))

    max_length = args.max_length if args.max_length is not None else getattr(model_args, 'max_length', 500)

    device = torch.device(args.device)
    model = ProDesign_Model(model_args).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    entries = load_validation_entries(args.data_root, max_length)
    if len(entries) == 0:
        raise RuntimeError("No usable validation proteins found under {}".format(args.data_root))
    print("Loaded {} ATLAS validation proteins from {}".format(len(entries), args.data_root))

    per_protein = evaluate(model, entries, device)
    summary = stats_utils.summarize_per_protein(per_protein, METRIC_KEYS)

    print(json.dumps(summary, indent=2))

    run = stats_utils.init_wandb(
        args.wandb_key_file, args.wandb_entity, args.wandb_project, args.wandb_run_name,
        config={
            'checkpoint': checkpoint_path,
            'model_param': vars(model_args),
            'data_root': args.data_root,
            'max_length': max_length,
            'n_validation_proteins': len(entries),
        },
    )
    run.log(summary)
    stats_utils.log_per_protein_table(per_protein, METRIC_KEYS)
    run.finish()


if __name__ == '__main__':
    main()
