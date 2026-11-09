"""Evaluate a trained MapDiff (Prior_Diff) checkpoint on a dataset's held-out test split.

Loads an EGNN+IPA Prior_Diff checkpoint produced by `train.py`, runs MC-DDIM inference over
every protein in the held-out split one at a time, and computes per-protein perplexity
and top-1/top-5/top-10 sequence recovery. Reports the mean +/- standard deviation (over
proteins) for each metric and logs everything to Weights & Biases.

Both shared datasets are supported and evaluated separately (`--ds-name`), the same splits
scripts/train_residue_classifier.py uses. The default is the **test** split -- ATLAS's
`cross_val` fold `--test-fold`, or mdCATH's `test` rows -- which training never saw and never
selected on; pass `--split val` to score the validation split instead.
`--structure-source traj` scores each protein by averaging its logits across every MD frame,
featurizing each frame on the fly in `--featurize-workers` DataLoader workers (nothing written to
disk), or -- with `--graph-root` -- from data/graphs_from_traj.py's saved graphs, featurizing
missing ones there first. Otherwise each protein is
featurized on the fly from its *relaxed* (deposited)
structure -- `lib/relaxed_pdb.py` downloads the RCSB entry, picks the chain the dataset id
names and crops it to the residues the trajectory covers -- then run through
`data.generate_graph_cath.pdb2graph`, exactly as train.py's data/generate_graph_relaxed.py
does. No precomputed graph directory is needed.

Must be run in an environment with MapDiff's full dependencies (torch, torch_geometric,
torch_scatter/torch_cluster, biopython, DSSP, wandb) -- e.g. the training
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
from dataloader.traj_dataset import (EnsembleCollator, OnlineTrajEnsembles, OnlineTrajFrames, TrajEnsembles,
                                     average_frame_logits)
from dataloader.pyg_inspector_compat import patch_inspector_distribute
from dataloader.pyg_safe_globals import allow_pyg_data_pickles
from model.egnn_pytorch.egnn_net import EGNN_NET
from model.ipa.ipa_net import IPANetPredictor
from model.prior_diff import Prior_Diff
from utils import enable_dropout

# lib/{stats_utils,relaxed_pdb,dataset_splits}.py are copied in flat next to this file
# (see README.md and eval.sh) -- fall back to walking up to a lib/ directory for local/dev
# runs from inside the source tree.
try:
    import dataset_splits
    import relaxed_pdb
    import stats_utils
    import traj_frames
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import dataset_splits
    import relaxed_pdb
    import stats_utils
    import traj_frames

allow_pyg_data_pickles()
patch_inspector_distribute()

METRIC_KEYS = stats_utils.METRIC_KEYS

# Fixed one-letter amino-acid order MapDiff's 20-dim one-hot (`data.x[:, :20]`)
# is built in -- see (upstream) data/generate_graph_cath.py's `amino_acids_type`.
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


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
    parser.add_argument('--split', default='test', choices=('test', 'val'),
                         help="Which held-out split to score (default: the test split, which "
                              "training neither fit nor selected on)")
    parser.add_argument('--index-csv', default=None,
                         help="Split index csv (default: the dataset's standard file name)")
    parser.add_argument('--traj-h5', default=None,
                         help="Trajectory store, read only for each protein's reference residue "
                              "sequence (default: the dataset's standard file name)")
    parser.add_argument('--pdb-cache', default='./pdb_cache', help="Where downloaded RCSB entries are cached")
    parser.add_argument('--structure-source', default='relaxed', choices=('relaxed', 'frame', 'traj'),
                         help="What each protein's coordinates come from: its deposited RCSB entry "
                              "('relaxed', the default), one frame of its MD trajectory ('frame'), or "
                              "every frame ('traj': logits averaged across each protein's frames). "
                              "Match whatever the checkpoint was trained on.")
    parser.add_argument('--graph-root', default=None,
                         help="--structure-source traj only: read saved graphs from data/graphs_from_traj.py's "
                              "--save-root, featurizing frames of the scored split that are not there yet "
                              "into it first. Default: featurize every frame on the fly, nothing saved")
    parser.add_argument('--frame-stride', type=int, default=1,
                         help="--structure-source traj only: featurize every N-th frame")
    parser.add_argument('--max-frames', type=int, default=None,
                         help="--structure-source traj only: at most this many evenly spaced frames per "
                              "protein are averaged")
    parser.add_argument('--frame-chunk', type=int, default=8,
                         help="--structure-source traj only: frames per forward pass")
    parser.add_argument('--featurize-workers', type=int, default=8,
                         help="--structure-source traj only: featurizing workers (frame chunks on the "
                              "fly, or proteins with --graph-root)")
    parser.add_argument('--frame-seed', type=int, default=dataset_splits.SEED,
                         help="--structure-source frame only: seeds the per-protein frame draw. Use "
                              "the same seed training used to score the same frames.")
    parser.add_argument('--frame-index', type=int, default=None,
                         help="--structure-source frame only: use this frame index for every protein "
                              "instead of drawing one at random (0 = first frame)")
    parser.add_argument('--val-fold', type=int, default=dataset_splits.DEFAULT_VAL_FOLD,
                         help="ATLAS only: cross_val fold used as the validation split")
    parser.add_argument('--test-fold', type=int, default=dataset_splits.DEFAULT_TEST_FOLD,
                         help="ATLAS only: cross_val fold held out as the test split")
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


def load_split_entries(ds_name, index_csv, traj_h5, pdb_cache, val_fold, test_fold, max_length,
                        split='test', structure_source='relaxed', frame_seed=dataset_splits.SEED,
                        frame_index=None):
    """Featurize every protein in `split` ('test'/'val') from its chosen structure source.

    `structure_source` must match what the checkpoint was trained on -- scoring a
    frame-trained model on deposited structures (or the reverse) measures the domain shift,
    not the model. See data/generate_graph_relaxed.py's --structure-source.
    """
    _, val_ids, test_ids = dataset_splits.get_splits(
        ds_name, index_csv, val_fold=val_fold, test_fold=test_fold)
    split_ids = test_ids if split == 'test' else val_ids
    traj_h5 = dataset_splits.default_h5(ds_name) if traj_h5 is None else traj_h5

    if structure_source == 'frame':
        if not traj_h5:
            raise SystemExit('--structure-source frame needs --traj-h5')
        records, failures = traj_frames.load_frame_structures(
            split_ids, traj_h5, seed=frame_seed, frame_index=frame_index)
        if failures:
            print(f'{len(failures)}/{len(split_ids)} {split} proteins had no usable trajectory frame')
    else:
        reference_seqs = relaxed_pdb.reference_seqs_from_h5(traj_h5, split_ids) if traj_h5 else {}
        records, failures = relaxed_pdb.load_relaxed_structures(
            split_ids, cache_dir=pdb_cache, reference_seqs=reference_seqs)
        if failures:
            print(f'{len(failures)}/{len(split_ids)} {split} proteins had no usable deposited structure')

    entries = []
    for protein_id in tqdm(split_ids, desc=f'featurizing {ds_name} {split} proteins'):
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


def load_traj_ensembles(args, split):
    """Per-protein frame ensembles of the scored split: featurized on the fly
    (`OnlineTrajEnsembles`), or -- with ``--graph-root`` -- a `TrajEnsembles` over saved graphs,
    featurizing missing frames there first."""
    _, val_ids, test_ids = dataset_splits.get_splits(
        args.ds_name, args.index_csv, val_fold=args.val_fold, test_fold=args.test_fold)
    split_ids = test_ids if split == 'test' else val_ids
    traj_h5 = args.traj_h5 or dataset_splits.default_h5(args.ds_name)
    if args.graph_root is None:
        frames = OnlineTrajFrames(traj_h5, split_ids, args.ds_name, max_length=args.max_length or None,
                                  max_frames=args.max_frames, stride=args.frame_stride, fallback=False)
        return OnlineTrajEnsembles(frames, args.frame_chunk, args.featurize_workers)

    from data import graphs_from_traj
    split_dir_name = 'test' if split == 'test' else 'validation'
    split_dir = os.path.join(args.graph_root, split_dir_name)

    store = traj_frames.TrajectoryStore(traj_h5)
    entries, failures = store.index(split_ids, args.ds_name, stride=args.frame_stride, max_frames=args.max_frames,
                                    max_length=args.max_length or None)
    store.close()
    if failures:
        print(f'{len(failures)}/{len(split_ids)} {split} proteins had no usable trajectory: {failures}')
    # Resumable: frames already featurized are skipped.
    graphs_from_traj.process_split(entries, split_dir_name, split_dir, args.featurize_workers, traj_h5)

    return TrajEnsembles(split_dir, [p for p in split_ids if p in entries],
                         max_length=args.max_length or None, max_frames=args.max_frames)


@torch.no_grad()
def mc_logits(model, g_batch, ipa_batch, ensemble_num, ddim_steps, device):
    g_batch, ipa_batch = g_batch.to(device), ipa_batch.to(device)
    samples = [model.mc_ddim_sample(g_batch, ipa_batch, diverse=True, step=ddim_steps)[0]
               for _ in range(ensemble_num)]
    return torch.stack(samples).mean(dim=0).cpu()


def protein_metrics(title, mean_logits, target):
    target_onehot = F.one_hot(target, num_classes=mean_logits.shape[1]).float()
    return {
        'title': title,
        'length': target.shape[0],
        'design': ''.join(AMINO_ACIDS[i.item()] for i in mean_logits.argmax(dim=1)),
        'perplexity': stats_utils.perplexity_from_nll(F.cross_entropy(mean_logits, target_onehot, reduction='mean')),
        'recovery_top1': stats_utils.top_k_acc(mean_logits, target, 1),
        'recovery_top5': stats_utils.top_k_acc(mean_logits, target, 5),
        'recovery_top10': stats_utils.top_k_acc(mean_logits, target, 10),
    }


@torch.no_grad()
def evaluate_traj(model, ensembles, frame_chunk, ensemble_num, ddim_steps, device):
    """Per protein: MC-sample mean on every frame, then the mean across frames."""
    per_protein = []
    if isinstance(ensembles, OnlineTrajEnsembles):
        loader = ensembles  # already yields per-protein items, featurizing in its own workers
    else:
        loader = torch.utils.data.DataLoader(ensembles, batch_size=1, shuffle=False, num_workers=4,
                                             collate_fn=EnsembleCollator(chunk_size=frame_chunk))
    for item in tqdm(loader, desc='evaluating held-out split (frame-averaged)'):
        # Consume every chunk before reading num_frames/target (filled in as chunks are featurized).
        frame_logits = [mc_logits(model, g, ipa, ensemble_num, ddim_steps, device) for g, ipa in item['chunks']]
        if not frame_logits:
            print(f"{item['name']}: no frame could be featurized, skipping")
            continue
        mean_logits = average_frame_logits(torch.cat(frame_logits), item['num_frames'])
        record = protein_metrics(item['name'], mean_logits, item['target'])
        record['num_frames'] = item['num_frames']
        per_protein.append(record)
    return per_protein


@torch.no_grad()
def evaluate(model, entries, collator, ensemble_num, ddim_steps, device):
    per_protein = []
    for entry in tqdm(entries, desc='evaluating held-out split'):
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

        per_protein.append({
            'title': entry['title'],
            'length': target.shape[0],
            'design': pred_seq,
            'perplexity': stats_utils.perplexity_from_nll(F.cross_entropy(mean_logits, target_onehot, reduction='mean')),
            'recovery_top1': stats_utils.top_k_acc(mean_logits, target, 1),
            'recovery_top5': stats_utils.top_k_acc(mean_logits, target, 5),
            'recovery_top10': stats_utils.top_k_acc(mean_logits, target, 10),
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

    if args.structure_source == 'traj':
        entries = load_traj_ensembles(args, args.split)
    else:
        entries = load_split_entries(args.ds_name, args.index_csv, args.traj_h5, args.pdb_cache,
                                      args.val_fold, args.test_fold, args.max_length, split=args.split,
                                      structure_source=args.structure_source,
                                      frame_seed=args.frame_seed, frame_index=args.frame_index)
    if len(entries) == 0:
        raise RuntimeError("No usable {} proteins found for dataset {}".format(args.split, args.ds_name))
    print("Loaded {} {} {} proteins".format(len(entries), args.ds_name, args.split))

    if args.structure_source == 'traj':
        per_protein = evaluate_traj(model, entries, args.frame_chunk, ensemble_num, ddim_steps, device)
    else:
        per_protein = evaluate(model, entries, CollatorDiff(), ensemble_num, ddim_steps, device)
    # ddof=1 (sample std) matches this script's previous hand-rolled aggregation.
    summary = stats_utils.summarize_per_protein(per_protein, METRIC_KEYS, ddof=1)

    print(json.dumps(summary, indent=2))

    run = stats_utils.init_wandb(
        args.wandb_key_file, args.wandb_entity, args.wandb_project,
        args.wandb_run_name or 'Eval_MapDiff_{}_{}'.format(
            dataset_splits.dataset_tag(args.ds_name, args.traj_h5), args.structure_source),
        config={
            'checkpoint': checkpoint_path,
            'config_path': config_path,
            'ds_name': args.ds_name,
            'index_csv': args.index_csv or dataset_splits.default_index_csv(args.ds_name),
            'split': args.split,
            'val_fold': args.val_fold,
            'test_fold': args.test_fold,
            'max_length': args.max_length,
            'ensemble_num': ensemble_num,
            'ddim_steps': ddim_steps,
            'n_eval_proteins': len(entries),
            'structure_source': args.structure_source,
            'max_frames': args.max_frames if args.structure_source == 'traj' else None,
        },
    )
    run.log(summary)
    stats_utils.log_per_protein_table(per_protein, METRIC_KEYS)
    # The designs, in the same table trainer.py logs via ResidueMetrics.to_log_dict:
    # `test_pred_seqs` (bare `pred_seqs` for val), columns pdb / seq.
    import wandb
    run.log({'pred_seqs' if args.split == 'val' else '{}_pred_seqs'.format(args.split): wandb.Table(
        columns=['pdb', 'seq'], data=[[p['title'], p['design']] for p in per_protein])})
    run.finish()


if __name__ == '__main__':
    main()
