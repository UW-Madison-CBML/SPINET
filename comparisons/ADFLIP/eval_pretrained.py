"""Zero-shot evaluation of PRE-TRAINED ADFLIP weights on a dataset's held-out split -- no training.

The other half of the ADFLIP comparison is that there is no training half: ADFLIP's released
checkpoint (`results/weights/ADFLIP_v1.pt`) was trained on ~149K PDB entries -- effectively all of
the PDB, split by MMseqs cluster -- so retraining it on our few thousand chains is a different
experiment. This script scores the released weights on our splits with our metrics, exactly as
PiFold/eval_pretrained.py does, and reports the training-overlap caveat as a number instead of a
footnote (see `--novel-subset`).

What is being compared
----------------------
ADFLIP conditions on an all-atom context and refines its own prediction over flow steps, so this
runs the *adaptive* sampler only (`multi_state.adaptive_sample_multi_state`). Upstream's
fixed-step `sample()` never rebuilds side chains from its own prediction -- it reuses whatever
the input file carried -- so depending on the input it either leaks the native answer or runs
with no side-chain context at all; see multi_state.py's docstring.

`--n-states N` is the paper's single-vs-multiple axis (Table 3):

* `--n-states 1` -- one conformation, reproducing upstream `adaptive_sample`. This is the
  "ADFLIP single" row and the apples-to-apples comparison against the other static baselines.
* `--n-states N>1` -- Algorithm 1: the denoiser runs on all N conformations each step and the
  per-state distributions are averaged. This is the "ADFLIP multiple" row and the one the
  reviewer asked for.

Both draw from the same frame pool (`traj_frames.TrajectoryStore.index` -> `select_frames`, the
sheaf model's own frames thinned to N evenly spaced ones), so the only thing that changes between
the two rows is how many of them the model sees. `--n-states 1` uses the first of that pool.

Three honest differences from the paper's numbers, worth stating wherever these are reported:

* The paper's gains are measured on residues within 5 A of non-protein atoms, in NMR ensembles.
  ATLAS and mdCATH are ligand-free single chains, so the all-atom context ADFLIP was built around
  is absent and only its ensemble mechanism is under test. The interacting-residue recovery it
  reports has no counterpart here.
* Our conformations are MD frames, not NMR models.
* With `--structure-source frame` the input PDBs hold backbone atoms only -- N, CA, C and O, see
  `traj_frames.frame_record` -- so ADFLIP starts with no side chains anywhere and builds every
  one of them itself while decoding. That makes the comparison against the backbone-only
  baselines a fair one, but it also means ADFLIP never gets the all-atom input it was trained
  on, so this path understates it. `--structure-source relaxed` keeps the deposited side chains
  and is the configuration that favours it.

Metrics
-------
`lib/stats_utils.ResidueMetrics` -- the same per-protein top-1/5/10 recovery, perplexity,
precision/recall/f1 and confusion matrix the sheaf model and every other comparison logs.
ADFLIP's vocabulary is wider than 20 (``<PAD>``, ``<MASK>``, 20 standard residues, ``<UNK>``,
nucleotides), so the final averaged distribution is restricted to the 20 standard residues and
renormalised before scoring; residues whose native token is not one of the 20 are dropped.
Recovery is computed from the tokens the sampler actually drew, not from the argmax of that
distribution -- for a sampler those differ, and the drawn sequence is the design.

Must be run in an environment with ADFLIP's dependencies (torch, torch-scatter/torch-cluster,
prody, biopython, ema_pytorch, hydra-core) plus PIPPack's, since side-chain repacking is part of
the adaptive sampler. See eval_pretrained.sh.

Usage:
    python eval_pretrained.py --ckpt ADFLIP/results/weights/ADFLIP_v1.pt \
        --data-name MDCATH --traj-h5 ./mdcath_spinet_320_0.h5 --n-states 8
"""
import argparse
import json
import os
import os.path as osp
import shutil
import sys
import tempfile

import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = osp.dirname(osp.abspath(__file__))
ADFLIP_ROOT = osp.join(HERE, 'ADFLIP')
if ADFLIP_ROOT not in sys.path:
    sys.path.insert(0, ADFLIP_ROOT)

try:
    import dataset_splits
    import relaxed_pdb
    import stats_utils
    import traj_frames
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = osp.join(HERE, _up, 'lib')
        if osp.isdir(_cand):
            sys.path.insert(0, osp.abspath(_cand))
            break
    import dataset_splits
    import relaxed_pdb
    import stats_utils
    import traj_frames

import multi_state  # noqa: E402

residue_tokens = multi_state.residue_tokens

# The 20-class label order every other comparison in this repo scores in.
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
# ADFLIP vocab index for each of those, so its logits can be restricted to the 20 standard
# residues. Built by name rather than assumed contiguous: upstream fills these from
# `prody.flagDefinition('stdaa')`, whose order is prody's, not ours.
_AA3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
        'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
STANDARD_TOKENS = [residue_tokens[_AA3[letter]] for letter in ALPHABET]
# Inverse: ADFLIP token -> 0..19, or absent for <PAD>/<MASK>/<UNK>/nucleotides.
TOKEN_TO_CLASS = {token: i for i, token in enumerate(STANDARD_TOKENS)}

# The checkpoint's output layer is indexed by this vocabulary, so a vocabulary that is not the
# one it was trained with would score a mis-labelled model rather than fail. Upstream builds the
# 20 standard residues as one contiguous block after <PAD> and <MASK>; check that holds.
if sorted(STANDARD_TOKENS) != list(range(min(STANDARD_TOKENS), min(STANDARD_TOKENS) + 20)):
    raise RuntimeError(
        'ADFLIP vocabulary changed: the 20 standard residues are not contiguous ({}). The '
        'released checkpoint was trained against the old layout.'.format(sorted(STANDARD_TOKENS)))

DATA_NAMES = {'ATLAS': 'atlas', 'MDCATH': 'mdcath'}


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', default=osp.join(ADFLIP_ROOT, 'results/weights/ADFLIP_v1.pt'),
                        help='Pre-trained ADFLIP weights')
    parser.add_argument('--weights-name', default='ADFLIP_v1',
                        help='What the weights were trained on -- only used to name the W&B run')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--res-dir', default='./results/pretrained_eval',
                        help='Where the summary json and per-protein metrics are written')
    parser.add_argument('--keep-pdbs', default='', type=str,
                        help='Keep the per-frame input PDBs under this directory (default: a '
                             'temporary directory, removed afterwards)')

    # dataset -- same meaning and defaults as the other comparisons' eval scripts
    parser.add_argument('--data-name', default='MDCATH', choices=sorted(DATA_NAMES))
    parser.add_argument('--split', default='test', choices=('test', 'valid'))
    parser.add_argument('--index-csv', default='', type=str)
    parser.add_argument('--traj-h5', default=None, type=str)
    parser.add_argument('--pdb-cache', default='./pdb_cache', type=str)
    parser.add_argument('--cath-dir', default=relaxed_pdb.DEFAULT_CATH_DIR, type=str)
    parser.add_argument('--structure-source', default='frame', choices=('frame', 'relaxed'),
                        help="'frame' takes conformations from the trajectory store (the only "
                             "source that supports --n-states > 1) and is backbone-only; "
                             "'relaxed' scores the single deposited/domain structure, as the "
                             'other static baselines default to. Only `relaxed` carries real '
                             'side chains, and only there can a residue with an incomplete '
                             'backbone appear -- ADFLIP never masks those, so they keep their '
                             'native identity and side chain as free context while being left '
                             'out of the metrics. Worth checking how many there are before '
                             'reporting a `relaxed` number.')
    parser.add_argument('--val-fold', default=dataset_splits.DEFAULT_VAL_FOLD, type=int)
    parser.add_argument('--test-fold', default=dataset_splits.DEFAULT_TEST_FOLD, type=int)
    parser.add_argument('--max-length', default=dataset_splits.MAX_LENGTH, type=int)
    parser.add_argument('--frame-stride', default=1, type=int,
                        help="Thin each trajectory group's frames before --n-states picks from them")
    parser.add_argument('--limit', default=0, type=int,
                        help='Score only the first N proteins of the split (0 = all), for smoke tests')

    # sampler -- test/design.py's defaults
    parser.add_argument('--n-states', default=1, type=int,
                        help='Conformations averaged per flow step (Algorithm 1). 1 reproduces '
                             'upstream adaptive_sample')
    parser.add_argument('--num-step', default=8, type=int)
    parser.add_argument('--threshold', default=0.9, type=float)
    parser.add_argument('--temp', default=0.1, type=float)
    parser.add_argument('--noise', default=1.0, type=float)
    parser.add_argument('--no-argmax-final', dest='argmax_final', action='store_false')
    parser.add_argument('--verbose-sampling', action='store_true')

    parser.add_argument('--novel-subset', default=1, type=int,
                        help="Also summarise the subset of scored proteins whose MMseqs cluster "
                             "is absent from ADFLIP's train_clusters.txt (0 disables)")

    parser.add_argument('--wandb', default=1, type=int, help='Log to W&B (0 disables)')
    parser.add_argument('--wandb-key-file', default='./wandb_api.txt')
    parser.add_argument('--wandb-entity', default='jenslundsgaard7-uw-madison')
    parser.add_argument('--wandb-project', default='SheafProtein')
    parser.add_argument('--wandb-run-name', default='', type=str)
    return parser.parse_args()


_DESIGN_MODULE = None


def load_design_module():
    """Upstream's `test/design.py` (its model builder and checkpoint loader), imported once.

    Loaded by path rather than imported: ADFLIP's package is called `test`, which would shadow
    the standard library's `test` module on `import test.design`.

    Two registrations make checkpoints loadable, both of them about where pickle looks for the
    `Config` class a checkpoint's config was saved as:

    * ``sys.modules['adflip_design']`` -- train.py saves its config as this module's `Config`, so
      the name has to resolve at `torch.load` time. Registered *before* `exec_module` so the
      module object is findable even while it is still executing.
    * ``__main__.Config`` -- the *released* checkpoint pickles its config as `__main__.Config`
      (upstream saved it from a script that defined Config at top level, and design.py works
      because it is itself run as __main__). Here __main__ is whichever script is running, so
      without this `torch.load` fails with "Can't get attribute 'Config' on <module '__main__'>".
    """
    global _DESIGN_MODULE
    if _DESIGN_MODULE is not None:
        return _DESIGN_MODULE
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'adflip_design', osp.join(ADFLIP_ROOT, 'test', 'design.py'))
    design = importlib.util.module_from_spec(spec)
    sys.modules['adflip_design'] = design
    spec.loader.exec_module(design)
    sys.modules['__main__'].Config = design.Config
    _DESIGN_MODULE = design
    return design


def load_flow_model(ckpt_path, device):
    """``(ema_flow, config)`` for a checkpoint -- the released one, or one train.py saved."""
    return load_design_module().load_flow_model(ckpt_path, device)


def adflip_clusters():
    """``(pdb_chain -> cluster id, {train cluster ids})`` from ADFLIP's own clustering."""
    import pickle
    cluster_dir = osp.join(ADFLIP_ROOT, 'data', 'cluster')
    with open(osp.join(cluster_dir, 'pdb_chain_to_cluster.pkl'), 'rb') as handle:
        pdb_chain_to_cluster = pickle.load(handle)
    with open(osp.join(cluster_dir, 'train_clusters.txt')) as handle:
        train_clusters = {int(line) for line in handle if line.strip()}
    return pdb_chain_to_cluster, train_clusters


def in_adflip_training(protein_id, pdb_chain_to_cluster, train_clusters):
    """Is this protein's chain in a cluster ADFLIP trained on?

    ``None`` when the chain is absent from ADFLIP's map entirely -- that is "unknown", not
    "novel", and is counted separately so the novel subset is not inflated by lookup misses.
    """
    pdb_id, chain_id = relaxed_pdb.parse_structure_id(protein_id)
    key = '{}_{}'.format(pdb_id.lower(), chain_id) if chain_id else None
    cluster = pdb_chain_to_cluster.get(key) if key else None
    if cluster is None:
        # Fall back to any chain of the entry: the cluster is a sequence cluster, and a
        # deposited entry's chains are usually the same sequence.
        prefix = '{}_'.format(pdb_id.lower())
        matches = [c for k, c in pdb_chain_to_cluster.items() if k.startswith(prefix)]
        if not matches:
            return None
        return any(c in train_clusters for c in matches)
    return cluster in train_clusters


def frame_pdb_paths(store, protein_id, entry, out_dir):
    """Write this protein's selected frames out as single-chain backbone PDBs.

    One file per conformation with a distinct stem (`<id>_frame<nnnnn>`), which
    `multi_state.prepare_states` requires: ADFLIP's `sc_packing` names its repacked output after
    the input stem, so same-stem conformations would overwrite each other.
    """
    paths = []
    for record in store.frame_records(protein_id, entry):
        stem = '{}_frame{:05d}'.format(protein_id, record['frame_index'])
        path = osp.join(out_dir, stem + '.pdb')
        relaxed_pdb.write_record_pdb(record, path)
        paths.append(path)
    return paths


def relaxed_pdb_paths(records, protein_id, out_dir):
    """The one deposited/domain structure, written out the same way."""
    path = osp.join(out_dir, '{}_relaxed.pdb'.format(protein_id))
    relaxed_pdb.write_record_pdb(records[protein_id], path)
    return [path]


def score_protein(flow, pdb_paths, args):
    """Sample one protein and reduce ADFLIP's output to the 20-class metrics.

    :returns: ``(log_probs20, targets, design)`` -- ``(R, 20)`` renormalised log-probabilities,
        ``(R,)`` class indices, and the sampled design as a one-letter string over the same R
        residues. Residues whose native token is not one of the 20 standard residues are dropped
        from all three.
    """
    samples, mean_log_probs, true_tokens = multi_state.adaptive_sample_multi_state(
        flow, pdb_paths,
        num_step=args.num_step, argmax_final=args.argmax_final, temp=args.temp,
        noise=args.noise, threshold=args.threshold, verbose=args.verbose_sampling)

    samples = samples.view(-1).cpu()
    native = true_tokens.view(-1).cpu()

    classes = torch.tensor([TOKEN_TO_CLASS.get(int(t), -1) for t in native])
    keep = classes >= 0
    if not bool(keep.any()):
        raise ValueError('no standard residues to score')

    # Restrict to the 20 standard residues and renormalise, so perplexity and top-k are over the
    # same label set as PiFold/MapDiff/the sheaf model rather than ADFLIP's wider vocabulary.
    probs20 = mean_log_probs.exp().cpu()[:, STANDARD_TOKENS]
    probs20 = probs20 / probs20.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs20.dtype).tiny)
    log_probs20 = probs20.clamp_min(torch.finfo(probs20.dtype).tiny).log()

    design = ''.join(
        ALPHABET[TOKEN_TO_CLASS[int(token)]] if int(token) in TOKEN_TO_CLASS else 'X'
        for token, k in zip(samples, keep) if bool(k))

    return log_probs20[keep], classes[keep], design


def evaluate(flow, proteins, args, split_name):
    """Score every protein. `proteins` yields ``(protein_id, [conformation paths])``."""
    accumulator = stats_utils.ResidueMetrics(
        stats_utils.three_letter_codes(ALPHABET), val_name=split_name)
    per_protein, failures = [], {}

    for protein_id, pdb_paths in tqdm(proteins, desc='sampling'):
        try:
            log_probs, targets, design = score_protein(flow, pdb_paths, args)
        except Exception as exc:  # noqa: BLE001 -- one bad protein should not lose the run
            failures[protein_id] = '{}: {}'.format(type(exc).__name__, exc)
            print('>>> {} failed: {}'.format(protein_id, failures[protein_id]), file=sys.stderr)
            continue

        nll = F.cross_entropy(log_probs, targets).item()
        # Two recovery numbers, because for a sampler they genuinely differ:
        #   recovery_top1    -- argmax of the averaged final distribution. Use this one in the
        #                       comparison table; it is what PiFold/MapDiff/the sheaf model
        #                       report, so it is the only like-for-like column.
        #   recovery_sampled -- the sequence the sampler actually drew, which is ADFLIP's own
        #                       protocol and the number the paper's Table 3 reports. Use it to
        #                       check these runs against the published figures.
        sampled = torch.tensor([ALPHABET.index(c) if c in ALPHABET else -1 for c in design])
        recovery = (sampled == targets).float().mean().item()

        # `add_protein` derives its confusion matrix from the distribution argmax while
        # `pred_seqs` records the drawn design, so the two views of this protein are the two
        # recoveries above -- consistent with each other, not with a single sequence.
        accumulator.add_protein(log_probs, targets, pdb_id=protein_id, nll=nll, sequence=design)
        per_protein.append({
            'title': protein_id,
            'length': int(targets.shape[0]),
            'n_states': len(pdb_paths),
            'perplexity': stats_utils.perplexity_from_nll(torch.tensor(nll)),
            'recovery_sampled': recovery,
            'recovery_top1': stats_utils.top_k_acc(log_probs, targets, 1),
            'recovery_top5': stats_utils.top_k_acc(log_probs, targets, 5),
            'recovery_top10': stats_utils.top_k_acc(log_probs, targets, 10),
            'design': design,
        })

    return accumulator, per_protein, failures


def build_protein_list(args, ds_name, work_dir):
    """``(proteins, dropped)`` -- each protein with the conformation files to condition on."""
    train_ids, val_ids, test_ids = dataset_splits.get_splits(
        ds_name, index_csv=args.index_csv or None,
        val_fold=args.val_fold, test_fold=args.test_fold)
    ids = test_ids if args.split == 'test' else val_ids
    if args.limit:
        ids = ids[:args.limit]
    print('>>> {} split: {} proteins'.format(args.split, len(ids)))

    if args.structure_source == 'relaxed':
        if args.n_states != 1:
            raise ValueError('--structure-source relaxed has only one conformation; '
                             'use --structure-source frame for --n-states > 1')
        if ds_name == 'mdcath':
            records, failures = relaxed_pdb.load_cath_structures(ids, cath_dir=args.cath_dir)
        else:
            reference_seqs = (relaxed_pdb.reference_seqs_from_h5(args.traj_h5, ids)
                              if args.traj_h5 else None)
            records, failures = relaxed_pdb.load_relaxed_structures(
                ids, cache_dir=args.pdb_cache, reference_seqs=reference_seqs)
        proteins = [(pid, relaxed_pdb_paths(records, pid, work_dir))
                    for pid in ids if pid in records]
        return proteins, failures

    if not args.traj_h5:
        raise ValueError('--structure-source frame needs --traj-h5')
    store = traj_frames.TrajectoryStore(args.traj_h5)
    entries, failures = store.index(
        ids, ds_name, stride=args.frame_stride, max_frames=args.n_states,
        max_length=args.max_length)

    proteins = []
    for protein_id, entry in entries.items():
        paths = frame_pdb_paths(store, protein_id, entry, work_dir)
        if len(paths) < args.n_states:
            # `select_frames` caps but cannot invent frames: a short trajectory contributes all
            # it has. Recorded rather than dropped, and `n_states` is kept per protein.
            print('>>> {}: {} conformations available, {} requested'.format(
                protein_id, len(paths), args.n_states))
        proteins.append((protein_id, paths))
    store.close()
    return proteins, failures


PATH_ARGS = ('ckpt', 'traj_h5', 'index_csv', 'pdb_cache', 'cath_dir', 'res_dir', 'keep_pdbs',
             'wandb_key_file')


def resolve_paths(args, ds_name, origin):
    """Make every path argument absolute against the cwd the user launched from.

    `main` chdirs into the checkout before sampling, because upstream reads
    `PIPPack/model_weights/...` and writes `results/new_samples/` relative to cwd. Anything the
    user named on the command line has to be pinned to the original directory first, including
    the index csv `dataset_splits` would otherwise look up under its own relative default name.
    """
    for name in PATH_ARGS:
        value = getattr(args, name)
        if value:
            setattr(args, name, osp.abspath(osp.join(origin, value)))
    if not args.index_csv:
        args.index_csv = osp.abspath(
            osp.join(origin, dataset_splits.default_index_csv(ds_name)))
    return args


def main():
    args = create_parser()
    device = torch.device(args.device)
    ds_name = DATA_NAMES[args.data_name]

    origin = os.getcwd()
    args = resolve_paths(args, ds_name, origin)
    # Upstream assumes it is run from the checkout root (README: `export PYTHONPATH=$PWD`), and
    # resolves PIPPack's weights and its own sample-save directory relative to cwd.
    os.chdir(ADFLIP_ROOT)
    print('>>> working directory: {} (upstream resolves PIPPack weights relative to it)'.format(
        ADFLIP_ROOT))

    ema_flow, _config = load_flow_model(args.ckpt, device)
    flow = ema_flow.ema_model
    num_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print('>>> ADFLIP loaded: {} trainable parameters, sidechain packing {}'.format(
        num_params, 'on' if hasattr(flow, 'sc') else 'OFF'))
    if not hasattr(flow, 'sc'):
        raise RuntimeError(
            'the flow model was built without side-chain packing; the adaptive sampler needs it '
            'to build context for residues it has already chosen')

    work_dir = args.keep_pdbs or tempfile.mkdtemp(prefix='adflip_eval_')
    os.makedirs(work_dir, exist_ok=True)
    # Keep the per-step repacked side-chain PDBs (one per conformation per flow step, for every
    # protein) inside the work dir so they are cleaned up with it, instead of accumulating under
    # the checkout's `results/new_samples/`.
    flow.sample_save_path = osp.join(work_dir, 'samples')
    os.makedirs(flow.sample_save_path, exist_ok=True)
    try:
        proteins, dropped = build_protein_list(args, ds_name, work_dir)
        if not proteins:
            raise RuntimeError('no usable {} proteins for {}'.format(args.split, ds_name))
        print('>>> scoring {} {} {} proteins with {} state(s) each'.format(
            len(proteins), ds_name, args.split, args.n_states))

        split_name = 'test' if args.split == 'test' else 'val'
        accumulator, per_protein, failures = evaluate(flow, proteins, args, split_name)
    finally:
        if not args.keep_pdbs:
            shutil.rmtree(work_dir, ignore_errors=True)

    if not per_protein:
        raise RuntimeError('every protein failed; see the errors above')

    metric_keys = stats_utils.METRIC_KEYS + ('recovery_sampled',)
    summary = stats_utils.summarize_per_protein(per_protein, metric_keys)
    summary['n_proteins'] = len(per_protein)

    novel = {}
    if args.novel_subset:
        pdb_chain_to_cluster, train_clusters = adflip_clusters()
        flags = {row['title']: in_adflip_training(row['title'], pdb_chain_to_cluster, train_clusters)
                 for row in per_protein}
        novel_rows = [row for row in per_protein if flags[row['title']] is False]
        novel = {
            'n_in_adflip_training': sum(1 for v in flags.values() if v is True),
            'n_novel': sum(1 for v in flags.values() if v is False),
            'n_unknown': sum(1 for v in flags.values() if v is None),
            'novel_ids': [row['title'] for row in novel_rows],
        }
        if novel_rows:
            novel['summary'] = stats_utils.summarize_per_protein(novel_rows, metric_keys)
            novel['summary']['n_proteins'] = len(novel_rows)
        print('>>> ADFLIP training overlap: {} of {} scored proteins are in a cluster it '
              'trained on ({} novel, {} not in its cluster map)'.format(
                  novel['n_in_adflip_training'], len(per_protein), novel['n_novel'],
                  novel['n_unknown']))

    print(json.dumps(summary, indent=2))
    if novel.get('summary'):
        print('>>> novel-subset summary:')
        print(json.dumps(novel['summary'], indent=2))

    os.makedirs(args.res_dir, exist_ok=True)
    with open(osp.join(args.res_dir, 'summary.json'), 'w') as handle:
        json.dump({'args': vars(args), 'summary': summary, 'novel_subset': novel,
                   'dropped': dropped, 'failed': failures}, handle, indent=2)
    with open(osp.join(args.res_dir, 'per_protein.json'), 'w') as handle:
        json.dump(per_protein, handle, indent=2)

    if args.wandb:
        run = stats_utils.init_wandb(
            args.wandb_key_file, args.wandb_entity, args.wandb_project,
            args.wandb_run_name or 'Eval_ADFLIP_{}_{}_{}states'.format(
                args.weights_name, dataset_splits.dataset_tag(ds_name, args.traj_h5),
                args.n_states),
            config=dict(vars(args), n_eval_proteins=len(per_protein), params=num_params))
        run.summary['{}_split_ids'.format(split_name)] = [row['title'] for row in per_protein]
        run.summary['dropped_proteins'] = dropped
        run.summary['failed_proteins'] = failures
        if novel:
            run.summary['adflip_training_overlap'] = {k: v for k, v in novel.items()
                                                      if k != 'novel_ids'}
        run.log({'params': num_params, **summary})
        if novel.get('summary'):
            run.log({'novel_' + k: v for k, v in novel['summary'].items()})
        run.log(accumulator.to_log_dict(
            cm_title='{} Amino Acid Confusion Matrix'.format(split_name.capitalize())))
        stats_utils.log_per_protein_table(per_protein, metric_keys)
        run.finish()


if __name__ == '__main__':
    main()
