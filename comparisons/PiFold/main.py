"""Overlaid onto a pristine PiFold checkout (see run_pifold.sh).

Same train/valid/test loop as upstream, plus the two things this comparison needs:

* **Weights & Biases logging** of per-epoch train/valid loss and perplexity, and of the
  held-out split's perplexity / median recovery / scRMSD -- one run per dataset, so PiFold's
  numbers land next to the sheaf model's in the same project.
* **scRMSD**: every held-out protein's designed sequence is folded with ESMFold
  (`lib/scrmsd.py`) and Kabsch-RMSD'd against that protein's ground-truth *relaxed*
  (deposited) backbone. No alignment step is needed: `API/relaxed_dataset.py` feeds PiFold
  those deposited structures directly, so the design and the reference are the same residues
  in the same order.

ATLAS and mdCATH are trained and evaluated separately -- one run each (`--data_name ATLAS`
or `MDCATH`) -- matching scripts/train_residue_classifier.py's `--ds-name`. For mdCATH the
held-out split is the `validation` rows only.
"""
import logging
import json
import sys
import numpy as np
import torch
import os.path as osp

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

from methods import ProDesign
from methods.utils import cuda
from API import Recorder
from utils import *

# lib/scrmsd.py is copied in flat next to this file (see README.md and run_pifold.sh) --
# fall back to walking up to a lib/ directory for local/dev runs from inside the source tree.
try:
    from scrmsd import load_esmfold, evaluate_scrmsd
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = osp.join(osp.dirname(osp.abspath(__file__)), _up, 'lib')
        if osp.isdir(_cand):
            sys.path.insert(0, osp.abspath(_cand))
            break
    from scrmsd import load_esmfold, evaluate_scrmsd

# PiFold's fixed label order (API.featurizer.featurize_GTrans).
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
# lib/scrmsd.py expects the atom axis in scripts/load_dynamics.py's BACKBONE_ATOMS order.
BACKBONE_KEYS = ('CA', 'N', 'C', 'O')


class Exp:
    def __init__(self, args, show_params=True):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self.run = self._init_wandb()
        self._preparation()
        if show_params:
            print_log(output_namespace(self.args))

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU:', device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_wandb(self):
        if not self.args.wandb:
            return None
        import wandb
        with open(self.args.wandb_key_file) as f:
            wandb.login(key=f.read().strip())
        return wandb.init(
            entity=self.args.wandb_entity,
            project=self.args.wandb_project,
            name=self.args.wandb_run_name or '{}_PiFold'.format(self.args.data_name),
            config=self.args.__dict__,
        )

    def _log(self, metrics):
        if self.run is not None:
            self.run.log(metrics)

    def _preparation(self):
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')

        self._get_data()
        self._build_method()

        # ESMFold is only needed for the final scRMSD pass; load it lazily so a --scrmsd 0
        # run (or a crash during training) never pays for the download.
        self.esmfold = None

    def _build_method(self):
        steps_per_epoch = 1000
        if self.args.method == 'ProDesign':
            self.method = ProDesign(self.args, self.device, steps_per_epoch)

    def _get_data(self):
        self.train_loader, self.valid_loader, self.test_loader = get_dataset(self.config)
        self._log_split_manifest()

    def _log_split_manifest(self):
        """Record the exact protein ids in each split.

        The split itself comes from `lib/dataset_splits.py`, so it is identical to
        MapDiff's and DynamicMPNN's by construction -- but each model additionally drops
        whatever it cannot featurize, and those drops are what make two held-out sets
        silently diverge. Logging the surviving ids per run makes that diff checkable
        instead of assumed.
        """
        datasets = {'train': self.train_loader.dataset, 'valid': self.valid_loader.dataset,
                    'test': self.test_loader.dataset}
        if not all(hasattr(ds, 'split_ids') for ds in datasets.values()):
            return  # CATH/TS: upstream's own splits, nothing to reconcile

        manifest = {name: ds.split_ids(name) for name, ds in datasets.items()}
        for name, ids in manifest.items():
            print_log('{} split: {} proteins'.format(name, len(ids)))
        print_log('held-out (test) split ids: {}'.format(manifest['test']))

        if self.run is not None:
            self.run.summary['split_counts'] = {name: len(ids) for name, ids in manifest.items()}
            self.run.summary['test_split_ids'] = manifest['test']
            self.run.summary['dropped_proteins'] = getattr(datasets['test'], 'dropped', {})

    def train(self):
        recorder = Recorder(self.args.patience, verbose=True)
        for epoch in range(self.args.epoch):
            train_loss, train_perplexity = self.method.train_one_epoch(self.train_loader)
            self._log({'train_loss': train_loss, 'train_perplexity': train_perplexity, 'epoch': epoch})

            if epoch % self.args.log_step == 0:
                with torch.no_grad():
                    valid_loss, valid_perplexity = self.valid()
                    self._log({'valid_loss': valid_loss, 'valid_perplexity': valid_perplexity, 'epoch': epoch})

                    # self._save(name=str(epoch))
                    self.test(epoch=epoch)

                print_log('Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Train Perp: {3:.4f} Valid Loss: {4:.4f} Valid Perp: {5:.4f}\n'.format(epoch + 1, len(self.train_loader), train_loss, train_perplexity, valid_loss, valid_perplexity))

                recorder(valid_loss, self.method.model, self.path)
                if recorder.early_stop:
                    print("Early stopping")
                    logging.info("Early stopping")
                    break

        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self.method.model.load_state_dict(torch.load(best_model_path))

    def valid(self):
        valid_loss, valid_perplexity = self.method.valid_one_epoch(self.valid_loader)

        print_log('Valid Perp: {0:.4f}'.format(valid_perplexity))

        return valid_loss, valid_perplexity

    def test(self, epoch=None, with_scrmsd=False):
        """Held-out-split metrics. `with_scrmsd` is off during training (ESMFold is far too
        slow to fold the whole split every epoch) and on for the final pass."""
        test_perplexity, test_recovery, test_subcat_recovery = self.method.test_one_epoch(self.test_loader)
        print_log('Test Perp: {0:.4f}, Test Rec: {1:.4f}\n'.format(test_perplexity, test_recovery))

        metrics = {'test_perplexity': test_perplexity, 'test_median_recovery': test_recovery,
                   'test_mean_recovery': float(self.method.mean_recovery),
                   'test_std_recovery': float(self.method.std_recovery)}
        if epoch is not None:
            metrics['epoch'] = epoch

        for cat in test_subcat_recovery.keys():
            print_log('Category {0} Rec: {1:.4f}\n'.format(cat, test_subcat_recovery[cat]))

        if with_scrmsd and self.args.scrmsd:
            scrmsd = self.compute_scrmsd(self.test_loader.dataset, self.test_loader.featurizer)
            if scrmsd.numel():
                print_log('Test scRMSD: {0:.4f} +/- {1:.4f}\n'.format(scrmsd.mean().item(), scrmsd.std().item()))
                metrics['test_scrmsd_mean'] = scrmsd.mean().item()
                metrics['test_scrmsd_std'] = scrmsd.std().item()

        self._log(metrics)
        return test_perplexity, test_recovery

    @torch.no_grad()
    def compute_scrmsd(self, dataset, featurizer):
        """Fold every held-out design with ESMFold and Kabsch-RMSD it against that protein's
        relaxed (deposited) backbone.

        Designs are produced one protein at a time (as `ProDesign._cal_recovery` does, so the
        residue order matches the dataset entry exactly), then folded in small batches --
        `lib.scrmsd.fold_sequences` folds each sequence on its own anyway, so the batch size
        only trades peak memory for fewer calls.
        """
        if self.esmfold is None:
            self.esmfold = load_esmfold(device=self.device)
        esmfold_tokenizer, esmfold_model = self.esmfold

        self.method.model.eval()
        seqs, references = [], []
        for entry in tqdm(dataset, desc='designing held-out split for scRMSD'):
            protein = featurizer([entry])
            X, S, score, mask, lengths = cuda(protein, device=self.device)
            X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = \
                self.method.model._get_features(S, score, X=X, mask=mask)
            log_probs = self.method.model(h_V, h_E, E_idx, batch_id)
            pred = log_probs.argmax(dim=1).cpu()

            reference = np.stack([entry[atom] for atom in BACKBONE_KEYS], axis=1)  # (R, 4, 3)
            if pred.shape[0] != reference.shape[0]:
                # The featurizer drops residues with non-finite coordinates; our relaxed
                # records have none, but skip rather than silently mis-pair if that changes.
                print('skip scRMSD for {}: {} designed residues vs {} reference residues'.format(
                    entry['title'], pred.shape[0], reference.shape[0]))
                continue
            seqs.append(''.join(ALPHABET[i] for i in pred.tolist()))
            references.append(torch.from_numpy(reference).float())

        values = []
        step = max(1, self.args.scrmsd_batch_size)
        for start in tqdm(range(0, len(seqs), step), desc='folding designs (ESMFold)'):
            values.extend(evaluate_scrmsd(
                seqs[start:start + step], references[start:start + step],
                esmfold_tokenizer, esmfold_model, device=self.device).tolist())
        return torch.tensor(values)


if __name__ == '__main__':
    from parser import create_parser
    args = create_parser()
    config = args.__dict__

    print(config)

    exp = Exp(args)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train()

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    test_perp, test_rec = exp.test(with_scrmsd=True)

    if exp.run is not None:
        exp.run.finish()
