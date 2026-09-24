"""Single trainer driving both of MapDiff's training stages against ATLAS:

  1. `fit_prior()`  -- mask-prior IPA pretraining (was `mask_ipa_pretrain.py` +
     `trainer/mask_ipa_trainer.py`).
  2. `train()` / `test()` -- denoising diffusion training (was `main.py` +
     `trainer/trainer.py`), fine-tuning the same prior model in place.

Both stages log to a single Weights & Biases run (no more comet_ml). Validation
(`train()`) and `test()` report MapDiff's own metrics (median/mean recovery,
BLOSUM NSSR) *and*, under the same keys, the full metric set
`scripts/train_residue_classifier.py`'s `run_val` logs -- per-protein top-1/5/10
recovery, perplexity, per-residue precision/recall/f1, an amino-acid confusion
matrix and a `pred_seqs` table of every design keyed by `<pdb>_<chain>`. That
set is built by `lib.stats_utils.ResidueMetrics`, which is what keeps the three
comparison models and the sheaf model reporting the same thing.
"""
import copy
import datetime
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from pathlib import Path
from prettytable import PrettyTable
from sklearn.metrics import f1_score
from tqdm import tqdm

from evaluator import Evaluator
from utils import inf_iterator, enable_dropout, cal_stats_metric

# lib/stats_utils.py is copied in flat next to this file (see ../README.md and
# train.sh) -- fall back to walking up to a lib/ directory for local/dev runs
# from inside the source tree.
try:
    from stats_utils import ResidueMetrics, three_letter_codes
except ImportError:
    import sys
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    from stats_utils import ResidueMetrics, three_letter_codes

# Fixed one-letter amino-acid order MapDiff's 20-dim one-hot (`data.x[:, :20]`)
# is built in -- see (upstream) data/generate_graph_cath.py's
# `amino_acids_type` / evaluator.py's `blosum_aa_order`, both of which use
# this exact order.
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

class MapDiffTrainer:
    def __init__(
            self,
            config,
            prior_model,
            prior_optimizer,
            mask_train_dataloader,
            diffusion_model,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            device,
            output_dir,
            wandb_run,
            prior_scheduler=None,
            scheduler=None,
            train_batch_size=512,
            train_num_steps=200000,
            save_and_sample_every=100,
            num_samples=25,
            ensemble_num=50,
            ddim_steps=50,
            sample_method='ddim',
            early_stopping_patience=None,
            early_stopping_min_delta=0.0,
    ):
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.wandb_run = wandb_run
        self.evaluator = Evaluator()

        Path(self.output_dir + '/model/').mkdir(parents=True, exist_ok=True)

        # ---- stage 1: mask-prior IPA pretraining ----
        self.prior_model = prior_model.to(self.device)
        self.prior_optimizer = prior_optimizer
        self.prior_scheduler = prior_scheduler
        self.mask_train_dataloader = mask_train_dataloader
        self.prior_epoch = 0
        self.prior_step = 0
        self.prior_train_table = PrettyTable(["# Epoch", "# Step", "Train_loss"])

        # ---- stage 2: denoising diffusion training ----
        self.model = diffusion_model.to(self.device)
        self.num_samples = num_samples
        self.ensemble_num = ensemble_num
        self.ddim_steps = ddim_steps
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.sample_method = sample_method

        self.train_dataloader = train_dataloader
        self.iter_one_epoch = len(train_dataloader)
        self.train_iterator = inf_iterator(train_dataloader)
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_step = 0
        self.best_val_epoch = 0
        self.step = 0
        self.epoch = 0
        # `best_val_perplexity` is the per-protein mean val perplexity (ResidueMetrics'
        # `val_perp_mean`); `best_val_recovery` is just the median recovery at that epoch.
        self.best_val_recovery, self.best_val_perplexity = 0, float('inf')
        self.best_model = None
        # Early stopping on val perplexity -- the protocol every comparison model shares (see
        # comparisons/DynamicMPNN/train.py's EarlyStopping) -- with patience counted in epochs
        # (validation only runs every `save_and_sample_every` epochs, so patience is effectively
        # rounded up to that). `None` disables it; the best val epoch is still what test scores.
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.train_table = PrettyTable(["# Epoch", "# Step", "Train_loss"])
        self.val_table = PrettyTable(["# Epoch", "# Step", "Recovery", "Perplexity"])
        self.test_table = PrettyTable(["# Epoch", "# Step", "Recovery", "Perplexity"])

        # Only used to head the LaTeX summary row `ResidueMetrics.summary_line` prints, the
        # same one scripts/train_residue_classifier.py's `run_val` ends with.
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @staticmethod
    def _batch_names(g_batch):
        """Per-graph protein ids for a collated `Batch`, in graph order.

        `dataloader.large_dataset.Cath` attaches `name` to every graph, so
        `Batch.from_data_list` hands it back as a list -- but a batch of one collapses to a
        bare string, and graphs featurized before `name` existed have none at all. Normalise
        both here so the `pred_seqs` table always has a label per design.
        """
        names = getattr(g_batch, 'name', None)
        num_graphs = int(g_batch.batch.max().item()) + 1
        if names is None:
            return ['unknown'] * num_graphs
        if isinstance(names, str):
            names = [names]
        return list(names)

    # ------------------------------------------------------------------
    # Stage 1: mask-prior IPA pretraining
    # ------------------------------------------------------------------

    def _prior_train_epoch(self):
        self.prior_model.train()
        self.prior_epoch += 1
        all_logits, all_labels, all_index = [], [], []
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        for idx, data in enumerate(tqdm(self.mask_train_dataloader, desc=f"[Pretrain] Epoch {self.prior_epoch}")):
            x, x_pos, x_pad, x_mask, aa_label = (t.to(self.device) for t in data)
            logits = self.prior_model(x, x_pos, x_mask, x_pad)
            loss = loss_fn(logits[x_mask >= 1], aa_label[x_mask >= 1]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.prior_model.parameters(), self.config.mask_train.clip_grad_norm)

            self.prior_optimizer.step()
            if self.prior_scheduler:
                self.prior_scheduler.step()
            self.prior_optimizer.zero_grad()
            self.prior_step += 1

            all_logits.append(logits.view(-1, 20).detach().cpu())
            all_labels.append(aa_label.view(-1).detach().cpu())
            all_index.append(x_mask.view(-1).detach().cpu())

            if self.wandb_run and idx % 10 == 0:
                self.wandb_run.log({'pretrain_loss': loss.item(), 'pretrain_step': self.prior_step,
                                     'pretrain_epoch': self.prior_epoch})

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_index = torch.cat(all_index, dim=0)
        return self._prior_log_metrics(all_labels, all_logits, all_index, loss_fn)

    def _prior_log_metrics(self, sl_labels, sl_predictions, sl_index, loss_fn):
        all_loss = loss_fn(sl_predictions[sl_index >= 1], sl_labels[sl_index >= 1]).mean()
        mask_loss = loss_fn(sl_predictions[sl_index == 1], sl_labels[sl_index == 1]).mean()
        replace_loss = loss_fn(sl_predictions[sl_index == 2], sl_labels[sl_index == 2]).mean()
        keep_loss = loss_fn(sl_predictions[sl_index == 3], sl_labels[sl_index == 3]).mean()

        pred_labels = np.argmax(sl_predictions.numpy(), axis=-1)
        labels = sl_labels.numpy()
        index = sl_index.numpy()

        metrics = {}
        for name, mask_val in (('mask', 1), ('replace', 2), ('keep', 3)):
            metrics[f'macro_{name}_f1'] = f1_score(labels[index == mask_val], pred_labels[index == mask_val],
                                                    average='macro')
            metrics[f'micro_{name}_f1'] = f1_score(labels[index == mask_val], pred_labels[index == mask_val],
                                                    average='micro')

        print(f"[Pretrain] Epoch {self.prior_epoch}: all_loss={all_loss.item():.4f} mask_loss={mask_loss.item():.4f} "
              f"replace_loss={replace_loss.item():.4f} keep_loss={keep_loss.item():.4f}")
        return all_loss, metrics

    def _save_prior(self, mode='last'):
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        data = {
            'config': config_dict,
            'epoch': self.prior_epoch,
            'model': self.prior_model.state_dict(),
            'opt': self.prior_optimizer.state_dict(),
        }
        save_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(self.output_dir, 'model',
                             f'{self.config.experiment.name}_prior_{mode}_{self.prior_epoch}_epochs_{save_time}.pt')
        torch.save(data, path)
        return path

    def fit_prior(self):
        """Stage 1: mask-prior IPA pretraining. Trains `self.prior_model` in
        place -- since `self.model` (the diffusion model) already holds a
        reference to the same module, stage 2 automatically sees these
        trained weights with no checkpoint round-trip needed."""
        epochs = self.config.mask_train.train_epochs
        save_epochs = self.config.mask_train.save_epochs
        for _ in range(epochs):
            train_loss, metrics = self._prior_train_epoch()
            self.prior_train_table.add_row([self.prior_epoch, self.prior_step, train_loss.item()])
            if self.wandb_run:
                self.wandb_run.log({f'pretrain_{k}': v for k, v in metrics.items()} | {'pretrain_epoch': self.prior_epoch})
            if self.prior_epoch % save_epochs == 0 and self.prior_epoch > 10:
                self._save_prior(mode='curr')
            torch.cuda.empty_cache()
        self._save_prior(mode='last')
        print("Stage 1 (mask-prior IPA pretraining) complete")

    # ------------------------------------------------------------------
    # Stage 2: denoising diffusion training
    # ------------------------------------------------------------------

    def save(self, save_epochs, save_steps, mode='best'):
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        state = self.best_model.state_dict() if mode == 'best' else self.model.state_dict()
        data = {
            'config': config_dict,
            'step': save_steps,
            'epoch': save_epochs,
            'model': state,
            'opt': self.optimizer.state_dict(),
        }
        save_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(data, os.path.join(self.output_dir, 'model',
                                      f'{self.config.experiment.name}_{mode}_{save_epochs}_epochs_{save_steps}_steps_{save_time}.pt'))

    def save_table_results(self):
        with open(os.path.join(self.output_dir, 'pretrain_markdowntable.txt'), 'w') as f:
            f.write(self.prior_train_table.get_string())
        with open(os.path.join(self.output_dir, 'train_markdowntable.txt'), 'w') as f:
            f.write(self.train_table.get_string())
        with open(os.path.join(self.output_dir, 'val_markdowntable.txt'), 'w') as f:
            f.write(self.val_table.get_string())
        with open(os.path.join(self.output_dir, 'test_markdowntable.txt'), 'w') as f:
            f.write(self.test_table.get_string())

    def _run_validation(self):
        self.model.eval()
        enable_dropout(self.model)
        with torch.no_grad():
            all_logits = torch.tensor([])
            all_seq = torch.tensor([])
            recovery = []
            # Everything scripts/train_residue_classifier.py's `run_val` logs, under the same
            # keys -- per-protein top-1/5/10, perplexity, per-residue precision/recall/f1, the
            # confusion matrix and the `pred_seqs` table (see lib/stats_utils.py).
            metrics = ResidueMetrics(three_letter_codes(AMINO_ACIDS), val_name='val')

            for g_batch, ipa_batch in tqdm(self.val_dataloader, desc=f"Epoch {self.epoch} [Val]", leave=False):
                g_batch = g_batch.to(self.device)
                ipa_batch = ipa_batch.to(self.device) if ipa_batch is not None else None
                ens_logits = []
                if self.sample_method == 'ddim':
                    for _ in range(self.ensemble_num):
                        logits, sample_graph = self.model.mc_ddim_sample(g_batch, ipa_batch, diverse=True,
                                                                          step=self.ddim_steps)
                        ens_logits.append(logits)
                ens_logits_tensor = torch.stack(ens_logits)
                batch_logits = ens_logits_tensor.mean(dim=0).cpu()
                all_logits = torch.cat([all_logits, batch_logits])
                all_seq = torch.cat([all_seq, g_batch.x.cpu()])

                batch_idx = g_batch.batch.cpu().numpy()
                batch_names = self._batch_names(g_batch)
                for i in range(batch_idx.max() + 1):
                    idx = np.where(batch_idx == i)
                    sample_logits = batch_logits[idx].argmax(dim=1)
                    sample_seq = g_batch.x.cpu()[idx].argmax(dim=1)
                    recovery.append(self.evaluator.cal_recovery(sample_logits, sample_seq))
                    metrics.add_protein(batch_logits[idx], sample_seq, pdb_id=batch_names[i],
                                        sequence=''.join(AMINO_ACIDS[j] for j in sample_logits.tolist()))

            mean_recovery, median_recovery = cal_stats_metric(recovery)
            full_recovery = ((all_logits.argmax(dim=1) == all_seq.argmax(dim=1)).sum() / all_seq.shape[0]).item()
            perplexity = self.evaluator.cal_perplexity(all_logits, all_seq)

            print(f'Val median recovery rate (step: {self.step}) is {median_recovery}')
            print(f'Val perplexity (step: {self.step}): {perplexity}')
            self.val_table.add_row([self.epoch, self.step, median_recovery, perplexity])

            print(metrics.summary_line(self.num_params))

            if self.wandb_run:
                self.wandb_run.log(metrics.to_log_dict(
                    cm_title='Validation Amino Acid Confusion Matrix', epoch=self.epoch) | {
                    'val_full_recovery': full_recovery, 'val_perplexity': perplexity,
                    'val_median_recovery': median_recovery, 'val_mean_recovery': mean_recovery,
                })

            # Selected on the per-protein mean perplexity -- `val_perp_mean` in the dict logged
            # above, the same number DynamicMPNN and PiFold select on -- not on `perplexity`,
            # which pools every residue of the split into one cross-entropy.
            val_perp_mean = torch.exp(torch.tensor(metrics.losses)).mean().item()
            if val_perp_mean < self.best_val_perplexity - self.early_stopping_min_delta:
                self.best_model = copy.deepcopy(self.model)
                self.best_val_step = self.step
                self.best_val_epoch = self.epoch
                self.best_val_recovery = median_recovery
                self.best_val_perplexity = val_perp_mean

    def train(self):
        """Stage 2: denoising diffusion training, seeded by (and jointly
        fine-tuning) the mask-prior IPA model trained in `fit_prior()`."""
        epoch_total_loss = 0
        with tqdm(initial=self.step, total=self.train_num_steps, desc="[Diffusion]") as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                g_batch, ipa_batch = next(self.train_iterator)
                g_batch = g_batch.to(self.device)
                ipa_batch = ipa_batch.to(self.device) if ipa_batch is not None else None
                base_loss, mask_loss = self.model(g_batch, ipa_batch)
                loss = base_loss + mask_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                self.step += 1
                epoch_total_loss += loss.item()

                if self.wandb_run:
                    self.wandb_run.log({'train_base_loss': base_loss.item(), 'train_mask_loss': mask_loss.item(),
                                         'train_loss': loss.item(), 'step': self.step, 'epoch': self.epoch})

                if self.step % self.iter_one_epoch == 0 and self.step != 0:
                    self.epoch += 1
                    self.train_table.add_row([self.epoch, self.step, epoch_total_loss / self.iter_one_epoch])
                    epoch_total_loss = 0
                    torch.cuda.empty_cache()

                pbar.update(1)
                if self.step != 0 and self.step % (self.save_and_sample_every * self.iter_one_epoch) == 0:
                    self._run_validation()
                    if self._should_stop_early():
                        break

        print('Stage 2 (diffusion) training complete')
        if self.wandb_run:
            self.wandb_run.log({'best_val_median_recovery': self.best_val_recovery,
                                 'best_val_perp_mean': self.best_val_perplexity,
                                 'best_val_epoch': self.best_val_epoch})
        self.save(self.best_val_epoch, self.best_val_step, mode='best')
        self.save(self.epoch, self.step, mode='last')

    def _should_stop_early(self):
        if self.early_stopping_patience is None:
            return False
        epochs_since_best = self.epoch - self.best_val_epoch
        if epochs_since_best < self.early_stopping_patience:
            return False
        print(f'Early stopping at epoch {self.epoch}: val perplexity has not improved by more than '
              f'{self.early_stopping_min_delta} in {epochs_since_best} epochs '
              f'(best {self.best_val_perplexity:.4f} at epoch {self.best_val_epoch})')
        if self.wandb_run:
            self.wandb_run.log({'early_stopped_epoch': self.epoch, 'epoch': self.epoch})
        return True

    def test(self):
        model = self.best_model if self.best_model is not None else self.model
        model.eval()
        enable_dropout(model)
        with torch.no_grad():
            print('Testing best model')
            all_logits = torch.tensor([])
            all_seq = torch.tensor([])
            recovery = []
            nssr42, nssr62, nssr80, nssr90 = [], [], [], []
            metrics = ResidueMetrics(three_letter_codes(AMINO_ACIDS), val_name='test')

            for g_batch, ipa_batch in tqdm(self.test_dataloader, desc="[Test]"):
                g_batch = g_batch.to(self.device)
                ipa_batch = ipa_batch.to(self.device) if ipa_batch is not None else None
                ens_logits = []
                if self.sample_method == 'ddim':
                    for _ in range(self.ensemble_num):
                        logits, sample_graph = model.mc_ddim_sample(g_batch, ipa_batch, diverse=True,
                                                                     step=self.ddim_steps)
                        ens_logits.append(logits)
                ens_logits_tensor = torch.stack(ens_logits)
                batch_logits = ens_logits_tensor.mean(dim=0).cpu()
                all_logits = torch.cat([all_logits, batch_logits])
                all_seq = torch.cat([all_seq, g_batch.x.cpu()])

                batch_idx = g_batch.batch.cpu().numpy()
                batch_names = self._batch_names(g_batch)
                for i in range(batch_idx.max() + 1):
                    idx = np.where(batch_idx == i)
                    sample_logits = batch_logits[idx].argmax(dim=1)
                    sample_seq = g_batch.x.cpu()[idx].argmax(dim=1)
                    sample_nssr42, sample_nssr62, sample_nssr80, sample_nssr90 = self.evaluator.cal_all_blosum_nssr(
                        sample_logits, sample_seq)
                    nssr42.append(sample_nssr42)
                    nssr62.append(sample_nssr62)
                    nssr80.append(sample_nssr80)
                    nssr90.append(sample_nssr90)
                    recovery.append(self.evaluator.cal_recovery(sample_logits, sample_seq))
                    metrics.add_protein(batch_logits[idx], sample_seq, pdb_id=batch_names[i],
                                        sequence=''.join(AMINO_ACIDS[j] for j in sample_logits.tolist()))

            test_mean_recovery, test_median_recovery = cal_stats_metric(recovery)
            test_mean_nssr42, test_median_nssr42 = cal_stats_metric(nssr42)
            test_mean_nssr62, test_median_nssr62 = cal_stats_metric(nssr62)
            test_mean_nssr80, test_median_nssr80 = cal_stats_metric(nssr80)
            test_mean_nssr90, test_median_nssr90 = cal_stats_metric(nssr90)

            test_recovery = ((all_logits.argmax(dim=1) == all_seq.argmax(dim=1)).sum() / all_seq.shape[0]).item()
            test_perplexity = self.evaluator.cal_perplexity(all_logits, all_seq)

            print(f'test median recovery rate with best model (step: {self.best_val_step}) is {test_median_recovery}')
            print(f'test perplexity with the best model (step: {self.best_val_step}) is: {test_perplexity}')
            self.test_table.add_row([self.best_val_epoch, self.best_val_step, test_median_recovery,
                                      test_perplexity])

            print(metrics.summary_line(self.num_params))

            if self.wandb_run:
                self.wandb_run.log(metrics.to_log_dict(
                    cm_title='Test Amino Acid Confusion Matrix', epoch=self.best_val_epoch) | {
                    'test_full_recovery_with_best_model': test_recovery,
                    'test_perplexity_with_best_model': test_perplexity,
                    'test_median_recovery_with_best_model': test_median_recovery,
                    'test_mean_recovery_with_best_model': test_mean_recovery,
                    'test_median_nssr42_with_best_model': test_median_nssr42,
                    'test_median_nssr62_with_best_model': test_median_nssr62,
                    'test_median_nssr80_with_best_model': test_median_nssr80,
                    'test_median_nssr90_with_best_model': test_median_nssr90,
                })
