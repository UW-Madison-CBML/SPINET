"""Train MapDiff on ATLAS end-to-end: mask-prior IPA pretraining followed by
denoising diffusion training, in a single process and a single Weights &
Biases run (see trainer.py's `MapDiffTrainer`).

Replaces the old two-script pipeline (`mask_ipa_pretrain.py` + `main.py`,
each with its own comet_ml/wandb run) -- pass `prior_model.path=<ckpt>` to
skip stage 1 and load an already-trained mask-prior IPA checkpoint instead.

Usage:
    python train.py --config-name=train dataset=atlas wandb.use=True
"""
import os
import sys

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from dataloader.collator import CollatorDiff, CollatorIPAPretrain
from dataloader.large_dataset import Cath
from dataloader.pyg_inspector_compat import patch_inspector_distribute
from dataloader.pyg_safe_globals import allow_pyg_data_pickles
from model.egnn_pytorch.egnn_net import EGNN_NET
from model.ipa.ipa_net import IPANetPredictor
from model.prior_diff import Prior_Diff
from utils import set_seed

from trainer import MapDiffTrainer

try:
    import dataset_splits
    import stats_utils
except ImportError:
    for _up in ('.', '..', '../..', '../../..'):
        _cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), _up, 'lib')
        if os.path.isdir(_cand):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    import dataset_splits
    import stats_utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

allow_pyg_data_pickles()
patch_inspector_distribute()

# cfg.dataset.name -> lib/dataset_splits.py dataset name. CATH is upstream's own benchmark
# and carries its own three directories, so it is not in the table.
SPLIT_DATASETS = {'ATLAS': 'atlas', 'MDCATH': 'mdcath'}


def resolve_devices(cfg):
    """The training device named by ``cfg.compute``.

    A requested index that does not exist on this machine falls back to cuda:0 with a
    warning rather than crashing, so single-GPU dev runs still work.
    """
    if not torch.cuda.is_available():
        print('No CUDA device available -- running everything on CPU')
        return torch.device('cpu'), torch.device('cpu')

    n_gpus = torch.cuda.device_count()

    def pick(name, what):
        dev = torch.device(name)
        if dev.type == 'cuda' and (dev.index or 0) >= n_gpus:
            print(f'WARNING: {what} requested {dev}, but this machine has {n_gpus} GPU(s) -- '
                  f'falling back to cuda:0. Lower train.batch_size/mask_train.batch_size if '
                  f'this OOMs.')
            return torch.device('cuda:0')
        return dev

    return pick(cfg.compute.device, 'training')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_file_ids(ds_name, index_csv, val_fold, test_fold, directories):
    """``{split: [<protein_id>.pt, ...]}`` for each processed graph directory.

    Deliberately *not* ``os.listdir(dir)``: that trusts whatever happens to be on disk (a
    stale file from an earlier fold, a half-finished featurization run) and returns it in
    filesystem order. Driving the lists from the split index instead means this run trains,
    validates and tests on exactly the folds lib/dataset_splits.py assigns -- the same three
    splits scripts/train_residue_classifier.py and comparisons/{PiFold,DynamicMPNN} cut --
    whatever the directories contain, and the returned order is the index csv's, not the
    filesystem's.

    Proteins the featurizer could not produce a graph for (no deposited entry, DSSP failure)
    are reported rather than silently skipped: they are the main way two models' held-out sets
    drift apart.
    """
    train_ids, val_ids, test_ids = dataset_splits.get_splits(
        ds_name, index_csv, val_fold=val_fold, test_fold=test_fold)
    wanted = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    present = {}
    for split, protein_ids in wanted.items():
        directory = directories[split]
        files = [f'{protein_id}.pt' for protein_id in protein_ids
                 if os.path.exists(os.path.join(directory, f'{protein_id}.pt'))]
        missing = [p for p in protein_ids if f'{p}.pt' not in files]
        if missing:
            print(f'{split}: {len(missing)}/{len(protein_ids)} proteins in the split index have no '
                  f'graph in {directory} and are skipped: {missing}')
        if not files:
            raise FileNotFoundError(
                f'{split}: none of the {len(protein_ids)} proteins in the split index were featurized '
                f'into {directory} -- run data/generate_graph_relaxed.py --ds-name {ds_name} first')
        present[split] = files
    return present


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(OmegaConf.to_yaml(cfg))
    print(f"Output directory: {output_dir}")

    device = resolve_devices(cfg)
    print(f"Training on {device}.")
    # The one seed every comparison run uses (lib/dataset_splits.SEED), rather than each
    # model's own upstream default.
    set_seed(dataset_splits.SEED)

    if cfg.wandb.use:
        wandb_run = stats_utils.init_wandb(
            cfg.wandb.key_file, cfg.wandb.entity, cfg.wandb.project,
            cfg.wandb.run_name or f"MapDiff_{dataset_splits.dataset_tag(SPLIT_DATASETS[cfg.dataset.name])}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        artifact = wandb.Artifact(name="scripts", type="model_file")
        for dependency in (__file__, "trainer.py", "dataloader/large_dataset.py", "model/ipa/ipa_net.py"):
            if os.path.exists(dependency):
                artifact.add_file(os.path.abspath(dependency))
        wandb_run.log_artifact(artifact)
        wandb_run.log({"output_dir": output_dir})
    else:
        wandb_run = None

    # ATLAS and MDCATH are trained and evaluated separately (one run each, dataset=atlas or
    # dataset=mdcath), matching scripts/train_residue_classifier.py's --ds-name.
    if cfg.dataset.name not in ('CATH', 'ATLAS', 'MDCATH'):
        raise ValueError(f"unknown dataset {cfg.dataset.name}")

    if cfg.dataset.name == 'CATH':
        train_ID = sorted(os.listdir(cfg.dataset.train_dir))
        val_ID = sorted(os.listdir(cfg.dataset.val_dir))
        test_ID = sorted(os.listdir(cfg.dataset.test_dir))
    else:
        split_ids = split_file_ids(
            SPLIT_DATASETS[cfg.dataset.name], cfg.dataset.get('index_csv') or None,
            cfg.dataset.get('val_fold', dataset_splits.DEFAULT_VAL_FOLD),
            cfg.dataset.get('test_fold', dataset_splits.DEFAULT_TEST_FOLD),
            {'train': cfg.dataset.train_dir, 'val': cfg.dataset.val_dir, 'test': cfg.dataset.test_dir})
        train_ID, val_ID, test_ID = split_ids['train'], split_ids['val'], split_ids['test']

    train_dataset = Cath(train_ID, cfg.dataset.train_dir, max_length=cfg.dataset.max_length)
    val_dataset = Cath(val_ID, cfg.dataset.val_dir, max_length=cfg.dataset.max_length)
    test_dataset = Cath(test_ID, cfg.dataset.test_dir, max_length=cfg.dataset.max_length)
    print(f'Train on {cfg.dataset.name} dataset with {len(train_dataset)} training data, {len(val_dataset)} '
          f'val data, {len(test_dataset)} test data')

    if wandb_run:
        # The exact proteins this run trained and scored on, *after* the max_length filter --
        # logged so PiFold's and DynamicMPNN's manifests can be diffed against it rather than
        # assumed identical (all three cut the same split, but each drops what it cannot
        # featurize).
        wandb_run.summary['split_counts'] = {'train': len(train_dataset), 'val': len(val_dataset),
                                              'test': len(test_dataset)}
        wandb_run.summary['test_split_ids'] = [ID[:-3] for ID in test_dataset.list_IDs]

    # ---- stage 1: mask-prior IPA pretraining dataloader ----
    mask_collator = CollatorIPAPretrain(candi_rate=cfg.mask_train.candi_rate, mask_rate=cfg.mask_train.mask_rate,
                                        replace_rate=cfg.mask_train.replace_rate, keep_rate=cfg.mask_train.keep_rate)
    mask_train_loader = DataLoader(train_dataset, batch_size=cfg.mask_train.batch_size, shuffle=True, num_workers=6,
                                   collate_fn=mask_collator)

    # ---- stage 2: denoising diffusion dataloaders ----
    diff_collator = CollatorDiff()
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=16,
                              collate_fn=diff_collator)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=16,
                            collate_fn=diff_collator)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=16,
                             collate_fn=diff_collator)

    train_num_steps = len(train_loader) * cfg.train.train_epochs + 1

    egnn_model = EGNN_NET(input_feat_dim=cfg.model.input_feat_dim, hidden_channels=cfg.model.hidden_dim,
                         edge_attr_dim=cfg.model.edge_attr_dim, dropout=cfg.model.drop_out, n_layers=cfg.model.depth,
                         update_edge=cfg.model.update_edge, norm_coors=cfg.model.norm_coors,
                         update_coors=cfg.model.update_coors, update_global=cfg.model.update_global,
                         embedding=cfg.model.embedding, embedding_dim=cfg.model.embedding_dim,
                         norm_feat=cfg.model.norm_feat, embed_ss=cfg.model.embed_ss)

    prior_model = IPANetPredictor(dropout=cfg.model.ipa_drop_out, max_length=cfg.model.ipa_pe_max_len)

    skip_stage1 = bool(cfg.prior_model.path)
    if skip_stage1:
        checkpoint = torch.load(cfg.prior_model.path, map_location='cpu')
        prior_model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded pretrained mask-prior IPA checkpoint from {cfg.prior_model.path}, skipping stage 1")

    diffusion_model = Prior_Diff(egnn_model, prior_model, timesteps=cfg.diffusion.timesteps,
                                 objective=cfg.diffusion.objective, noise_type=cfg.diffusion.noise_type,
                                 sample_method=cfg.diffusion.sample_method,
                                 min_mask_ratio=cfg.mask_prior.min_mask_ratio,
                                 dev_mask_ratio=cfg.mask_prior.dev_mask_ratio,
                                 marginal_dist_path=cfg.dataset.marginal_train_dir,
                                 ensemble_num=cfg.diffusion.ensemble_num)

    print(f"Prior model parameters: {count_parameters(prior_model)}")
    print(f"Diffusion model parameters: {count_parameters(diffusion_model)}")
    if wandb_run:
        # `params` is the key scripts/train_residue_classifier.py logs its parameter count
        # under; the two MapDiff-specific breakdowns sit alongside it.
        wandb_run.log({"params": count_parameters(diffusion_model),
                        "prior_params": count_parameters(prior_model),
                        "diffusion_params": count_parameters(diffusion_model)})

    prior_optimizer = Adam(prior_model.parameters(), lr=cfg.mask_train.lr, betas=(0.95, 0.999),
                          weight_decay=cfg.mask_train.weight_decay)
    prior_scheduler = None
    if cfg.mask_train.scheduler and not skip_stage1:
        prior_scheduler = lr_scheduler.OneCycleLR(
            prior_optimizer, max_lr=cfg.mask_train.lr,
            total_steps=cfg.mask_train.train_epochs * len(mask_train_loader))

    optimizer = Adam(diffusion_model.parameters(), lr=cfg.train.lr, betas=(0.95, 0.999),
                     weight_decay=cfg.train.weight_decay)
    scheduler = None
    if cfg.train.scheduler:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.train.lr, total_steps=train_num_steps)

    trainer = MapDiffTrainer(
        cfg,
        prior_model=prior_model, prior_optimizer=prior_optimizer, mask_train_dataloader=mask_train_loader,
        diffusion_model=diffusion_model, optimizer=optimizer,
        train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader,
        device=device, output_dir=output_dir, wandb_run=wandb_run,
        prior_scheduler=prior_scheduler, scheduler=scheduler,
        train_batch_size=cfg.train.batch_size, train_num_steps=train_num_steps,
        save_and_sample_every=cfg.train.save_and_sample_every,
        ddim_steps=cfg.diffusion.ddim_steps, sample_method=cfg.diffusion.sample_method,
        ensemble_num=cfg.diffusion.ensemble_num,
    )

    if not skip_stage1:
        trainer.fit_prior()

    trainer.train()
    trainer.test()
    trainer.save_table_results()

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
