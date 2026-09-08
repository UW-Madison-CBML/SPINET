# Training MapDiff on ATLAS data

This folder holds everything needed to train/evaluate MapDiff against a
**pristine, unmodified checkout of
[peizhenbai/MapDiff](https://github.com/peizhenbai/MapDiff)** -- the repo
itself never needs to be edited. `chtc/run_*.sh` extract a tarball of the
repo and copy `overlay/` (ATLAS dataset support + wandb wiring alongside the
existing comet_ml logging) on top of it at run time, so re-cloning/re-pulling
`MapDiff/` and re-tarring it always works.

```
train_MapDiff/
  overlay/                       # files copied onto a pristine MapDiff/ checkout before running:
    conf/dataset/atlas.yaml        # new: ATLAS dataset config
    conf/dataset/cath.yaml         # + max_length (shared Cath dataset class, used for ATLAS too)
    conf/model/egnn.yaml           # + ipa_pe_max_len (variable-length IPA positional encoding)
    conf/diff_config.yaml, conf/mask_pretrain.yaml  # + `wandb:` in Hydra defaults
    conf/wandb/{basic,mask_pretrain}.yaml  # new: wandb.use toggle (off by default)
    dataloader/large_dataset.py    # + max_length filtering
    model/ipa/ipa_net.py           # + configurable positional-encoding table size
    data/generate_graph_atlas.py   # new: ATLAS PDB -> MapDiff graph featurization
    main.py, mask_ipa_pretrain.py  # + ATLAS wiring, + wandb_run alongside comet Experiment
    trainer/trainer.py, trainer/mask_ipa_trainer.py  # + wandb.log() alongside every comet log_metric()
  eval_atlas.py         # standalone re-eval of a saved checkpoint on ATLAS test split (uses lib/stats_utils.py)
  wandb_api.txt         # your W&B API key (not checked in)
  chtc/                 # CHTC entry points + submit files (see chtc/README.md)
  results/              # past run outputs (checkpoints, logs, configs) copied back from CHTC
```

## 1. Build the pristine repo tarball

```bash
cd /path/to/Sheaf_protein_dynamics
git clone https://github.com/peizhenbai/MapDiff
tar czf train_MapDiff/MapDiff.tar.gz MapDiff/
```

Re-run this any time you want to pick up upstream changes -- nothing in
`train_MapDiff/` needs to change.

## 2. CHTC training/eval

See `chtc/README.md` for the full CHTC workflow (Docker image, dataset
staging + featurization, the two-stage training pipeline, submitting
`mask_pretrain.sub` -> `train.sub` -> `eval.sub`). In short:

```bash
cd chtc
mkdir -p logs
condor_submit mask_pretrain.sub   # stage 1: mask-prior IPA pretraining
# ... then, with the resulting checkpoint filled into train.sub:
condor_submit train.sub           # stage 2: denoising diffusion training
condor_submit eval.sub            # evaluate a trained checkpoint on the ATLAS test split
```

Both training stages log to W&B by default (`wandb.use=True` in each
submit file's `arguments`, alongside `comet.use=False` since CHTC has no
interactive comet login) -- per-step/epoch train loss, per-epoch validation
recovery/perplexity, and final test metrics, matching exactly what used to
go only to comet_ml.

## 3. Local (non-CHTC) runs

```bash
git clone https://github.com/peizhenbai/MapDiff /tmp/MapDiff_run
cp -r overlay/. /tmp/MapDiff_run/
cp wandb_api.txt /tmp/MapDiff_run/
cd /tmp/MapDiff_run
python mask_ipa_pretrain.py --config-name=mask_pretrain dataset=atlas wandb.use=True
python main.py --config-name=diff_config dataset=atlas prior_model.path=./ipa_checkpoint.pt wandb.use=True
python eval_atlas.py --checkpoint ./checkpoint.pt --config ./config.yaml --data_root /path/to/surffold_data/
```

(`main.py`/`mask_ipa_pretrain.py`/`eval_atlas.py` fall back to walking up to
a `lib/` directory to import `stats_utils` if it isn't sitting flat next to
them.)
