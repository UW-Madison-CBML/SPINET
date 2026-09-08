# Training PiFold on ATLAS data

This folder holds everything needed to train/evaluate PiFold against a
**pristine, unmodified checkout of
[A4Bio/PiFold](https://github.com/A4Bio/PiFold)** -- the repo itself never
needs to be edited. `chtc/run_pifold.sh` / `chtc/run_eval.sh` extract a
tarball of the repo and copy `overlay/` (ATLAS dataset support + wandb
wiring) on top of it at run time, so re-cloning/re-pulling `PiFold/` and
re-tarring it always works.

```
train_PiFold/
  overlay/            # files copied onto a pristine PiFold/ checkout before running:
    parser.py           # + ATLAS choice/--max_length, + --wandb_* args
    main.py              # + wandb logging in Exp.train()/valid()/test()
    API/atlas_dataset.py # new: ATLAS dataset loader
    API/dataloader.py, API/__init__.py  # wire ATLAS into load_data()
    API/dataloader_gtrans.py, API/featurizer.py  # numpy np.int->int compat fixes
  eval_atlas.py        # standalone re-eval of a saved checkpoint on ATLAS validation (uses lib/stats_utils.py)
  model_size.py        # reports a PiFold config's parameter count
  wandb_api.txt        # your W&B API key (not checked in)
  chtc/                # CHTC entry points + submit files (see chtc/README.md)
  results/             # past run outputs (checkpoints, logs) copied back from CHTC
  legacy/              # superseded PiFold_train.sh/.sub, kept for reference only
```

## 1. Build the pristine repo tarball

```bash
cd /path/to/Sheaf_protein_dynamics
git clone https://github.com/A4Bio/PiFold
tar czf train_PiFold/PiFold.tar.gz PiFold/
```

Re-run this any time you want to pick up upstream changes -- nothing in
`train_PiFold/` needs to change.

## 2. CHTC training/eval

See `chtc/README.md` for the full CHTC workflow (Docker image, dataset
staging, submitting `pifold.sub`/`eval.sub`). In short:

```bash
cd chtc
mkdir -p logs
condor_submit pifold.sub   # train (logs train/valid/test metrics to W&B every log_step epochs)
condor_submit eval.sub     # evaluate a trained checkpoint on the ATLAS validation split
```

## 3. Local (non-CHTC) runs

```bash
git clone https://github.com/A4Bio/PiFold /tmp/PiFold_run
cp -r overlay/. /tmp/PiFold_run/
cp wandb_api.txt /tmp/PiFold_run/
cd /tmp/PiFold_run
python main.py --data_name ATLAS --data_root /path/to/surffold_data/ --epoch 100
python eval_atlas.py --run_dir ./results/debug --data_root /path/to/surffold_data/
```

`model_size.py` (parameter count / memory footprint for a given set of model
hyperparameters) works the same way -- copy it into the deployed checkout
alongside `main.py` before running it.

(`main.py`/`eval_atlas.py` fall back to walking up to a `lib/` directory to
import `stats_utils` if it isn't sitting flat next to them -- pass
`--no_wandb` to `main.py` to skip W&B for a quick smoke test.)
