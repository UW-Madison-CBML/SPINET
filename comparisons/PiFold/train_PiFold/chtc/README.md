# Running PiFold on CHTC

This bundles a Docker image with PiFold's dependencies (CUDA 11.8, PyTorch
2.1.2, torch_scatter, biopython, numpy, tqdm, wandb) plus HTCondor submit
files to run training/inference as GPU jobs on CHTC.

The image ships only the software environment. Code is a **pristine PiFold
checkout + this folder's overlay**, and data is staged separately -- neither
is baked into the image, so you don't need to rebuild/repush it when the code
or ATLAS overlay changes (only when a dependency changes).

## 1. Build and push the image

Docker must run somewhere with internet access and a Docker Hub account
(CHTC execute nodes pull the image from a registry, they can't use a
locally-built image directly):

```bash
cd chtc
./build_and_push.sh <your_dockerhub_username>
```

## 2. Point the submit files at your image

Edit `pifold.sub` and `eval.sub` and set:

```
docker_image = <your_dockerhub_username>/pifold:latest
```

## 3. Build the pristine PiFold tarball

```bash
cd ..   # train_PiFold/
git clone https://github.com/A4Bio/PiFold
tar czf PiFold.tar.gz PiFold/
```

`pifold.sub`/`eval.sub` transfer `../PiFold.tar.gz` and `../overlay/` in;
`run_pifold.sh`/`run_eval.sh` extract the tarball and copy `overlay/` on top
of it at run time -- `PiFold/` itself is never modified, so re-running this
step to pick up upstream changes always works.

## 4. Dataset

The ATLAS-derived dataset (first frame of each MD simulation, pre-split into
`train/`, `valid/`, `test/` folders of PDB files) is staged at:

```
/staging/groups/bhaskar_group/sheaf_protein_dynamics/surffold_data.tar.gz
```

`pifold.sub`/`eval.sub` already reference this path in `transfer_input_files`,
so HTCondor copies it into the job's scratch directory. `run_pifold.sh` /
`run_eval.sh` auto-extract it into `PiFold/surffold_data/` if not already
present, and `--data_root ./surffold_data/` (set in each submit file's
`arguments`) points `API.atlas_dataset.ATLAS` at it.

For other/smaller datasets (e.g. CATH, TS50), tar them as `data.tar.gz`
(expanding to a top-level `data/` folder) and add them to
`transfer_input_files` instead, or stage them under `/staging/<username>/`
the same way.

## 5. Weights & Biases

Put your W&B API key in `../wandb_api.txt` (referenced by both submit files'
`transfer_input_files`, not checked in). Training now logs to W&B every
`log_step` epochs (train loss/perplexity, valid loss/perplexity, test
perplexity/recovery, per-category recovery) in addition to evaluation.

## 6. Submit

```bash
cd chtc
mkdir -p logs
condor_submit pifold.sub
```

Edit the `arguments` line in `pifold.sub` to change hyperparameters -- they're
forwarded straight to `main.py` (see `../overlay/parser.py` for all options,
e.g. `--data_name ATLAS`, `--num_encoder_layers`, `--lr`, `--no_wandb`, etc.).

Monitor with `condor_q` / `condor_watch_q`; logs land in `chtc/logs/`.
Results (including `checkpoint.pth`/`model_param.json`) are copied back to
`../results/<ClusterId>_<ProcId>/chtc_run/` on job exit.

## 7. Evaluating a checkpoint

`eval.sub` / `run_eval.sh` run `../eval_atlas.py` against the ATLAS
validation split on CHTC (same image and staged dataset as training) and log
per-protein recovery (top-1/5/10) and perplexity, mean +/- standard
deviation, to Weights & Biases. Edit `eval.sub`'s checkpoint/model_param
paths in `transfer_input_files` to point at the run you want to evaluate,
then:

```bash
cd chtc
mkdir -p logs
condor_submit eval.sub
```
