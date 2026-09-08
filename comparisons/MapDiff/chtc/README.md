# Running MapDiff on CHTC

This bundles a Docker image with MapDiff's dependencies (CUDA 11.7,
PyTorch 1.13.1, PyG 2.4.0, torch_scatter/torch_cluster, DSSP, RDKit,
Hydra, comet_ml, wandb) plus HTCondor submit files to run MapDiff's full
pipeline -- data featurization, both training stages, and evaluation --
as GPU jobs on CHTC.

The image ships only the software environment. Code is a **pristine MapDiff
checkout + `../overlay/`**, and data is staged separately -- neither is
baked into the image, so you don't need to rebuild/repush it when the code
or ATLAS overlay changes (only when a dependency changes).

## 1. Build and push the image

Docker must run somewhere with internet access and a Docker Hub account
(CHTC execute nodes pull the image from a registry, they can't use a
locally-built image directly):

```bash
cd chtc
./build_and_push.sh <your_dockerhub_username>
```

Then set `docker_image = <your_dockerhub_username>/mapdiff:v1` in
`mask_pretrain.sub`, `train.sub`, and `eval.sub`.

## 2. Build the pristine MapDiff tarball

```bash
cd ..   # train_MapDiff/
git clone https://github.com/peizhenbai/MapDiff
tar czf MapDiff.tar.gz MapDiff/
```

Each submit file transfers `../MapDiff.tar.gz` and `../overlay/` in;
`run_*.sh` extract the tarball and copy `overlay/` on top of it at run time
-- `MapDiff/` itself is never modified, so re-running this step to pick up
upstream changes always works.

## 3. Dataset

The ATLAS-derived dataset (the initial static structure of each MD
simulation, pre-split into `train/`, `validation/`, `test/` folders of raw
`.pdb` files) is staged at:

```
/staging/groups/bhaskar_group/sheaf_protein_dynamics/surffold_data.tar.gz
```

Each submit file below references this path in `transfer_input_files`, so
HTCondor copies it into the job's scratch directory. The corresponding
`run_*.sh` script auto-extracts it into `MapDiff/surffold_data/` if not
already present, then runs `overlay/data/generate_graph_atlas.py` to
featurize the raw PDBs (DSSP secondary structure + k-NN residue graphs,
mirroring `data/generate_graph_cath.py`'s pipeline for CATH) into
`MapDiff/surffold_data/atlas_process/{train,validation,test}/`, computing
the ATLAS amino-acid marginal distribution along the way (needed by
`model.prior_diff.Prior_Diff`'s `marginal` noise model). This featurizing
step re-runs every job (CHTC scratch dirs are ephemeral), so expect it to
add real time on top of training/eval -- it's skipped automatically if
`MapDiff/surffold_data/atlas_process/train/` already exists in the job's
scratch.

## 4. Weights & Biases

Put your W&B API key in `../wandb_api.txt` (referenced by every submit
file's `transfer_input_files`, not checked in). Both training stages log to
W&B by default now (`wandb.use=True` in each submit file's `arguments`) in
addition to the existing comet_ml path -- see `../overlay/conf/wandb/`.

## 5. Training (two stages)

MapDiff trains in two stages: mask-prior IPA pretraining, then denoising
diffusion training seeded by that IPA checkpoint.

### Stage 1: mask-prior IPA pretraining

```bash
cd chtc
mkdir -p logs
condor_submit mask_pretrain.sub
```

Edit `mask_pretrain.sub`'s `arguments` line to change hyperparameters --
they're forwarded straight to `mask_ipa_pretrain.py` (see
`../overlay/conf/mask_pretrain.yaml` / `../overlay/conf/train/train_mask_ipa.yaml`
for all options). Results (including the IPA checkpoint under
`outputs/*/model/`) land in `../results/mask_pretrain_<ClusterId>_<ProcId>/`
on job exit.

### Stage 2: denoising diffusion training

Once stage 1 finishes, edit `train.sub`: fill in the real IPA checkpoint
path (replacing the `FIXME` placeholders in both `transfer_input_files`
and `arguments`' `prior_model.path`) with the `.pt` file from
`../results/mask_pretrain_<ClusterId>_<ProcId>/outputs/*/model/`. Then:

```bash
cd chtc
mkdir -p logs
condor_submit train.sub
```

Results land in `../results/diffusion_<ClusterId>_<ProcId>/` on job exit,
including the trained checkpoint (`outputs/*/model/*_best_*.pt`) and its
resolved config (`outputs/*/configs/config.yaml`) -- both needed for eval.

## 6. Evaluating a checkpoint

`eval.sub` / `run_eval.sh` run `../eval_atlas.py` against the ATLAS test
split on CHTC (same image and staged dataset as training) and log
per-protein perplexity and top-1/5/10 sequence recovery, mean +/- standard
deviation across test proteins, to Weights & Biases.

Edit `eval.sub`: fill in the real checkpoint + config paths (replacing the
`FIXME` placeholders in `transfer_input_files`) from your `train.sub` run,
matching the flat `checkpoint.pt` / `config.yaml` names referenced in
`arguments`. Then:

```bash
cd chtc
mkdir -p logs
condor_submit eval.sub
```

Unlike training, `eval_atlas.py` featurizes each ATLAS test PDB on the fly
(one protein at a time, no precomputed graph directory needed) --
matching the same per-PDB inference pattern as `../model_inference.ipynb`
(in the pristine `MapDiff/` checkout).

Monitor any of the above with `condor_q` / `condor_watch_q`; logs land in
`chtc/logs/`.
