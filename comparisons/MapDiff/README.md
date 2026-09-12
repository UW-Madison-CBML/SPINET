# Training MapDiff on ATLAS data

This folder trains/evaluates MapDiff against a **pristine, unmodified
checkout of [peizhenbai/MapDiff](https://github.com/peizhenbai/MapDiff)** --
the repo itself never needs to be edited. `train.sh`/`eval.sh` extract a
tarball of the repo and copy the ATLAS-dataset + wandb/scRMSD support below
on top of it at run time, so re-cloning/re-pulling `MapDiff/` and re-tarring
it always works.

```
MapDiff/
  conf/                          # Hydra config layered onto the pristine conf/ tree:
    train.yaml                     new: combined top-level config for train.py (both stages)
    dataset/atlas.yaml             new: ATLAS dataset config
    dataset/cath.yaml              + max_length (shared Cath dataset class, used for ATLAS too)
    model/egnn.yaml                + ipa_pe_max_len (variable-length IPA positional encoding)
    mask_train/default.yaml        new: stage-1 (mask-prior IPA pretrain) hyperparameters
    train/train_diff.yaml          stage-2 (diffusion) hyperparameters
    wandb/basic.yaml                new: wandb.use toggle (off by default), one run for both stages
  dataloader/large_dataset.py    + max_length filtering
  model/ipa/ipa_net.py           + configurable positional-encoding table size
  data/generate_graph_atlas.py   new: ATLAS hdf5 -> MapDiff graph featurization
  train.py                       new: single entry point, runs both training stages in one wandb run
  trainer.py                     new: MapDiffTrainer -- merges trainer/trainer.py + trainer/mask_ipa_trainer.py,
                                  drops comet_ml, adds scRMSD + a confusion-matrix image (see below)
  eval_atlas.py                  standalone re-eval of a saved checkpoint on ATLAS validation (+ scRMSD)
train.sh / train.sub             CHTC entry point + submit file for training
eval.sh / eval.sub                CHTC entry point + submit file for evaluation
Dockerfile / build_and_push.sh   training image (CUDA 12.1, PyTorch 2.1.2, PyG 2.4.0, DSSP, Hydra, wandb, transformers)
wandb_api.txt                     your W&B API key (not checked in)
results/                          past run outputs (checkpoints, logs, configs) copied back from CHTC
```

## Why one script instead of two

Upstream MapDiff trains in two stages -- mask-prior IPA pretraining, then
denoising diffusion training seeded by that IPA checkpoint -- as two
separate scripts (`mask_ipa_pretrain.py`, `main.py`), each with its own
comet_ml/wandb run and a checkpoint round-trip through disk in between.
`train.py` runs both stages back-to-back in one process and one W&B run:
`MapDiffTrainer.fit_prior()` trains the mask-prior IPA model in place, and
since the diffusion model (`Prior_Diff`) holds a direct reference to that
same model instance, stage 2 (`MapDiffTrainer.train()`) sees the trained
weights immediately -- no checkpoint save/reload needed between stages
(though `fit_prior()` still writes one to `<output_dir>/model/` for the
record). Pass `prior_model.path=<ckpt>` to skip stage 1 and load an
already-trained IPA checkpoint instead.

## Metrics logged (a la `scripts/train_residue_classifier.py`)

Every validation pass (during diffusion training) and the final test pass
log to the same W&B run:
- sequence recovery (top-1/mean/median) and perplexity (as before)
- BLOSUM-weighted recovery (NSSR42/62/80/90, test only)
- an amino-acid confusion matrix, logged as a `wandb.Image`
- **scRMSD**: the predicted sequence is folded with ESMFold
  (`lib/scrmsd.py`) and Kabsch-RMSD'd against the protein's ground-truth
  backbone (CA/N/C/O) coordinates -- the same self-consistency metric
  `scripts/train_residue_classifier.py` computes for the sheaf model, so
  results are directly comparable.

## 1. Build the pristine repo tarball

```bash
cd comparisons/MapDiff
git clone https://github.com/peizhenbai/MapDiff
tar czf MapDiff.tar.gz MapDiff/
```

Re-run this any time you want to pick up upstream changes -- nothing else
in this folder needs to change.

## 2. Build and push the training image

```bash
./build_and_push.sh <your_dockerhub_username>
```

Then set `docker_image = <your_dockerhub_username>/mapdiff:[version]` in
`train.sub` and `eval.sub`.

## 3. Dataset

MapDiff reads directly from the same shared ATLAS store every other model in this
repo uses -- `atlas_data.h5` (one representative frame per protein, plus the full
trajectory) and `atlas_cross_val_index.csv` (the `pdb` -> `cross_val` fold assignment
and each protein's pre-selected `random_indices` frame) -- instead of a pre-split
folder of raw PDB files. **Fold 0 is held out as validation, and reused as the test
split too** (MapDiff wants train/val/test; ATLAS only has one held-out fold), the exact
same split `scripts/train_residue_classifier.py` and
`comparisons/{gvp-pytorch,DynamicMPNN}/train.py` use.

Both files are staged on Pelican (see `train.sub`/`eval.sub`'s `transfer_input_files`).
`train.sh`/`eval.sh` run `data/generate_graph_atlas.py`, which for every protein: looks
up its representative frame in the hdf5 store, writes it out as a minimal single-model
PDB (`lib/atlas_frame_pdb.write_frame_pdb`) so DSSP secondary-structure assignment
(`data.generate_graph_cath.pdb2graph`) works unchanged, featurizes it, and deletes the
temporary PDB. Output lands in `surffold_data/atlas_process/{train,validation}/`
(`validation/` is pointed at by both `dataset.val_dir` and `dataset.test_dir`, see
`conf/dataset/atlas.yaml`), plus the ATLAS amino-acid marginal. This step re-runs every
job (CHTC scratch dirs are ephemeral) unless `surffold_data/atlas_process/train/`
already exists in the job's scratch. `eval_atlas.py` does the same featurization
on the fly, per protein, with no precomputed graph directory needed.

## 4. Weights & Biases

Put your W&B API key in `wandb_api.txt` (referenced by both submit files'
`transfer_input_files`, not checked in).

## 5. Training (both stages, one job)

```bash
mkdir -p logs
condor_submit train.sub
```

Edit `train.sub`'s `arguments` line to change hyperparameters -- they're
forwarded straight to `train.py` (see `conf/train.yaml`,
`conf/mask_train/default.yaml`, `conf/train/train_diff.yaml`). Results
(both the mask-prior IPA checkpoint and the diffusion checkpoint, both
under `outputs/*/model/`) land in `results/atlas_mapdiff_<Cluster>_<Process>/`
on job exit.

## 6. Evaluating a checkpoint

`eval.sub`/`eval_atlas.py` run against fold 0 (the same validation/test split
training used) and log per-protein perplexity, top-1/5/10 sequence recovery, and
scRMSD (mean +/- standard deviation across proteins) to W&B.

Edit `eval.sub`: fill in the real checkpoint + config paths (replacing the
`FIXME` placeholders in `transfer_input_files`) from your `train.sub` run.
Then:

```bash
mkdir -p logs
condor_submit eval.sub
```

Monitor any of the above with `condor_q`/`condor_watch_q`; logs land in
`logs/`.

## 7. Local (non-CHTC) runs

```bash
git clone https://github.com/peizhenbai/MapDiff /tmp/MapDiff_run
cp -r conf dataloader model data train.py trainer.py /tmp/MapDiff_run/
cp ../../lib/scrmsd.py ../../lib/stats_utils.py ../../lib/atlas_frame_pdb.py ../../scripts/load_dynamics.py wandb_api.txt /tmp/MapDiff_run/
cp /path/to/atlas_data.h5 /path/to/atlas_cross_val_index.csv /tmp/MapDiff_run/
cd /tmp/MapDiff_run
python train.py dataset=atlas wandb.use=True
python eval_atlas.py --checkpoint ./checkpoint.pt --config ./config.yaml --atlas-h5 ./atlas_data.h5 --cross-val-csv ./atlas_cross_val_index.csv
```

(`train.py`/`eval_atlas.py`/`trainer.py` fall back to walking up to a
`lib/` directory to import `stats_utils`/`scrmsd`/`atlas_frame_pdb` if they
aren't sitting flat next to them.)
