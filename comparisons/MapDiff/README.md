# Training MapDiff on our ATLAS / mdCATH data

This folder trains/evaluates MapDiff against a **pristine, unmodified
checkout of [peizhenbai/MapDiff](https://github.com/peizhenbai/MapDiff)** --
the repo itself never needs to be edited. `train.sh`/`eval.sh` extract a
tarball of the repo and copy the dataset + wandb/scRMSD support below on top
of it at run time, so re-cloning/re-pulling `MapDiff/` and re-tarring it
always works.

ATLAS and mdCATH are trained and evaluated **separately** -- one job each --
matching `scripts/train_residue_classifier.py`'s `--ds-name`.

```
MapDiff/
  conf/                          # Hydra config layered onto the pristine conf/ tree:
    train.yaml                     new: combined top-level config for train.py (both stages)
    dataset/atlas.yaml             new: ATLAS dataset config
    dataset/mdcath.yaml            new: mdCATH dataset config
    dataset/cath.yaml              + max_length (shared Cath dataset class, used for both)
    model/egnn.yaml                + ipa_pe_max_len (variable-length IPA positional encoding)
    mask_train/default.yaml        new: stage-1 (mask-prior IPA pretrain) hyperparameters
    train/train_diff.yaml          stage-2 (diffusion) hyperparameters
    wandb/basic.yaml                new: wandb.use toggle (off by default), one run for both stages
  dataloader/large_dataset.py    + max_length filtering
  model/ipa/ipa_net.py           + configurable positional-encoding table size
  data/generate_graph_relaxed.py new: relaxed (deposited) PDB -> MapDiff graph featurization
  train.py                       new: single entry point, runs both training stages in one wandb run
  trainer.py                     new: MapDiffTrainer -- merges trainer/trainer.py + trainer/mask_ipa_trainer.py,
                                  drops comet_ml, adds scRMSD + a confusion-matrix image (see below)
  eval_relaxed.py                standalone re-eval of a saved checkpoint on the held-out split (+ scRMSD)
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
  **relaxed (deposited)** backbone (CA/N/C/O) coordinates -- the same
  self-consistency metric `comparisons/{PiFold,DynamicMPNN}` compute, so
  results are directly comparable. No alignment step is needed here: the
  graphs MapDiff trains on are themselves featurized from those deposited
  structures, so `atom_pos` already *is* the relaxed reference.

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

MapDiff is a static-structure inverse-folding model, so it trains and evaluates on each
protein's **relaxed (deposited) PDB entry**, not on a frame pulled out of the MD
trajectory. `lib/relaxed_pdb.py` downloads the RCSB entry, picks the chain the dataset id
names, and crops it to the residues the trajectory covers (using the trajectory's own
residue list as the reference sequence), so MapDiff scores the same residues as every
other model in the comparison. For mdCATH that cropping is also what reduces a deposited
chain to the single CATH *domain* the `domain` id names (e.g. `12asA00` -> entry `12as`,
chain `A`, domain 02). Downloads are cached under `pdb_cache/` in the job's scratch dir.

(`comparisons/DynamicMPNN` deliberately does *not* do this -- it keeps training on MD
conformer ensembles, since that ensemble input is the thing being benchmarked. It uses
the relaxed structures only as the scRMSD reference.)

Splits come from `lib/dataset_splits.py`, identical to
`scripts/train_residue_classifier.py`:

- **ATLAS** (`--ds-name atlas`, `atlas_cross_val_index.csv`): fold 0 of the `cross_val`
  column is held out, the other four folds train. ATLAS has no further held-out set, so
  fold 0 doubles as both validation and test (see `conf/dataset/atlas.yaml`).
- **mdCATH** (`--ds-name mdcath`, `mdcath_320_0_topology_split.csv`): the `train` and
  `test` rows both train (both are topology-split slices of the training pool);
  **evaluation is on the `validation` rows only**, which likewise double as the test
  split (see `conf/dataset/mdcath.yaml`).

The dataset's trajectory store and split index are staged on Pelican (see
`train.sub`/`eval.sub`'s `transfer_input_files`); the store is read *only* for the
reference residue sequences. `train.sh` runs `data/generate_graph_relaxed.py`, which for
every protein writes its cropped deposited chain out as a PDB (side chains and residue
numbering intact, so DSSP secondary-structure assignment via
`data.generate_graph_cath.pdb2graph` works unchanged), featurizes it, and deletes the
temporary file. Output lands in `surffold_data/<ds>_process/{train,validation}/`
(`validation/` is pointed at by both `dataset.val_dir` and `dataset.test_dir`), plus the
train-split amino-acid marginal. This step re-runs every job (CHTC scratch dirs are
ephemeral) unless `surffold_data/<ds>_process/train/` already exists in the job's
scratch. `eval_relaxed.py` does the same featurization on the fly, per protein, with no
precomputed graph directory needed.

## 4. Weights & Biases

Put your W&B API key in `wandb_api.txt` (referenced by both submit files'
`transfer_input_files`, not checked in).

## 5. Training (both stages, one job)

```bash
mkdir -p logs
condor_submit train.sub
```

For mdCATH, override the dataset macros:

```bash
condor_submit train.sub DS_NAME=mdcath \
    DS_H5='$(ResearchDrive)/mdcath_spinet_320_0.h5' \
    DS_CSV='$(ResearchDrive)/mdcath_320_0_topology_split.csv'
```

Edit `train.sub`'s `arguments` line to change hyperparameters -- they're
forwarded straight to `train.py` (see `conf/train.yaml`,
`conf/mask_train/default.yaml`, `conf/train/train_diff.yaml`). Results
(both the mask-prior IPA checkpoint and the diffusion checkpoint, both
under `outputs/*/model/`) land in `results/<ds>_mapdiff_<Cluster>_<Process>/`
on job exit.

## 6. Evaluating a checkpoint

`eval.sub`/`eval_relaxed.py` run against the dataset's held-out split (the same one
training used) and log per-protein perplexity, top-1/5/10 sequence recovery, and
scRMSD (mean +/- standard deviation across proteins) to W&B. `DS_NAME`/`DS_H5`/`DS_CSV`
select the dataset, exactly as in `train.sub`.

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
cp ../../lib/scrmsd.py ../../lib/stats_utils.py ../../lib/relaxed_pdb.py ../../lib/dataset_splits.py wandb_api.txt /tmp/MapDiff_run/
cp /path/to/atlas_data.h5 /path/to/atlas_cross_val_index.csv /tmp/MapDiff_run/
cd /tmp/MapDiff_run
python data/generate_graph_relaxed.py --ds-name atlas
python train.py dataset=atlas wandb.use=True
python eval_relaxed.py --checkpoint ./checkpoint.pt --config ./config.yaml --ds-name atlas
```

(`train.py`/`eval_relaxed.py`/`trainer.py` fall back to walking up to a
`lib/` directory to import
`stats_utils`/`scrmsd`/`relaxed_pdb`/`dataset_splits` if they aren't sitting
flat next to them.)
