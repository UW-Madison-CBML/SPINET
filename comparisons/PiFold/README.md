# Running PiFold on CHTC

This bundles a Docker image with PiFold's dependencies (CUDA 11.8, PyTorch
2.1.2, torch_scatter, biopython, transformers, h5py, pandas, wandb) plus an
HTCondor submit file to run training/inference as a GPU job on CHTC.

PiFold itself is used as a **pristine, unmodified checkout of
[A4Bio/PiFold](https://github.com/A4Bio/PiFold)** — `run_pifold.sh` extracts a
tarball of it and copies this folder's overlay on top at run time, so
re-cloning/re-pulling and re-tarring always works (same pattern as
`../MapDiff`):

```
API/relaxed_dataset.py   new: relaxed (deposited) PDB -> PiFold dataset, for ATLAS and mdCATH
API/dataloader.py        + the ATLAS/MDCATH branch that builds it
API/__init__.py          + re-export of RelaxedStructures
parser.py                + dataset-selection, scRMSD and W&B flags
main.py                  + W&B logging and the scRMSD pass
run_pifold.sh            CHTC entry point: extract, overlay, run
pifold.sub               submit file
Dockerfile / build_and_push.sh
```

## 1. Build the pristine repo tarball

```bash
cd comparisons/PiFold
git clone https://github.com/A4Bio/PiFold
tar czf PiFold.tar.gz PiFold/
```

## 2. Build and push the image

Docker must run somewhere with internet access and a Docker Hub account
(CHTC execute nodes pull the image from a registry, they can't use a
locally-built image directly):

```bash
./build_and_push.sh <your_dockerhub_username>
```

Then set `docker_image = <your_dockerhub_username>/pifold:latest` in `pifold.sub`.

## 3. Dataset

PiFold is a static-structure inverse-folding model, so it trains and evaluates on each
protein's **relaxed (deposited) PDB entry**, not on a frame pulled out of the MD
trajectory. `lib/relaxed_pdb.py` downloads the RCSB entry, picks the chain the dataset
id names, and crops it to the residues the trajectory covers (using the trajectory's own
residue list as the reference sequence), so PiFold scores the same residues as every
other model in the comparison. For mdCATH that cropping is also what reduces a deposited
chain to the single CATH *domain* the `domain` id names (e.g. `12asA00` → entry `12as`,
chain `A`, domain 02). Downloads are cached under `pdb_cache/` in the job's scratch dir.

(`comparisons/DynamicMPNN` deliberately does *not* do this — it keeps training on MD
conformer ensembles, since that ensemble input is the thing being benchmarked. It uses
the relaxed structures only as the scRMSD reference.)

ATLAS and mdCATH are trained and evaluated **separately**, one job each, matching
`scripts/train_residue_classifier.py`'s `--ds-name`. Splits come from
`lib/dataset_splits.py`:

- **ATLAS** (`--data_name ATLAS`, `atlas_cross_val_index.csv`): fold 0 of the `cross_val`
  column is held out, the other four folds train.
- **mdCATH** (`--data_name MDCATH`, `mdcath_320_0_topology_split.csv`): the `train` and
  `test` rows both train (both are topology-split slices of the training pool);
  **evaluation is on the `validation` rows only**.

Neither dataset has a further held-out set, so the held-out split serves as both PiFold's
`valid` and `test` split.

### Keeping the comparison apples-to-apples

`lib/dataset_splits.py` owns not just the partition but the three knobs that would
otherwise silently make two models score different proteins, and every comparison model
reads its defaults from there:

| constant | value | why it has to be shared |
| --- | --- | --- |
| `DEFAULT_VAL_FOLD` | 0 | the fold `scripts/train_residue_classifier.py` hardcodes |
| `MAX_LENGTH` | 1200 | MapDiff's IPA positional-encoding table is the binding limit; PiFold and DynamicMPNN apply the same cutoff so nobody is scored on proteins the others dropped |
| `SEED` | 42 | anything still drawing from an RNG (weight init, batch order, DynamicMPNN's k-of-pool conformer draw) |

Cutting the same split is necessary but not sufficient: each model additionally drops
whatever it cannot featurize (no deposited RCSB entry, a DSSP failure, too few usable
conformers). So every run logs `split_counts` and `test_split_ids` to its W&B summary --
diff those across runs to confirm the held-out sets really are the same proteins.

The trajectory store and split index csv are staged on Pelican and transferred in flat
(see `pifold.sub`'s `transfer_input_files`); `--data_root ./` points
`API/relaxed_dataset.py` at the job's scratch dir, where they land. The store is read
*only* for the reference residue sequences.

## 4. Metrics logged

`main.py` logs to one W&B run per dataset:

- per-epoch train/valid loss and perplexity
- held-out perplexity and sequence recovery (median/mean/std), per upstream's
  `ProDesign.test_one_epoch`
- **scRMSD** on the final pass: each held-out design is folded with ESMFold
  (`lib/scrmsd.py`) and Kabsch-RMSD'd against that protein's ground-truth relaxed
  backbone — the same self-consistency metric `comparisons/{MapDiff,DynamicMPNN}`
  compute, so results are directly comparable. Pass `--scrmsd 0` to skip it.

W&B auth comes from `api_keys.txt` (line 1 = HF token, last line = W&B key), which
`run_pifold.sh` splits into `wandb_api.txt`.

## 5. Submit

```bash
mkdir -p logs
condor_submit pifold.sub                     # ATLAS
condor_submit pifold.sub DS_NAME=MDCATH \
    DS_H5='$(ResearchDrive)/mdcath_spinet_320_0.h5' \
    DS_CSV='$(ResearchDrive)/mdcath_320_0_topology_split.csv'
```

Edit the `arguments` line in `pifold.sub` to change hyperparameters — they're
forwarded straight to `main.py` (see `parser.py` for all options, e.g.
`--num_encoder_layers`, `--lr`).

Monitor with `condor_q` / `condor_watch_q`; logs land in `logs/`.
Results are copied back to `results/<DS_NAME>_<ClusterId>_<ProcId>/` on job exit.
