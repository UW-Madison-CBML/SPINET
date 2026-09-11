# Running PiFold on CHTC

This bundles a Docker image with PiFold's dependencies (CUDA 11.8, PyTorch
2.1.2, torch_scatter, biopython, numpy, tqdm) plus an HTCondor submit file
to run training/inference as a GPU job on CHTC.

Code and data are **not** baked into the image — they're transferred into
the job by HTCondor, so you don't need to rebuild/repush the image every
time PiFold's code changes.

## 1. Build and push the image

Docker must run somewhere with internet access and a Docker Hub account
(CHTC execute nodes pull the image from a registry, they can't use a
locally-built image directly):

```bash
cd chtc
./build_and_push.sh <your_dockerhub_username>
```

## 2. Point the submit file at your image

Edit `pifold.sub` and set:

```
docker_image = <your_dockerhub_username>/pifold:latest
```

## 3. Dataset

`API.atlas_dataset.ATLAS` reads directly from the same shared ATLAS store every
other model in this repo uses: `atlas_data.h5` (one representative frame per
protein, plus the full trajectory) and `atlas_cross_val_index.csv` (the `pdb` ->
`cross_val` fold assignment and each protein's pre-selected `random_indices`
frame) -- the same store/split `scripts/train_residue_classifier.py` and
`comparisons/{gvp-pytorch,DynamicMPNN}/train.py` use, instead of a pre-split
folder of raw PDB files. **Fold 0 is held out as the `valid` split, and reused
as `test` too** (PiFold wants train/valid/test; ATLAS only has one held-out
fold).

`pifold.sub` transfers both files in flat (staged on Pelican) via
`transfer_input_files`; `--data_root ./` (set in `pifold.sub`'s `arguments`)
points `API.atlas_dataset.ATLAS` at the job's scratch dir, where they land.

For other/smaller datasets (e.g. CATH, TS50), tar them as `data.tar.gz`
(expanding to a top-level `data/` folder) and add them to
`transfer_input_files` instead, or stage them under `/staging/<username>/`
the same way.

## 4. Submit

```bash
cd chtc
mkdir -p logs
condor_submit pifold.sub
```

Edit the `arguments` line in `pifold.sub` to change hyperparameters —
they're forwarded straight to `main.py` (see `../parser.py` for all
options, e.g. `--data_name ATLAS`, `--num_encoder_layers`, `--lr`, etc.).

Monitor with `condor_q` / `condor_watch_q`; logs land in `chtc/logs/`.
Results are copied back to `results/<ClusterId>_<ProcId>/` on job exit.
