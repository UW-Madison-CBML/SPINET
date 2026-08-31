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

The ATLAS-derived dataset (first frame of each MD simulation, pre-split
into `train/`, `valid/`, `test/` folders of PDB files) is staged at:

```
/staging/groups/bhaskar_group/sheaf_protein_folding/surffold_data.tar.gz
```

`pifold.sub` already references this path in `transfer_input_files`, so
HTCondor copies it into the job's scratch directory. `run_pifold.sh`
auto-extracts it into a top-level `surffold_data/` folder if not already
present, and `--data_root ./surffold_data/` (set in `pifold.sub`'s
`arguments`) points `API.atlas_dataset.ATLAS` at it.

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
