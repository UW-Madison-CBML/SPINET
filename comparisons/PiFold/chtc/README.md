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

## 3. (Optional) stage a dataset

For CATH/TS50/ATLAS data too large to transfer per-job, tar it as
`data.tar.gz` (expanding to a top-level `data/` folder) and either:
- add it to `transfer_input_files` in `pifold.sub`, or
- put it under `/staging/<username>/` on CHTC and reference that path in
  `transfer_input_files` instead (recommended for large datasets — see
  CHTC's docs on the `/staging` filesystem).

`run_pifold.sh` auto-extracts `data.tar.gz` if it's present in the job's
scratch directory.

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
