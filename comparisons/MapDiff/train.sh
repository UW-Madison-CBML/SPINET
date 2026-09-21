#!/bin/bash
# Entry point run inside the MapDiff Docker container by HTCondor (or
# locally) to train MapDiff on one of our datasets end-to-end (mask-prior IPA
# pretraining + denoising diffusion training, see train.py).
#
# Extracts a *pristine* checkout of peizhenbai/MapDiff (MapDiff.tar.gz) and
# copies this folder's dataset/wandb additions on top of it -- MapDiff/
# itself is never modified, so re-cloning/re-pulling it always works. See
# README.md.
#
# $DS_NAME (atlas, the default, or mdcath) selects which dataset to featurize;
# ATLAS and mdCATH are trained and evaluated separately, one job each, matching
# scripts/train_residue_classifier.py's --ds-name. Everything after 'train.sh'
# is forwarded straight to train.py, e.g.:
#   DS_NAME=mdcath ./train.sh dataset=mdcath wandb.use=True
#
# $STRUCTURE_SOURCE ('relaxed', the default, or 'frame') picks what MapDiff sees: each
# protein's deposited RCSB entry, or one random frame of its MD trajectory. It selects both
# the --structure-source the graphs are featurized with and the dataset.process_root they are
# read back from, so the two can never disagree. Each source gets its own directory, so
# switching back and forth does not re-featurize:
#   STRUCTURE_SOURCE=frame ./train.sh wandb.use=True
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DS_NAME="${DS_NAME:-atlas}"
STRUCTURE_SOURCE="${STRUCTURE_SOURCE:-relaxed}"
case "${STRUCTURE_SOURCE}" in
    relaxed) SOURCE_SUFFIX="" ;;
    frame)   SOURCE_SUFFIX="_frame" ;;
    *) echo "STRUCTURE_SOURCE must be 'relaxed' or 'frame', got '${STRUCTURE_SOURCE}'" >&2; exit 1 ;;
esac
PROCESS_ROOT="./surffold_data/${DS_NAME}${SOURCE_SUFFIX}_process"
MARGINAL_OUT="./surffold_data/train_marginal_x_${DS_NAME}${SOURCE_SUFFIX}.pt"

if [[ -f api_keys.txt ]]; then
    export HF_TOKEN=$(head -n 1 api_keys.txt)
    tail -n 1 api_keys.txt > wandb_api.txt
fi

python -m ruff check . --select F821,E9 || exit 1

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [[ ! -d MapDiff ]]; then
    echo ">>> extracting MapDiff.tar.gz"
    tar -xzf MapDiff.tar.gz
fi

echo ">>> overlaying dataset support + wandb wiring onto pristine MapDiff/"
cp -r conf dataloader model data train.py trainer.py MapDiff/
for f in api_keys.txt wandb_api.txt stats_utils.py relaxed_pdb.py traj_frames.py dataset_splits.py \
         atlas_data.h5 atlas_cross_val_index.csv mdcath_spinet_320_0.h5 mdcath_320_0_topology_split.csv; do
    [[ -f "$f" ]] && cp "$f" MapDiff/
done

cd MapDiff
# `test/` is checked too, not just `train/`: a scratch dir featurized before the test split
# existed has train/ and validation/ but no test/, and generate_graph_relaxed.py skips any
# protein already written, so re-running it is cheap.
if [[ ! -d "${PROCESS_ROOT}/train" || ! -d "${PROCESS_ROOT}/test" ]]; then
    echo ">>> featurizing ${DS_NAME} ${STRUCTURE_SOURCE} structures into MapDiff graphs (runs DSSP once per protein)"
    python data/generate_graph_relaxed.py --ds-name "${DS_NAME}" \
        --structure-source "${STRUCTURE_SOURCE}" \
        --save-root "${PROCESS_ROOT}" --marginal-out "${MARGINAL_OUT}"
fi

# Passed before "$@" so an explicit override on the command line still wins.
echo ">>> running: python train.py dataset.process_root=${PROCESS_ROOT} dataset.marginal_train_dir=${MARGINAL_OUT} $*"
exec python train.py \
    "dataset.process_root=${PROCESS_ROOT}" \
    "dataset.marginal_train_dir=${MARGINAL_OUT}" \
    "$@"
