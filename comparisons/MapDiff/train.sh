#!/bin/bash
# Entry point run inside the MapDiff Docker container by HTCondor (or
# locally) to train MapDiff on one of our datasets end-to-end (mask-prior IPA
# pretraining + denoising diffusion training, see train.py).
#
# Extracts a *pristine* checkout of peizhenbai/MapDiff (MapDiff.tar.gz) and
# copies this folder's dataset/wandb/scRMSD additions on top of it -- MapDiff/
# itself is never modified, so re-cloning/re-pulling it always works. See
# README.md.
#
# $DS_NAME (atlas, the default, or mdcath) selects which dataset to featurize;
# ATLAS and mdCATH are trained and evaluated separately, one job each, matching
# scripts/train_residue_classifier.py's --ds-name. Everything after 'train.sh'
# is forwarded straight to train.py, e.g.:
#   DS_NAME=mdcath ./train.sh dataset=mdcath wandb.use=True
set -euo pipefail

DS_NAME="${DS_NAME:-atlas}"

# api_keys.txt: line 1 = HF_TOKEN, last line = WANDB_KEY (see
# ../../scripts/train_residue_classifier.sh). `lib.stats_utils.init_wandb` wants a file
# holding *only* the W&B key, so split it out here -- conf/wandb/basic.yaml's
# `key_file: ./wandb_api.txt` then resolves without any submit-file override.
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

echo ">>> overlaying dataset support + wandb/scRMSD wiring onto pristine MapDiff/"
cp -r conf dataloader model data train.py trainer.py MapDiff/
for f in api_keys.txt wandb_api.txt stats_utils.py scrmsd.py relaxed_pdb.py dataset_splits.py \
         atlas_data.h5 atlas_cross_val_index.csv mdcath_spinet_320_0.h5 mdcath_320_0_topology_split.csv; do
    [[ -f "$f" ]] && cp "$f" MapDiff/
done

cd MapDiff
if [[ ! -d "surffold_data/${DS_NAME}_process/train" ]]; then
    echo ">>> featurizing ${DS_NAME} relaxed (deposited) structures into MapDiff graphs (downloads from RCSB, runs DSSP once per protein)"
    python data/generate_graph_relaxed.py --ds-name "${DS_NAME}"
fi

echo ">>> running: python train.py $*"
exec python train.py "$@"
