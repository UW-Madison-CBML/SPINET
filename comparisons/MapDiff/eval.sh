#!/bin/bash
# Entry point run inside the MapDiff Docker container by HTCondor (or
# locally), for evaluating a trained Prior_Diff checkpoint on a dataset's
# held-out test split with eval_relaxed.py.
#
# Extracts a pristine MapDiff.tar.gz, overlays the dataset/wandb/scRMSD support
# on top of it, copies eval_relaxed.py + the checkpoint/config/wandb key/index
# files transferred in flat, and runs eval_relaxed.py. Everything after
# 'eval.sh' is forwarded straight to it, e.g.:
#
#   ./eval.sh --checkpoint ./checkpoint.pt --config ./config.yaml --ds-name mdcath
#
set -euo pipefail

# The IPA stack's pair representation is (B, L, L, C), so its transient tensors are both huge
# and wildly varying in size from batch to batch (padding is to the longest protein *in the
# batch*). That fragments the caching allocator badly enough to OOM on nominally free memory
# -- the failure in logs/train_atlas_6211405_0.err asked for 8.45 GiB with 5.47 GiB free.
# expandable_segments lets PyTorch grow one segment instead of stranding many fixed-size ones.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# api_keys.txt: line 1 = HF_TOKEN, last line = WANDB_KEY (see
# ../../scripts/train_residue_classifier.sh). `lib.stats_utils.init_wandb` wants a file
# holding *only* the W&B key, so split it out here -- conf/wandb/basic.yaml's
# `key_file: ./wandb_api.txt` then resolves without any submit-file override.
if [[ -f api_keys.txt ]]; then
    export HF_TOKEN=$(head -n 1 api_keys.txt)
    tail -n 1 api_keys.txt > wandb_api.txt
fi

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [[ ! -d MapDiff ]]; then
    echo ">>> extracting MapDiff.tar.gz"
    tar -xzf MapDiff.tar.gz
fi

echo ">>> overlaying dataset support + wandb/scRMSD wiring onto pristine MapDiff/"
cp -r conf dataloader model data MapDiff/
for f in eval_relaxed.py *.pt config.yaml wandb_api.txt stats_utils.py scrmsd.py relaxed_pdb.py dataset_splits.py \
         atlas_data.h5 atlas_cross_val_index.csv mdcath_spinet_320_0.h5 mdcath_320_0_topology_split.csv; do
    [[ -f "$f" ]] && cp "$f" MapDiff/
done

echo ">>> running: python eval_relaxed.py $*"
cd MapDiff
exec python eval_relaxed.py "$@"
