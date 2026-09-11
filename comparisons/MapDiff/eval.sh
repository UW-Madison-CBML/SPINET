#!/bin/bash
# Entry point run inside the MapDiff Docker container by HTCondor (or
# locally), for evaluating a trained Prior_Diff checkpoint on the ATLAS
# validation/test split (fold 0) with eval_atlas.py.
#
# Extracts a pristine MapDiff.tar.gz, overlays ATLAS/wandb/scRMSD support on
# top of it, copies eval_atlas.py + the checkpoint/config/wandb key/ATLAS
# data transferred in flat, and runs eval_atlas.py. Everything after
# 'eval.sh' is forwarded straight to it, e.g.:
#
#   ./eval.sh --checkpoint ./checkpoint.pt --config ./config.yaml --atlas-h5 ./atlas_data.h5 --cross-val-csv ./atlas_cross_val_index.csv
#
set -euo pipefail

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

echo ">>> overlaying ATLAS dataset support + wandb/scRMSD wiring onto pristine MapDiff/"
cp -r conf dataloader model data MapDiff/
for f in eval_atlas.py *.pt config.yaml wandb_api.txt stats_utils.py scrmsd.py load_dynamics.py atlas_frame_pdb.py atlas_data.h5 atlas_cross_val_index.csv; do
    [[ -f "$f" ]] && cp "$f" MapDiff/
done

echo ">>> running: python eval_atlas.py $*"
cd MapDiff
exec python eval_atlas.py "$@"
