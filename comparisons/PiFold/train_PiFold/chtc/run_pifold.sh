#!/bin/bash
# Entry point run inside the PiFold Docker container by HTCondor (or locally).
#
# Extracts a *pristine* checkout of A4Bio/PiFold (PiFold.tar.gz) and copies the
# ATLAS-dataset-support + wandb overlay on top of it -- PiFold/ itself is never
# modified, so re-cloning it always works. See ../README.md.
#
# Everything after 'run_pifold.sh' in the submit file's `arguments` line is
# forwarded straight to main.py, e.g.:
#
#   arguments = --data_name ATLAS --data_root ./surffold_data/ --batch_size 8 --epoch 100
#
set -euo pipefail

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [[ ! -d PiFold ]]; then
    echo ">>> extracting PiFold.tar.gz"
    tar -xzf PiFold.tar.gz
fi

echo ">>> overlaying ATLAS dataset support + wandb wiring onto pristine PiFold/"
cp -r overlay/. PiFold/
for f in wandb_api.txt stats_utils.py; do
    [[ -f "$f" ]] && cp "$f" PiFold/
done

if [[ -f surffold_data.tar.gz && ! -d PiFold/surffold_data ]]; then
    echo ">>> extracting surffold_data.tar.gz"
    tar -xzf surffold_data.tar.gz -C PiFold/
fi

mkdir -p PiFold/results
cd PiFold
echo ">>> running: python main.py $*"
exec python main.py "$@"
