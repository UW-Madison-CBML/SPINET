#!/bin/bash
# Entry point run inside the PiFold Docker container by HTCondor (or locally),
# for evaluating a trained checkpoint on the ATLAS validation split.
#
# Extracts a pristine PiFold.tar.gz, overlays ATLAS-dataset support on top of
# it, copies eval_atlas.py + the checkpoint/model_param.json/wandb key
# transferred in flat, and runs eval_atlas.py. Everything after 'run_eval.sh'
# in the submit file's `arguments` line is forwarded straight to it, e.g.:
#
#   arguments = --run_dir . --data_root ./surffold_data/ --wandb_key_file ./wandb_api.txt
#
set -euo pipefail

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [[ ! -d PiFold ]]; then
    echo ">>> extracting PiFold.tar.gz"
    tar -xzf PiFold.tar.gz
fi

echo ">>> overlaying ATLAS dataset support onto pristine PiFold/"
cp -r overlay/. PiFold/
for f in eval_atlas.py checkpoint.pth model_param.json wandb_api.txt stats_utils.py; do
    [[ -f "$f" ]] && cp "$f" PiFold/
done

if [[ -f surffold_data.tar.gz && ! -d PiFold/surffold_data ]]; then
    echo ">>> extracting surffold_data.tar.gz"
    tar -xzf surffold_data.tar.gz -C PiFold/
fi

cd PiFold
echo ">>> running: python eval_atlas.py $*"
exec python eval_atlas.py "$@"
