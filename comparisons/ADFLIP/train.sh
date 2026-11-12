#!/bin/bash
# Entry point run inside the ADFLIP Docker container by HTCondor.
#
# Everything after the script name is forwarded straight to train.py, e.g.:
#
#   ./train.sh --data-name MDCATH --traj-h5 ./mdcath_spinet_320_0_v2.h5 --k 20
#
# Unlike eval_pretrained.sh this does not need PIPPack's weights: side-chain repacking belongs to
# the iterative sampler, and training never samples (train.py builds the flow model without it).
set -euo pipefail

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# Created BEFORE anything can fail: train.sub lists both in transfer_output_files, and a missing
# output directory holds the job on "No such file or directory" without transferring the logs
# that say what actually went wrong.
mkdir -p logs/ weights/ results/

# api_keys.txt: line 1 = HF_TOKEN, last line = WANDB_KEY (see the other comparisons' train.sh).
if [[ -f api_keys.txt ]]; then
    export HF_TOKEN=$(head -n 1 api_keys.txt)
    tail -n 1 api_keys.txt > wandb_api.txt
fi

if [[ ! -d ADFLIP ]]; then
    if [[ -f ADFLIP.tar.gz ]]; then
        echo ">>> extracting ADFLIP.tar.gz"
        tar -xzf ADFLIP.tar.gz
    else
        echo ">>> no ADFLIP/ checkout and no ADFLIP.tar.gz to extract" >&2
        exit 1
    fi
fi

# Fine-tuning starts from the released checkpoint, which lives in the tarball rather than the
# upstream git tree. Checking for it here turns a missing-weights tarball into an immediate,
# readable failure instead of a traceback after the data has been indexed.
if [[ " $* " != *" --from-scratch "* ]]; then
    if [[ ! -f ADFLIP/results/weights/ADFLIP_v1.pt ]]; then
        echo ">>> missing ADFLIP/results/weights/ADFLIP_v1.pt -- rebuild ADFLIP.tar.gz with the" >&2
        echo ">>> released weights in it, or pass --from-scratch" >&2
        exit 1
    fi
fi

export PYTHONPATH="$PWD/ADFLIP:${PYTHONPATH:-}"

echo ">>> running: python train.py $*"
exec python train.py "$@"
