#!/bin/bash
# Entry point for zero-shot scoring of the released ADFLIP weights on our held-out split.
#
# Unlike PiFold/MapDiff there is no overlay step: the ADFLIP checkout lives at ADFLIP/ beside
# this script and eval_pretrained.py adds it to sys.path itself, then chdirs into it (upstream
# resolves PIPPack's weights and its sample-save directory relative to cwd). lib/ is copied in
# next to the script so `import dataset_splits` resolves the same way it does for the others.
#
# Everything after the script name is forwarded straight through, e.g.:
#
#   ./eval_pretrained.sh --data-name MDCATH --traj-h5 ./mdcath_spinet_320_0.h5 --n-states 8
#
set -euo pipefail

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

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

# The adaptive sampler repacks side chains every step, so PIPPack's weights must be there --
# they are part of the checkout, but a partial transfer is easy to miss and the failure would
# otherwise only surface after the model is loaded.
for f in inference.pickle pippack_model_1_ckpt.pt pippack_model_1_config.pickle; do
    if [[ ! -f "ADFLIP/PIPPack/model_weights/$f" ]]; then
        echo ">>> missing ADFLIP/PIPPack/model_weights/$f -- side-chain packing cannot run" >&2
        exit 1
    fi
done

export PYTHONPATH="$PWD/ADFLIP:${PYTHONPATH:-}"

echo ">>> running: python eval_pretrained.py $*"
exec python eval_pretrained.py "$@"
