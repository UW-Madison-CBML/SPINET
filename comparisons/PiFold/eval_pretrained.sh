#!/bin/bash
# Entry point run inside the PiFold Docker container by HTCondor (or locally) for
# eval_pretrained.sub: zero-shot scoring of released PiFold weights on our held-out split.
#
# Same extract + overlay as train.sh (pristine PiFold.tar.gz, this folder's API/ + parser.py +
# eval_pretrained.py and lib/ copied on top), then runs eval_pretrained.py. Everything after
# 'eval_pretrained.sh' is forwarded straight to it, e.g.:
#
#   ./eval_pretrained.sh --checkpoint ./results.pt --data_name MDCATH --data_root ./
#
set -euo pipefail

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# api_keys.txt: line 1 = HF_TOKEN, last line = WANDB_KEY (see train.sh).
if [[ -f api_keys.txt ]]; then
    export HF_TOKEN=$(head -n 1 api_keys.txt)
    tail -n 1 api_keys.txt > wandb_api.txt
fi

if [[ ! -d PiFold ]]; then
    echo ">>> extracting PiFold.tar.gz"
    tar -xzf PiFold.tar.gz
fi

# numpy >= 1.24 removed np.int -- same patch as train.sh.
sed -i 's/\.astype(np\.int)/.astype(int)/' PiFold/API/featurizer.py PiFold/API/dataloader_gtrans.py
if grep -rn 'np\.int)' PiFold/API/featurizer.py PiFold/API/dataloader_gtrans.py; then
    echo ">>> np.int patch did not apply -- upstream changed these lines" >&2
    exit 1
fi

echo ">>> overlaying dataset support + eval script onto pristine PiFold/"
cp -r API parser.py eval_pretrained.py PiFold/
for f in wandb_api.txt *.pt *.pth relaxed_pdb.py traj_frames.py dataset_splits.py stats_utils.py scrmsd.py \
         atlas_data.h5 atlas_cross_val_index.csv "${DS_H5_FILE:-}" "${DS_CSV_FILE:-}"; do
    [[ -n "$f" && -f "$f" ]] && cp "$f" PiFold/
done

cd PiFold
mkdir -p results

echo ">>> running: python eval_pretrained.py $*"
exec python eval_pretrained.py "$@"
