#!/bin/bash
# Entry point run inside the MapDiff Docker container by HTCondor (or locally),
# for evaluating a trained Prior_Diff checkpoint on the ATLAS test split.
#
# Extracts a pristine MapDiff.tar.gz, overlays ATLAS-dataset support on top of
# it, copies eval_atlas.py + the checkpoint/config/wandb key transferred in
# flat, and runs eval_atlas.py. Everything after 'run_eval.sh' in the submit
# file's `arguments` line is forwarded straight to it, e.g.:
#
#   arguments = --checkpoint ./checkpoint.pt --config ./config.yaml --data_root ./surffold_data/ --wandb_key_file ./wandb_api.txt
#
set -euo pipefail

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [[ ! -d MapDiff ]]; then
    echo ">>> extracting MapDiff.tar.gz"
    tar -xzf MapDiff.tar.gz
fi

echo ">>> overlaying ATLAS dataset support onto pristine MapDiff/"
cp -r overlay/. MapDiff/
for f in eval_atlas.py checkpoint.pt config.yaml wandb_api.txt stats_utils.py; do
    [[ -f "$f" ]] && cp "$f" MapDiff/
done

if [[ -f surffold_data.tar.gz && ! -d MapDiff/surffold_data ]]; then
    echo ">>> extracting surffold_data.tar.gz"
    tar -xzf surffold_data.tar.gz -C MapDiff/
fi

cd MapDiff
if [[ ! -f surffold_data/train_marginal_x_atlas.pt ]]; then
    # The resolved training config points marginal_dist_path at this file, but it's
    # only ever computed on the fly during training (run_train.sh) and training's
    # transfer_output_files never ships it back out -- so it doesn't exist yet here.
    # Train-split only: eval doesn't need validation/test re-featurized.
    echo ">>> computing train-split amino-acid marginal (needed by Prior_Diff's marginal noise model)"
    python data/generate_graph_atlas.py --pdb_root surffold_data --save_root surffold_data/atlas_process \
        --splits train --marginal_out surffold_data/train_marginal_x_atlas.pt
fi

echo ">>> running: python eval_atlas.py $*"
exec python eval_atlas.py "$@"
