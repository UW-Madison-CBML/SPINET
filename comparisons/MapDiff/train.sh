#!/bin/bash
# Entry point run inside the MapDiff Docker container by HTCondor (or
# locally) to train MapDiff on ATLAS end-to-end (mask-prior IPA pretraining
# + denoising diffusion training, see train.py).
#
# Extracts a *pristine* checkout of peizhenbai/MapDiff (MapDiff.tar.gz) and
# copies this folder's ATLAS/wandb/scRMSD additions on top of it -- MapDiff/
# itself is never modified, so re-cloning/re-pulling it always works. See
# README.md.
#
# Everything after 'train.sh' is forwarded straight to train.py, e.g.:
#   ./train.sh dataset=atlas wandb.use=True prior_model.path=./ipa_checkpoint.pt
set -euo pipefail

python -m ruff check . --select F821,E9 || exit 1

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [[ ! -d MapDiff ]]; then
    echo ">>> extracting MapDiff.tar.gz"
    tar -xzf MapDiff.tar.gz
fi

echo ">>> overlaying ATLAS dataset support + wandb/scRMSD wiring onto pristine MapDiff/"
cp -r conf dataloader model data train.py trainer.py MapDiff/
for f in wandb_api.txt stats_utils.py scrmsd.py load_dynamics.py atlas_frame_pdb.py atlas_data.h5 atlas_cross_val_index.csv; do
    [[ -f "$f" ]] && cp "$f" MapDiff/
done

cd MapDiff
if [[ ! -d surffold_data/atlas_process/train ]]; then
    echo ">>> featurizing ATLAS structures (one representative frame per protein) into MapDiff graphs (runs DSSP once per protein)"
    python data/generate_graph_atlas.py --atlas-h5 atlas_data.h5 --cross-val-csv atlas_cross_val_index.csv \
        --save-root surffold_data/atlas_process --marginal-out surffold_data/train_marginal_x_atlas.pt
fi

echo ">>> running: python train.py $*"
exec python train.py "$@"
