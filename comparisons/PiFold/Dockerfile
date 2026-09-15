# PiFold runtime environment for CHTC (Docker universe).
#
# This image only provides the software environment. The PiFold source code and
# datasets are NOT baked in here -- CHTC jobs pull them in at runtime via
# HTCondor file transfer / the /staging area, so the image stays small and
# doesn't need to be rebuilt every time the code changes.
#
# Build:  docker build -t <dockerhub_user>/pifold:latest -f Dockerfile .
# Push:   docker push <dockerhub_user>/pifold:latest
# (see build_and_push.sh for a one-shot script)

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Scope of the dependency list below is everything run_pifold.sh touches:
#
#  * the pristine upstream PiFold/ checkout (PiFold.tar.gz) -- torch,
#    torch_scatter, numpy, tqdm, and nothing else;
#  * this folder's overlay (main.py, parser.py, API/relaxed_dataset.py);
#  * the shared ../../lib modules the overlay imports -- scrmsd.py,
#    relaxed_pdb.py, dataset_splits.py, stats_utils.py.
#
# Mapped back to the imports that need them:
#   biopython   -- relaxed_pdb.py (Bio.Align/PDB/Data.IUPACData) and
#                  stats_utils.py (Bio.SeqUtils.seq1)
#   requests    -- relaxed_pdb.py, downloading deposited structures from RCSB
#   h5py        -- relaxed_pdb.reference_seqs_from_h5 (trajectory store)
#   pandas      -- dataset_splits.py, reading the split index csv
#   transformers/accelerate/scipy -- lib/scrmsd.py's ESMFold scRMSD metric
#   matplotlib + scikit-learn     -- stats_utils.to_log_dict's confusion matrix
#   wandb       -- main.py's --wandb_key_file run logging
#
# Deliberately NOT here: mdtraj. relaxed_pdb.py re-exports BACKBONE_ATOMS
# precisely so this image never has to import scripts/load_dynamics.py.

# torch_scatter must be built/pinned against the exact torch+CUDA combo
# above (2.1.2 / cu118) -- see https://data.pyg.org/whl/ for the wheel index.
RUN pip install --no-cache-dir \
        torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.2+cu118.html \
    && pip install --no-cache-dir \
        numpy==1.26.4 \
        biopython==1.83 \
        tqdm==4.66.2 \
        requests \
        h5py \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        "transformers==4.55.4" \
        accelerate==1.10.1

# wandb is installed on its own, AFTER the block above, for the same reason as
# in ../DynamicMPNN: listed alongside everything else, pip is free to backtrack
# it down to a release whose `apikey.write_key` enforces `len(key) == 40`, and
# W&B now issues longer keys ("API key must be 40 characters long, yours was
# 86" killed a job that way). 0.22.3 is the first line to relax that check.
RUN pip install --no-cache-dir wandb==0.22.3

# Fail the build (rather than a queued CHTC job that has already claimed a GPU)
# if the resolved wheels don't actually hang together. The failure modes this
# catches, all of which have bitten the comparison images before:
#
#  * transformers silently disabling its torch backend. transformers >=4.56
#    calls torch.utils._pytree.register_pytree_node, which only exists in
#    torch >=2.2, and transformers 5.x hard-requires torch >=2.5 -- with torch
#    pinned at 2.1.2 either one degrades to a tokenizers-only install and
#    EsmForProteinFolding then raises "requires the PyTorch library but it was
#    not found" deep inside scrmsd.load_esmfold(). 4.55.4 is the newest release
#    that works against torch 2.1.2.
#  * numpy 2.x getting pulled in. torch 2.1.2 is compiled against numpy 1.x and
#    dies at import with "_ARRAY_API not found".
#  * scipy missing -- ESMFold silently falls back to an approximate init.
#  * accelerate missing -- scrmsd.load_esmfold passes low_cpu_mem_usage=True,
#    which transformers 4.x refuses without it.
RUN python -c "\
import numpy, scipy, sklearn, pandas, h5py, requests, tqdm, Bio, torch, transformers, accelerate, wandb; \
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot; \
from transformers.utils import is_torch_available, is_scipy_available; \
from transformers import AutoTokenizer, EsmForProteinFolding; \
from sklearn.metrics import ConfusionMatrixDisplay; \
from Bio import Align, PDB; from Bio.SeqUtils import seq1; \
assert numpy.__version__.startswith('1.'), 'numpy must be 1.x for torch 2.1.2, got ' + numpy.__version__; \
assert is_torch_available(), 'transformers ' + transformers.__version__ + ' disabled its torch backend against torch ' + torch.__version__; \
assert is_scipy_available(), 'scipy missing: ESMFold falls back to an approximate init'; \
EsmForProteinFolding.from_pretrained; \
print('>>> ok: torch', torch.__version__, '| transformers', transformers.__version__, '| numpy', numpy.__version__, '| wandb', wandb.__version__)"

# torch_scatter ships compiled kernels that a bare import does not touch, and
# upstream PiFold's prodesign_module.py calls scatter_sum/softmax/mean on every
# forward pass -- run one on real (CPU) tensors so a torch/torch_scatter ABI
# mismatch shows up here instead of mid-epoch.
RUN python -c "\
import torch; from torch_scatter import scatter_sum, scatter_softmax, scatter_mean; \
idx = torch.tensor([0, 0, 1, 1]); \
assert torch.allclose(scatter_sum(torch.ones(4), idx), torch.tensor([2., 2.])); \
scatter_softmax(torch.randn(4), idx); scatter_mean(torch.randn(4), idx); \
print('SCATTER KERNEL OK')"

# The key-length rule lives inside wandb's own source, not in its metadata, so
# the version pin alone does not prove the fix is present. Read the installed
# `write_key` back and require the relaxed wording.
RUN python -c "\
import wandb; from wandb.sdk.lib import apikey; \
src = __import__('inspect').getsource(apikey.write_key); \
assert 'at least 40' in src, 'wandb reintroduced the exact-40-character API key check'; \
print('WANDB LONG KEY OK')"

WORKDIR /workspace
