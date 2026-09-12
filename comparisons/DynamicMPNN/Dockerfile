# Build:  docker build -t <dockerhub_user>/dynamicmpnn:latest -f chtc/Dockerfile .
# Push:   docker push <dockerhub_user>/dynamicmpnn:latest

# Use official miniconda3 base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# The miniconda3 base ships Anaconda's `defaults` channels (pkgs/main, pkgs/r).
# Those have no sharded repodata, so every solve falls back to flat repodata and
# burns ~30s fetching and parsing records -- then hands the solver a much larger,
# partly-duplicate search space to backtrack through. Every package this image
# needs (mamba included) lives on conda-forge, so `--override-channels` drops
# defaults outright rather than merely ranking it lower. Strict priority keeps
# the solver from mixing channels if one is ever added back.
ENV CONDA_CHANNEL_PRIORITY=strict

# Update conda and install mamba for faster dependency resolution
RUN conda install -n base -y --override-channels -c conda-forge mamba && \
    conda clean --all -f -y

## Mirrors DynamicMPNN/environment.yml, plus the deps that file omits but the
## code actually imports. Scope is everything train.sh touches: all of
## DynamicMPNN/src (including its scripts/ subpackages), the repo's lib/ and
## scripts/ modules train.py pulls in via sys.path, and train.py itself. These
## are all *direct* imports -- listing them explicitly means an upstream
## dependency bump can't silently drop a module out from under the job.
## Deliberately NOT here: alphafold/openmm/pdbfixer/torch-sparse, which belong
## to unrelated repo entrypoints (train_residue_classifier.py, the sheaf model,
## the GROMACS prep script), not to DynamicMPNN.
##
## numpy/pandas are left loose on purpose. graphein==1.7.6 hard-pins
## `numpy<1.24.0` / `pandas<2.0.0` / `typing-extensions==4.5.0`, so the pip step
## below quietly downgrades all three. Restating those caps here instead looks
## tidier but wedges the conda solve: conda-forge scipy needs `numpy>=1.23.5`,
## which leaves 1.23.5 as the single satisfiable numpy. The conda-forge builds
## used here declare `numpy>=1.21,<3` at runtime (post numpy-2 ABI migration),
## so they tolerate the pip downgrade fine.
RUN mamba create -n dynamicmpnn -y \
    --override-channels -c conda-forge \
    python=3.10 \
    pip'>=24' \
    'numpy<2' \
    pandas \
    scipy \
    scikit-learn \
    biopython \
    biopandas \
    matplotlib \
    omegaconf \
    hydra-core \
    loguru \
    rich \
    python-dotenv \
    beartype \
    jaxtyping \
    einops \
    tqdm \
    joblib \
    h5py \
    mdtraj \
    requests \
    urllib3 \
    aiofiles \
    aiohttp \
    ruff \
    pytest && \
    mamba clean --all -f -y

# Activate the environment and install pip dependencies
SHELL ["conda", "run", "-n", "dynamicmpnn", "/bin/bash", "-c"]

# Pin PyTorch to a CUDA 12.1 build and fetch torch-scatter/torch-cluster from
# PyG's wheel index built for that exact torch+cuda combo (must match
# torch-${TORCH_VERSION}+cu121 exactly -- see https://data.pyg.org/whl/).
#
# torch-cluster is here despite nothing importing it by name: it is what backs
# `torch_geometric.nn.pool.knn_graph`, which is how graphein's `compute_edges`
# builds the `knn_32` edges that configs/features/ca_bb_single_chain.yaml asks
# for. Drop it and the failure lands on the first batch, not on import.
# torch-sparse is deliberately absent -- nothing in graphein or DynamicMPNN
# imports it, and torch-geometric 2.6 only reaches for it in sparse-tensor
# paths this featuriser never takes.
ENV TORCH_VERSION=2.1.2
ENV CUDA_TAG=cu121

RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_TAG} && \
    pip install --no-cache-dir \
    torch-scatter==2.1.2 torch-cluster==1.6.3 torch-geometric==2.6.1 \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html

## Every pin below is load-bearing, not cosmetic:
##   pytorch-lightning  -- 2.5+ requires `typing-extensions>4.5.0`, which is
##     unsatisfiable against graphein's `typing-extensions==4.5.0`. Left as
##     `>=2.0` pip has to backtrack out of that conflict on its own; 2.4.0 is the
##     newest release that asks for `>=4.4.0` and so resolves cleanly.
##   pydantic -- same `typing-extensions==4.5.0` pin excludes pydantic 2
##     (needs >=4.6.1). graphein 1.7.6 is pydantic-v1 API (`from pydantic import
##     BaseModel, validator`) anyway, so state it instead of making pip find it.
##   transformers -- 5.x requires torch>=2.5 but declares torch only as an
##     extra, so pip installs it happily against the torch 2.1.2 pinned above and
##     `EsmForProteinFolding` then fails at run time. 4.41.2 matches the version
##     the repo's own requirements.txt uses for the same ESMFold scRMSD path.
##   accelerate -- lib/scrmsd.py's `load_esmfold` passes
##     `low_cpu_mem_usage=True`, which transformers 4.x refuses without it.
RUN pip install --no-cache-dir \
    graphein==1.7.6 \
    'pydantic<2' \
    pytorch-lightning==2.4.0 \
    lovely-tensors \
    transformers==4.41.2 \
    accelerate

## wandb is installed on its own, AFTER the block above, and that ordering is
## the whole point. graphein 1.7.6 hard-pins `typing-extensions==4.5.0`; every
## wandb >=0.20 asks for `typing-extensions>=4.8`. Listed together, pip cannot
## satisfy both, so it silently backtracks wandb down to a release old enough to
## predate that requirement -- which is how the v7 image ended up with a wandb
## whose `apikey.write_key` enforces `len(key) == 40` and killed the job with
## "API key must be 40 characters long, yours was 86". W&B now issues longer
## keys; 0.22.3 is the first line to relax that to "at least 40" (and it is what
## comparisons/MapDiff already runs).
##
## Installing it separately lets pip upgrade typing-extensions past graphein's
## pin. pip prints a dependency-conflict warning for graphein and continues --
## typing-extensions is additive and backward compatible, so graphein 1.7.6 runs
## unchanged on 4.8+ (the import check below is what actually proves that).
## pydantic stays at v1: wandb declares only `pydantic<3` and ships
## `wandb/_pydantic/v1_compat.py` for exactly this case.
RUN pip install --no-cache-dir wandb==0.22.3

# Fail the *build* rather than the job. The pip steps above freely downgrade
# packages conda already placed (numpy, pandas, typing-extensions, and whatever
# pip backtracks into), and a missing or broken module otherwise only surfaces
# once a queued CHTC job has already claimed a GPU -- which is exactly how the
# v5 image shipped without `ruff` and died on train.sh's first line.
#
# Import-level only: DynamicMPNN itself is not in the image, it arrives as
# DynamicMPNN.tar.gz via transfer_input_files, so this covers the third-party
# floor everything under train.sh stands on.
RUN python -m ruff --version && \
    python -c "\
import numpy, pandas, scipy, sklearn, matplotlib; matplotlib.use('Agg');\
import matplotlib.pyplot, h5py, mdtraj, requests, urllib3, tqdm, joblib;\
import Bio, biopandas, aiofiles, aiohttp, dotenv, rich, loguru, omegaconf, hydra;\
import beartype, jaxtyping, einops, wandb, lovely_tensors, typing_extensions;\
import torch, torch_geometric, torch_scatter, torch_cluster, pytorch_lightning;\
import graphein, graphein.protein.tensor.data, graphein.protein.tensor.edges;\
from torch_geometric.nn.pool import knn_graph;\
from sklearn.metrics import ConfusionMatrixDisplay;\
from transformers import AutoTokenizer, EsmForProteinFolding;\
print('numpy', numpy.__version__, '| pandas', pandas.__version__);\
print('wandb', wandb.__version__, '| typing-extensions', typing_extensions.__version__);\
print('torch', torch.__version__, torch.version.cuda, '| pyg', torch_geometric.__version__);\
print('DEP CHECK OK')"

# knn_graph needs torch_cluster's compiled kernels, which the bare import above
# does not touch -- run one on real (CPU) tensors so a torch/torch-cluster ABI
# mismatch shows up here instead of mid-epoch.
RUN python -c "\
import torch; from torch_geometric.nn.pool import knn_graph;\
e = knn_graph(torch.randn(64, 3), k=8);\
assert e.shape[0] == 2 and e.shape[1] == 64 * 8, e.shape;\
print('KNN KERNEL OK', tuple(e.shape))"

# The key-length rule lives inside wandb's own source, not in its metadata, so
# a version pin alone does not prove the fix is present -- and pip is free to
# resolve wandb elsewhere if a future edit reshuffles the install order above.
# Read the installed `write_key` back and require the relaxed wording, so a
# regression to the exact-40 check fails the build instead of the GPU job.
RUN python -c "\
import wandb; from wandb.sdk.lib import apikey;\
print('wandb', wandb.__version__);\
src = __import__('inspect').getsource(apikey.write_key);\
assert 'at least 40' in src, 'wandb reintroduced the exact-40-character API key check';\
print('WANDB LONG KEY OK')"

# Set the default command to bash with the environment activated
ENTRYPOINT ["conda", "run", "-n", "dynamicmpnn"]
CMD ["/bin/bash"]

