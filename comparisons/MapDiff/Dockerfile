# MapDiff runtime environment for CHTC (Docker universe).
#
# This image only provides the software environment (CUDA + PyTorch +
# PyG/torch_scatter/torch_cluster + DSSP + the pure-Python deps MapDiff
# needs, plus transformers for the ESMFold self-consistency RMSD metric).
# The MapDiff source code and datasets are NOT baked in here -- train.sh
# pulls them in at run time (a pristine MapDiff.tar.gz + this folder's
# ATLAS/wandb/scRMSD overlay, transferred in flat by HTCondor / the
# /staging area), so the image stays small and doesn't need to be
# rebuilt every time the code changes.
#
# Build:  docker build -t <dockerhub_user>/mapdiff:latest -f Dockerfile .
# Push:   docker push <dockerhub_user>/mapdiff:latest
# (see build_and_push.sh for a one-shot script)

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# mkdssp (DSSP) computes secondary structure in data/generate_graph_relaxed.py
# and data/generate_graph_cath.py (Bio.PDB.DSSP is invoked as dssp='mkdssp').
# The base image ships conda, which is the only reliable way to get a
# prebuilt mkdssp binary.
RUN conda install -y -c ostrokach dssp && conda clean -afy

# pyg/torch_scatter/torch_cluster must be built/pinned against the exact
# torch+CUDA combo in the FROM line above (2.1.2 / cu121).
RUN conda install -y pyg=2.4.0 -c pyg && conda clean -afy
RUN pip install --no-cache-dir \
        torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.1.2+cu121.html \
    && pip install --no-cache-dir \
        "numpy<2" \
        rdkit==2023.3.3 \
        hydra-core==1.3.2 \
        biopython==1.81 \
        einops==0.7.0 \
        prettytable \
        wandb==0.22.3 \
        "transformers==4.55.4" \
        accelerate==1.10.1 \
        scipy \
        scikit-learn \
        h5py \
        matplotlib \
        pandas \
        tqdm \
        mdtraj==1.10.3 \
        requests \
        ruff

# Fail the build (rather than a 30-minute CHTC job) if the resolved wheels
# don't actually hang together. The two failure modes this catches, both of
# which have bitten this image before:
#
#  * transformers silently disabling its torch backend. transformers >=4.56
#    calls torch.utils._pytree.register_pytree_node, which only exists in
#    torch >=2.2, and transformers 5.x hard-requires torch >=2.5 -- with
#    torch pinned at 2.1.2 either one degrades to a tokenizers-only install
#    and EsmForProteinFolding then raises "requires the PyTorch library but
#    it was not found" deep inside scrmsd.load_esmfold(). 4.55.4 is the
#    newest release that works against torch 2.1.2.
#  * numpy 2.x getting pulled in. torch 2.1.2 is compiled against numpy 1.x
#    and dies at import with "_ARRAY_API not found" / "module compiled using
#    NumPy 1.x cannot be run in NumPy 2.0.2".
RUN python -c "\
import numpy, scipy, sklearn, torch, transformers; \
from transformers.utils import is_torch_available, is_scipy_available; \
from transformers import EsmForProteinFolding; \
assert numpy.__version__.startswith('1.'), 'numpy must be 1.x for torch 2.1.2, got ' + numpy.__version__; \
assert is_torch_available(), 'transformers ' + transformers.__version__ + ' disabled its torch backend against torch ' + torch.__version__; \
assert is_scipy_available(), 'scipy missing: ESMFold falls back to an approximate init'; \
EsmForProteinFolding.from_pretrained; \
print('>>> ok: torch', torch.__version__, '| transformers', transformers.__version__, '| numpy', numpy.__version__, '| sklearn', sklearn.__version__)"

WORKDIR /workspace
