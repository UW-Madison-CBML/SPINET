# MapDiff runtime environment for CHTC (Docker universe).
#
# This image only provides the software environment (CUDA + PyTorch +
# PyG/torch_scatter/torch_cluster + DSSP + the pure-Python deps MapDiff
# needs). The MapDiff source code and datasets are NOT baked in here --
# CHTC jobs pull them in at runtime via HTCondor file transfer / the
# /staging area, so the image stays small and doesn't need to be
# rebuilt every time the code changes.
#
# Build:  docker build -t <dockerhub_user>/mapdiff:latest -f chtc/Dockerfile .
# Push:   docker push <dockerhub_user>/mapdiff:latest
# (see build_and_push.sh for a one-shot script)

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# mkdssp (DSSP) computes secondary structure in data/generate_graph_atlas.py
# and data/generate_graph_cath.py. The base image ships conda, which is the
# only reliable way to get a prebuilt mkdssp binary.
RUN conda install -y -c ostrokach dssp && conda clean -afy

# pyg/torch_scatter/torch_cluster must be built/pinned against the exact
# torch+CUDA combo above (1.13.1 / cu117).
RUN conda install -y pyg=2.4.0 -c pyg && conda clean -afy
RUN pip install --no-cache-dir \
        torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-1.13.0+cu117.html \
    && pip install --no-cache-dir \
        rdkit==2023.3.3 \
        hydra-core==1.3.2 \
        biopython==1.81 \
        einops==0.7.0 \
        prettytable \
        comet_ml \
        wandb==0.22.3

WORKDIR /workspace
