# PiFold runtime environment for CHTC (Docker universe).
#
# This image only provides the software environment (CUDA + PyTorch +
# torch_scatter + the small set of pure-Python deps PiFold needs). The
# PiFold source code and datasets are NOT baked in here — CHTC jobs pull
# them in at runtime via HTCondor file transfer / the /staging area, so
# the image stays small and doesn't need to be rebuilt every time the
# code changes.
#
# Build:  docker build -t <dockerhub_user>/pifold:latest -f chtc/Dockerfile .
# Push:   docker push <dockerhub_user>/pifold:latest
# (see build_and_push.sh for a one-shot script)

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# torch_scatter must be built/pinned against the exact torch+CUDA combo
# above (2.1.2 / cu118) -- see https://data.pyg.org/whl/ for the wheel index.
RUN pip install --no-cache-dir \
        torch_scatter -f https://data.pyg.org/whl/torch-2.1.2+cu118.html \
    && pip install --no-cache-dir \
        biopython==1.83 \
        numpy==1.26.4 \
        tqdm==4.66.2

WORKDIR /workspace
