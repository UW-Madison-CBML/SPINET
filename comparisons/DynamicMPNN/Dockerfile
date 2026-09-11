# Build:  docker build -t <dockerhub_user>/dynamicmpnn:latest -f chtc/Dockerfile .
# Push:   docker push <dockerhub_user>/dynamicmpnn:latest

# Use official miniconda3 base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Update conda and install mamba for faster dependency resolution
RUN conda install -n base -c conda-forge mamba -y && \
    conda clean --all -f -y

RUN mamba create -n dynamicmpnn -y \
    -c conda-forge \
    python=3.10 \
    pip'>=24' \
    'numpy<2' \
    pandas \
    scipy \
    scikit-learn \
    biopython \
    matplotlib \
    omegaconf \
    hydra-core \
    loguru \
    python-dotenv \
    beartype \
    jaxtyping \
    einops \
    pytest && \
    mamba clean --all -f -y

# Activate the environment and install pip dependencies
SHELL ["conda", "run", "-n", "dynamicmpnn", "/bin/bash", "-c"]

# Pin PyTorch to a CUDA 12.1 build and fetch torch-scatter/torch-cluster from
# PyG's wheel index built for that exact torch+cuda combo (must match
# torch-${TORCH_VERSION}+cu121 exactly -- see https://data.pyg.org/whl/).
ENV TORCH_VERSION=2.1.2
ENV CUDA_TAG=cu121

RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_TAG} && \
    pip install --no-cache-dir \
    torch-scatter torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html

RUN pip install --no-cache-dir \
    graphein==1.7.6 \
    'pytorch-lightning>=2.0' \
    lovely-tensors \
    wandb \
    transformers

# Set the default command to bash with the environment activated
ENTRYPOINT ["conda", "run", "-n", "dynamicmpnn"]
CMD ["/bin/bash"]

