# Build:  docker build -t <dockerhub_user>/dynamicmpnn:latest -f chtc/Dockerfile .
# Push:   docker push <dockerhub_user>/dynamicmpnn:latest

# Use official miniconda3 base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Update conda and install mamba for faster dependency resolution
RUN conda install -n base -c conda-forge mamba -y && \
    conda clean --all -f -y

# Create the conda environment with all dependencies
# Using mamba for faster dependency solving
RUN mamba create -n dynamicmpnn -y \
    -c pyg -c pytorch -c conda-forge \
    python=3.10 \
    pip'>=24' \
    pytorch \
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
    pytest \
    pyg::pyg \
    pyg::pytorch-scatter \
    pyg::pytorch-cluster && \
    mamba clean --all -f -y

# Activate the environment and install pip dependencies
SHELL ["conda", "run", "-n", "dynamicmpnn", "/bin/bash", "-c"]

RUN pip install --no-cache-dir \
    graphein==1.7.6 \
    'pytorch-lightning>=2.0' \
    lovely-tensors \
    wandb

# Set the default command to bash with the environment activated
ENTRYPOINT ["conda", "run", "-n", "dynamicmpnn"]
CMD ["/bin/bash"]
