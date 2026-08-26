# Use the official Miniconda3 base image
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy the environment file into the container
COPY environment.yml .

# Create the conda environment
# We use 'conda clean' at the end to keep the Docker image size small
RUN conda env create -f environment.yml && \
    conda clean -a -y

# Make sure the new environment is loaded by default when running commands
# By adding the environment's bin folder to the PATH
ENV PATH /opt/conda/envs/Surffold/bin:$PATH

# (Optional) Copy your application code into the image
# COPY . /app

# Set the default command
CMD ["python", "--version"]
