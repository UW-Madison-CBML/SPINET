#!/bin/bash

# Usage: run.sh <protein_name>
# Example: run.sh ACRIIA4

# Exit on error and print commands
set -ex
PROTEIN_NAME=$1

tar -xvf inputs.tar.gz 
mv inputs/$PROTEIN_NAME ./

# Crappy workaround for condor path issues
export HOME=$_CONDOR_SCRATCH_DIR
export XDG_CACHE_HOME=$_CONDOR_SCRATCH_DIR/.cache

# Detect and echo hardware, count cpus and gpus
echo "Detected hardware:"
lscpu
echo "Number of CPUs: $(nproc)"

if nvidia-smi &> /dev/null; then
    echo "Detected NVIDIA GPUs:"
    nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c
    echo "CUDA Version:"
    nvidia-smi | grep "CUDA Version"
else
    echo "No NVIDIA GPUs detected."
fi
################################################################################
# Platform Diagnostics
echo "Checking GROMACS installation..."
gmx -version

echo "Running GROMACS job: ${PROTEIN_NAME}"
tar -xzvf mdp.tar.gz
tar -xvzf md_data.tar.gz
mkdir data
time ./gromacs_pipeline.sh ${PROTEIN_NAME}

tar -czvf md_data.tar.gz md_data/

rm -rf md_data/ mdp mdp.tar.gz
