# Dockerfile for neural network, protein dynamics simulations will be done with a gromacs image
FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y pigz

RUN pip install --no-cache-dir -r requirements.txt
# TODO figure this out
#RUN pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.9.0+cu128.html

ENV FORCE_CUDA=1
RUN pip install --no-cache-dir torch-geometric
#torch-scatter torch-sparse torch-cluster 
COPY lib/ ./lib/

