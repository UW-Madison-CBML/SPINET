FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3-pip python3-dev git && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip && pip3 install --no-cache-dir torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124


RUN pip3 install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html


RUN pip3 install --no-cache-dir torch-geometric

RUN pip3 install --no-cache-dir scipy biopython scikit-learn tqdm easydict pyyaml



