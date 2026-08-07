FROM nvcr.io/nvidia/pytorch:24.03-py3
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV FORCE_CUDA=1

RUN pip install --no-cache-dir torch-geometric==2.5.3 \
    && pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
       -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

COPY lib/ ./lib/
