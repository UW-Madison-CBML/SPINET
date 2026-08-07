FROM nvcr.io/nvidia/pytorch:24.08-py3@sha256:60cc7c9cf24b423c28f621287779ebe5e52cd39cfd284df04983df52566c1e39
WORKDIR /app


COPY requirements.txt .
RUN echo "numpy<2" > /tmp/constraints.txt

RUN pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt

ENV FORCE_CUDA=1
RUN pip install --no-cache-dir -c /tmp/constraints.txt torch-geometric==2.5.3
RUN pip install --no-cache-dir -c /tmp/constraints.txt torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

COPY lib/ ./lib/
