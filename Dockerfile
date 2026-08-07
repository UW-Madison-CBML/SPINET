FROM nvcr.io/nvidia/pytorch:24.08-py3
WORKDIR /app

RUN python -c "import torch; print('BUILD-TIME TORCH:', torch.__version__, torch.version.cuda)"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import torch; print('POST-REQS TORCH:', torch.__version__, torch.version.cuda)"

ENV FORCE_CUDA=1
RUN pip install --no-cache-dir torch-geometric==2.5.3
RUN pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

COPY lib/ ./lib/
