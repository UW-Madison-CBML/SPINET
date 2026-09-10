FROM nvcr.io/nvidia/pytorch:24.08-py3@sha256:60cc7c9cf24b423c28f621287779ebe5e52cd39cfd284df04983df52566c1e39
RUN sed -i '/extra-index-url/d; /pypi.ngc.nvidia.com/d; /trusted-host/d' /usr/pip.conf /etc/pip.conf
WORKDIR /app

RUN python -c "import torch; print('BASE TORCH:', torch.__version__, torch.version.cuda)"

COPY requirements.txt .
RUN echo "numpy<2" > /tmp/constraints.txt

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="9.0"
RUN export MAX_JOBS=$(nproc) && pip install --no-cache-dir -c /tmp/constraints.txt --no-build-isolation \
    torch-geometric==2.5.3 torch-scatter torch-sparse torch-cluster
RUN pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt
RUN python -c "import torch; print('POST-REQS TORCH:', torch.__version__)"
COPY lib/ ./lib/
