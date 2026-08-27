FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda clean -a -y

ENV PATH=/opt/conda/envs/Surffold/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/Surffold/lib:$LD_LIBRARY_PATH


CMD ["python", "-c", "import torch; print('CUDA Available:', torch.cuda.is_available()); print('H200 Device detected:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Fail')"]

