FROM --platform=linux/amd64 nvidia/cuda:11.8.0-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_DIR=/job/hdxrank-conda
ENV PATH=$CONDA_DIR/bin:$PATH
WORKDIR /job

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget git curl ca-certificates bzip2 gcc libxrender1 libxext6 libsm6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /job/conda && \
    rm Miniforge3-Linux-x86_64.sh && \
    /job/conda/bin/conda create -p $CONDA_DIR python=3.10 numpy=1.26.3 mkl=2024.0 -y && \
    /job/conda/bin/conda config --set channel_priority strict && \
    /job/conda/bin/conda clean -afy

RUN /job/conda/bin/conda install -p $CONDA_DIR -c pytorch -c nvidia \
        pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -y && \
    /job/conda/bin/conda clean -afy

RUN $CONDA_DIR/bin/pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html && \
    $CONDA_DIR/bin/pip install --resume-retries 3 torchdrug && \
    $CONDA_DIR/bin/pip install scikit-learn biotite && \
    $CONDA_DIR/bin/pip install biopython==1.83 openpyxl pdb2sql pyyaml && \
    rm -rf /root/.cache/pip

SHELL ["/bin/bash", "-c"]
ENV KMP_AFFINITY=disabled

WORKDIR /job
CMD ["/bin/bash"]
