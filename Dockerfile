FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget git curl ca-certificates sudo build-essential \
    python3-dev python3-pip python3-venv \
    libopenblas-dev libomp-dev libboost-all-dev libeigen3-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH
ENV CONDA_DEFAULT_ENV=3detr
ENV ENV_NAME=3detr

RUN conda create -y -n $ENV_NAME python=3.6 && conda clean -afy

SHELL ["/bin/bash", "-c"]
RUN source activate $ENV_NAME && \
    conda install -y -c conda-forge \
        matplotlib \
        plyfile \
        'trimesh>=2.35.39,<2.35.40' \
        'networkx>=2.2,<2.3' \
        scipy \
        opencv=4.5.3 && \
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install open3d==0.15.2 && \
    conda clean -afy

WORKDIR /workspace
RUN git clone https://github.com/seclusioner/3detr_custom . && \
    git submodule update --init --recursive

RUN source activate $ENV_NAME && \
    export TORCH_CUDA_ARCH_LIST="8.6" && \
    cd third_party/pointnet2 && \
    python setup.py install

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]
