FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL version="1.0" 

ARG USER_ID
ARG USER_NAME
ARG GROUP_ID 

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# -- install basic utilities
RUN apt-get update --fix-missing && \
    apt-get install -y \
        wget bzip2 ca-certificates curl git apt-utils build-essential cmake pkg-config vim \
        libsm6 libxrender1 libfontconfig1 libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev && \
    apt-get autoclean && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache 

# install miniconda
# https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh
# https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV CONDA_AUTO_UPDATE_CONDA=false

# -- install the torch package and its dependences
COPY src/bert/2-feature-extraction/requirements.txt /tmp/ 
WORKDIR /tmp/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

RUN addgroup --gid ${GROUP_ID} ${USER_NAME}
RUN adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${USER_NAME}

WORKDIR /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/data/clean/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/data/metrics/feature-extraction
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/data/tensors/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/data/dtms/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/src/bert/2-feature-extraction/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/src/tfidf/2-feature-extraction/
COPY ./src/bert/2-feature-extraction/*.py /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/src/bert/2-feature-extraction/
COPY ./src/tfidf/2-feature-extraction/*.py /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/src/tfidf/2-feature-extraction/
COPY ./data/clean/*.csv /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/data/clean/
COPY ./data/tensors/*.pt /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/data/tensors/
COPY ./data/dtms/*.npy /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/data/dtms/
COPY ./data/metrics/feature-extraction/*.csv /home/${USER_NAME}/Workspace/long-lived-bug-deep-predictor/data/metrics/feature-extraction/

RUN chown $USER_ID:$GROUP_ID -R /home/${USER_NAME}/
USER ${USER_NAME}