FROM rocker/verse:3.6
LABEL version="1.0"

ARG USER_ID
ARG USER_NAME
ARG GROUP_ID

# install R packages 
COPY scripts/install_packages.R /tmp/
WORKDIR /tmp/
RUN Rscript install_packages.R

# create the owner user
RUN addgroup --gid ${GROUP_ID} ${USER_NAME}
RUN adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${USER_NAME}

# create directories
WORKDIR /home/${USER_NAME}/Workspace/long-lived-bug-predictor-w-ml/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-predictor-w-ml/R/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-predictor-w-ml/data/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-predictor-w-ml/output/data/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-predictor-w-ml/output/logs/
COPY  ./*.R /home/${USER_NAME}/Workspace/long-lived-bug-predictor-w-ml/
COPY  ./R/* /home/${USER_NAME}/Workspace/long-lived-bug-predictor-w-ml/R/
COPY  ./data/*.csv /home/${USER_NAME}/Workspace/long-lived-bug-predictor-w-ml/data/

RUN chown ${USER_ID}:${GROUP_ID} -R /home/${USER_NAME}
USER ${USER_NAME}
