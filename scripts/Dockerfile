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
WORKDIR /home/${USER_NAME}/Workspace/long-lived-bug-predictor-ml-in-r/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-predictor-ml-in-r/R/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-predictor-ml-in-r/data/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-predictor-ml-in-r/output/data/
RUN mkdir -p /home/${USER_NAME}/Workspace/long-lived-bug-predictor-ml-in-r/output/logs/
COPY  ./*.R /home/${USER_NAME}/Workspace/long-lived-bug-predictor-ml-in-r/
COPY  ./R/* /home/${USER_NAME}/Workspace/long-lived-bug-predictor-ml-in-r/R/
COPY  ./data/*.csv /home/${USER_NAME}/Workspace/long-lived-bug-predictor-ml-in-r/data/

RUN chown ${USER_ID}:${GROUP_ID} -R /home/${USER_NAME}
USER ${USER_NAME}
