FROM rocker/verse:3.6
LABEL version="1.0"

ARG USER_ID
ARG GROUP_ID

# install R packages 
COPY scripts/install_packages.R /tmp/
WORKDIR /tmp/
RUN Rscript install_packages.R

# create the owner user
RUN addgroup --gid ${GROUP_ID} g348286
RUN adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} g348286

# create directories
WORKDIR /home/g348286/Workspace/long-lived-bug-predictor-ml-in-r/
RUN mkdir -p /home/g348286/Workspace/long-lived-bug-predictor-ml-in-r/R/
RUN mkdir -p /home/g348286/Workspace/long-lived-bug-predictor-ml-in-r/data/
RUN mkdir -p /home/g348286/Workspace/long-lived-bug-predictor-ml-in-r/output/data/
RUN mkdir -p /home/g348286/Workspace/long-lived-bug-predictor-ml-in-r/output/logs/
COPY  ./*.R /home/g348286/Workspace/long-lived-bug-predictor-ml-in-r/
COPY  ./R/* /home/g348286/Workspace/long-lived-bug-predictor-ml-in-r/R/
COPY  ./data/*.csv /home/g348286/Workspace/long-lived-bug-predictor-ml-in-r/data/

RUN chown ${USER_ID}:${GROUP_ID} -R /home/g348286
USER g348286