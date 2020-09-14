#!/bin/bash
USER=`whoami`
IMAGE=long-lived-bug-prediction-ml-v1
SHAREDIR=/home/${USER}/Workspace/long-lived-bug-predictor-ml-in-r/
WORKDIR=/home/${USER}/Workspace/long-lived-bug-predictor-ml-in-r/
LOGFILE=/home/${USER}/Workspace/long-lived-bug-predictor-ml-in-r/output/logs/r4_2b_predict_long_lived_bug.error
docker run -d -it --ipc=host --userns=host -v ${SHAREDIR}/:${SHAREDIR}/ --workdir ${WORKDIR}/ ${IMAGE}:latest Rscript ${WORKDIR}/r4_2b_predict_long_lived_bug.R | tee -a $LOGFILE
