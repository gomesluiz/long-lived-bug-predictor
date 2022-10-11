#!/bin/bash
USER=`whoami`
IMAGE=long-lived-bug-prediction-w-ml-v1
SHAREDIR=/home/${USER}/Workspace/long-lived-bug-predictor-w-ml/
WORKDIR=/home/${USER}/Workspace/long-lived-bug-predictor-w-ml/
LOGFILE=/home/${USER}/Workspace/long-lived-bug-predictor-w-ml/output/logs/r4_4b_predict_long_lived_bug.error
docker run -d -it --ipc=host --userns=host -v ${SHAREDIR}/:${SHAREDIR}/ --workdir ${WORKDIR}/ ${IMAGE}:latest Rscript ${WORKDIR}/r4_4b_predict_long_lived_bug.R | tee -a $LOGFILE
