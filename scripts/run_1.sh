#!/bin/bash

IMAGE=long-lived-bug-prediction-ml-v1
SHAREDIR=/home/g348286/Workspace/long-lived-bug-predictor-ml-in-r/data/
WORKDIR=/home/g348286/Workspace/long-lived-bug-predictor-ml-in-r/
LOGFILE=/home/g348286/Workspace/long-lived-bug-predictor-ml-in-r/output/logs/r3_1a_predict_long_lived_bug.error
docker run -d -it --ipc=host --userns=host -v ${SHAREDIR}/:${SHAREDIR}/ --workdir ${WORKDIR}/ ${IMAGE}:latest Rscript ${WORKDIR}/r3_1a_predict_long_lived_bug.R | tee -a $LOGFILE
