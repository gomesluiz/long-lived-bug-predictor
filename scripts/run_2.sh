#!/bin/bash

IMAGE=long-lived-bug-prediction-ml-v1
SHAREDIR=/home/lgomes/Workspace/doctorate/projects/long-lived-bug-prediction/machine-learning/source/datasets
WORKDIR=/home/lgomes/Workspace/doctorate/projects/long-lived-bug-prediction/machine-learning/R
LOGFILE=/home/lgomes/Workspace/doctorate/projects/long-lived-bug-prediction/machine-learning/source/datasets/predict_long_lived_bug_e1_train_metrics_rev1.error
docker run  -d -it --ipc=host --userns=host -v ${SHAREDIR}/:${SHAREDIR}/ --workdir ${WORKDIR}/ ${IMAGE}:latest Rscript ${WORKDIR}/predict_long_lived_bug_e1_train_metrics_rev1.R | tee -a $LOGFILE
