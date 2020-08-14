#!/bin/bash
docker build -t long-lived-bug-prediction-ml-v1 -f scripts/Dockerfile \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) .
