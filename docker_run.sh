#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DOCKER_IMAGE="rad:1.0"

# check cuda visible device set
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo env CUDA_VISIBLE_DEVICES is not set!
  exit 1
fi

if [ ! -f ./mjkey.txt ]; then
  echo "./mjkey.txt not exist!!."
  exit 1
fi

# build docker images
docker build --tag ${DOCKER_IMAGE} .

# run run.sh
# docker run --gpus ${CUDA_VISIBLE_DEVICES} -it -v $(pwd):/workspace ${DOCKER_IMAGE}
time docker run --gpus ${CUDA_VISIBLE_DEVICES} -it -v $(pwd):/workspace ${DOCKER_IMAGE} bash test.sh


# revert back ownerships of the mounted files
docker run -it -v $(pwd):/workspace ${DOCKER_IMAGE} chown -R $(id -u):$(id -g) /workspace

