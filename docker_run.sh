#!/bin/bash

export JOB_NAME="ml-project1"
export IMAGE="ml/project1"
export TAG="latest"
export PYTHON_ENV="development"
export API_PORT=8080
export WORKERS=2
export TIMEOUT=300
export LOG_FOLDER="$(pwd)/logs"

echo ${IMAGE}:${TAG}

# Create log folder if not exists
if [ ! -d ${LOG_FOLDER} ]; then
     mkdir -p ${LOG_FOLDER}
fi

# stop running container with same job name, if any
if [ "$(docker ps -a | grep $JOB_NAME)" ]; then
  docker stop ${JOB_NAME} && docker rm ${JOB_NAME}
fi

# start docker container
# Note: --gpus flag removed for macOS compatibility (not supported on Mac)
docker run -d \
  --rm \
  --platform linux/amd64 \
  -p ${API_PORT}:80 \
  -e "WORKERS=${WORKERS}" \
  -e "TIMEOUT=${TIMEOUT}" \
  -e "PYTHON_ENV=${PYTHON_ENV}" \
  -v "${LOG_FOLDER}:/app/log" \
  --name="${JOB_NAME}" \
  ${IMAGE}:${TAG}
