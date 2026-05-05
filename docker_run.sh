#!/bin/bash

export JOB_NAME="ml-project1"
export IMAGE="stef04/plant-care-card"
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

# load env vars if .env exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
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
  -e "MISTRAL_API_KEY=${MISTRAL_API_KEY}" \
  -e "TAVILY_API_KEY=${TAVILY_API_KEY}" \
  -v "${LOG_FOLDER}:/app/log" \
  --name="${JOB_NAME}" \
  ${IMAGE}:${TAG}
