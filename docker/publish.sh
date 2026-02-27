#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=supervisely/yolov8
IMAGE_TAG=1.1.0
IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"

docker build -t "${IMAGE_REF}" . && \
docker push "${IMAGE_REF}"
