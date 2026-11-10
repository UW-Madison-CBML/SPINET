#!/bin/bash
# Build the ADFLIP CHTC image and push it to Docker Hub so CHTC execute nodes can pull it
# (CHTC's docker universe pulls images from a public, or credentialed private, registry --
# it cannot use a local image).
#
# Usage: ./build_and_push.sh <dockerhub_username> [tag]
set -euo pipefail

USERNAME=${1:?Usage: $0 cmikulski4 [tag]}
TAG=${2:-v1}
IMAGE="${USERNAME}/adflip:${TAG}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Building ${IMAGE}"
docker build -t "${IMAGE}" -f "${HERE}/Dockerfile" "${HERE}"

echo ">>> Pushing ${IMAGE}"
docker push "${IMAGE}"

echo ">>> Done. Set 'docker_image = ${IMAGE}' in eval_pretrained.sub"
