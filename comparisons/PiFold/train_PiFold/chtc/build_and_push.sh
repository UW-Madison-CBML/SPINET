#!/bin/bash
# Build the PiFold CHTC image and push it to Docker Hub so CHTC execute
# nodes can pull it (CHTC's docker universe pulls images from a public,
# or credentialed private, registry -- it cannot use a local image).
#
# Usage: ./build_and_push.sh <dockerhub_username> [tag]
set -euo pipefail

USERNAME=${1:?Usage: $0 cmikulski4 [tag]}
TAG=${2:-v2}
IMAGE="${USERNAME}/pifold:${TAG}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">>> Building ${IMAGE}"
docker build -t "${IMAGE}" -f "${REPO_ROOT}/chtc/Dockerfile" "${REPO_ROOT}"

echo ">>> Pushing ${IMAGE}"
docker push "${IMAGE}"

echo ">>> Done. Set 'docker_image = ${IMAGE}' in chtc/pifold.sub"
