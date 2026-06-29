#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  HOST_MODEL_PATH=/data/models/Qwen3.5-397B-A17B scripts/my-exps/dflash-container-up.sh

Environment variables:
  HOST_MODEL_PATH        Required. Explicit local model directory to mount.
  GPU_IDS                AMD GPU IDs to expose, for example: 0,1,2,3,4,5,6,7.
                         Default: all.
  TP_SIZE                Tensor parallel size passed to sglang.launch_server.
                         Default: 8.
  IMAGE_NAME             Docker image tag to build/run.
                         Default: sglang-dflash:rocm72.
  CONTAINER_NAME         Docker container name.
                         Default: sglang-dflash.
  CONTAINER_MODEL_PATH   Model path inside the container.
                         Default: /models/Qwen3.5-397B-A17B.
  BUILD_IMAGE            Build docker/dflash.Dockerfile before running.
                         Default: 1. Set to 0 to run an existing image.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DOCKERFILE="${REPO_ROOT}/docker/dflash.Dockerfile"

IMAGE_NAME="${IMAGE_NAME:-sglang-dflash:rocm72}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-dflash}"
CONTAINER_MODEL_PATH="${CONTAINER_MODEL_PATH:-/models/Qwen3.5-397B-A17B}"
GPU_IDS="${GPU_IDS:-all}"
TP_SIZE="${TP_SIZE:-8}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"

if [[ -z "${HOST_MODEL_PATH:-}" ]]; then
  echo "HOST_MODEL_PATH is required." >&2
  echo >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Dockerfile not found: ${DOCKERFILE}" >&2
  exit 1
fi

if [[ ! -d "${HOST_MODEL_PATH}" ]]; then
  echo "HOST_MODEL_PATH does not exist or is not a directory: ${HOST_MODEL_PATH}" >&2
  exit 1
fi

HOST_MODEL_PATH="$(cd -- "${HOST_MODEL_PATH}" && pwd -P)"

if [[ ! -f "${HOST_MODEL_PATH}/config.json" ]]; then
  echo "Expected ${HOST_MODEL_PATH}/config.json. Point HOST_MODEL_PATH at the model directory itself." >&2
  exit 1
fi

if [[ "${HOST_MODEL_PATH}" == *"/.cache/huggingface" || "${HOST_MODEL_PATH}" == *"/.cache/huggingface/"* ]]; then
  echo "HOST_MODEL_PATH must be an explicit local model directory, not the default Hugging Face cache." >&2
  exit 1
fi

if [[ ! "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP_SIZE must be a positive integer: ${TP_SIZE}" >&2
  exit 1
fi

if [[ "${BUILD_IMAGE}" != "0" ]]; then
  docker build \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_NAME}" \
    "${REPO_ROOT}"
fi

gpu_env=()
if [[ "${GPU_IDS}" != "all" ]]; then
  gpu_env+=(-e "HIP_VISIBLE_DEVICES=${GPU_IDS}")
  gpu_env+=(-e "ROCR_VISIBLE_DEVICES=${GPU_IDS}")

  IFS=',' read -r -a selected_gpus <<< "${GPU_IDS}"
  if [[ "${#selected_gpus[@]}" -ne "${TP_SIZE}" ]]; then
    echo "Warning: TP_SIZE=${TP_SIZE}, but GPU_IDS has ${#selected_gpus[@]} ID(s)." >&2
  fi
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --shm-size 64g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e "MODEL_PATH=${CONTAINER_MODEL_PATH}" \
  -e "TP_SIZE=${TP_SIZE}" \
  "${gpu_env[@]}" \
  -v "${HOST_MODEL_PATH}:${CONTAINER_MODEL_PATH}:ro" \
  "${IMAGE_NAME}"
