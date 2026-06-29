# DFlash ROCm Docker Cookbook

This cookbook builds `docker/dflash.Dockerfile` and starts the SGLang DFlash
server on AMD GPUs. The target model is mounted from an explicit local
directory and passed as `MODEL_PATH`; do not point it at the default Hugging
Face cache.

For AMD GPUs, the container receives ROCm device nodes and GPU selection is
controlled with `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`.

## Requirements

- ROCm-capable host with Docker access to `/dev/kfd` and `/dev/dri`.
- A local Qwen model directory that contains `config.json`, tokenizer files,
  and model weights.
- `docker/dflash.Dockerfile` in the repository root.

Example local model path:

```bash
/data/models/Qwen3.5-397B-A17B
```

Avoid paths like:

```bash
~/.cache/huggingface/hub/...
```

## Bootstrap Script

Use `scripts/my-exps/dflash-container-up.sh` to build `docker/dflash.Dockerfile`
and start the container. The script requires `HOST_MODEL_PATH`, mounts that
directory into the container, and passes `MODEL_PATH` to the SGLang server so it
does not resolve the target model from the default Hugging Face cache.

By default the script builds the image tag `sglang-dflash:rocm72` before
running. Set `BUILD_IMAGE=0` to run an image that has already been built.
`TP_SIZE` is passed into `docker/dflash.Dockerfile` and defaults to `8`.

## Launch Examples

Use all visible AMD GPUs:

```bash
HOST_MODEL_PATH=/data/models/Qwen3.5-397B-A17B \
scripts/my-exps/dflash-container-up.sh
```

Choose GPUs by ID:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
TP_SIZE=8 \
HOST_MODEL_PATH=/data/models/Qwen3.5-397B-A17B \
scripts/my-exps/dflash-container-up.sh
```

Choose fewer GPUs and match tensor parallelism:

```bash
GPU_IDS=0,1,2,3 \
TP_SIZE=4 \
HOST_MODEL_PATH=/data/models/Qwen3.5-397B-A17B \
scripts/my-exps/dflash-container-up.sh
```

Run an already-built image:

```bash
BUILD_IMAGE=0 \
TP_SIZE=8 \
HOST_MODEL_PATH=/data/models/Qwen3.5-397B-A17B \
scripts/my-exps/dflash-container-up.sh
```

Selected GPU mode should expose the same number of GPU IDs as `TP_SIZE`. The
script will warn if `GPU_IDS` and `TP_SIZE` disagree.

The server listens on `0.0.0.0:30000` by default through host networking.

## Profile with AMD Perf v3

AMD Perf v3 is exposed as `rocprofv3` in ROCm. Use it to capture HIP/HSA,
kernel, memory, marker, and RCCL activity from the running SGLang server
process. The examples below attach to the server inside the `sglang-dflash`
container, collect a bounded trace window, drive traffic from the host with
`InferenceX/utils/bench_serving/benchmark_serving.py`, then copy the profile
artifacts back to the repo.

Check that `rocprofv3` is available in the container:

```bash
docker exec sglang-dflash bash -lc 'command -v rocprofv3 || command -v /opt/rocm/bin/rocprofv3'
```

Set common variables on the host:

```bash
export HOST_MODEL_PATH=/data/models/Qwen3.5-397B-A17B
export SERVED_MODEL=/models/Qwen3.5-397B-A17B
export BENCH=/Users/tienpham2/Documents/my-exps/inference/InferenceX/utils/bench_serving/benchmark_serving.py
export PROFILE_DELAY_SEC=5
export PROFILE_DURATION_SEC=45
```

Start the DFlash server in one terminal:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
TP_SIZE=8 \
HOST_MODEL_PATH="${HOST_MODEL_PATH}" \
scripts/my-exps/dflash-container-up.sh
```

Start an AMD Perf v3 collection window from another terminal:

```bash
export PROFILE_LABEL=dflash-single
export PROFILE_CONTAINER_DIR=/tmp/dflash-profile/${PROFILE_LABEL}
export PROFILE_HOST_DIR="${PWD}/profiles/${PROFILE_LABEL}-$(date +%Y%m%d_%H%M%S)"

docker exec -d \
  -e "PROFILE_CONTAINER_DIR=${PROFILE_CONTAINER_DIR}" \
  -e "PROFILE_LABEL=${PROFILE_LABEL}" \
  -e "PROFILE_DELAY_SEC=${PROFILE_DELAY_SEC}" \
  -e "PROFILE_DURATION_SEC=${PROFILE_DURATION_SEC}" \
  sglang-dflash \
  bash -lc '
set -euo pipefail
export PATH="/opt/rocm/bin:${PATH}"
SERVER_PID="$(pgrep -f "sglang.launch_server" | head -n 1)"
rm -rf "${PROFILE_CONTAINER_DIR}"
mkdir -p "${PROFILE_CONTAINER_DIR}"
rocprofv3 \
  --pid "${SERVER_PID}" \
  --sys-trace \
  --stats \
  --summary \
  --summary-output-file "${PROFILE_CONTAINER_DIR}/summary.txt" \
  --output-format pftrace rocpd \
  --output-directory "${PROFILE_CONTAINER_DIR}" \
  --output-file "${PROFILE_LABEL}" \
  --collection-period "${PROFILE_DELAY_SEC}:${PROFILE_DURATION_SEC}:1"
'
```

After `PROFILE_DELAY_SEC` seconds, send one request:

```bash
sleep "${PROFILE_DELAY_SEC}"

python "${BENCH}" \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --endpoint /v1/completions \
  --dataset-name random \
  --model "${SERVED_MODEL}" \
  --tokenizer "${HOST_MODEL_PATH}" \
  --trust-remote-code \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --ignore-eos \
  --save-result \
  --result-dir "${PROFILE_HOST_DIR}"
```

Wait for collection to finish and copy the trace before stopping the container:

```bash
sleep "$((PROFILE_DURATION_SEC + 5))"
mkdir -p "${PROFILE_HOST_DIR}"
docker cp "sglang-dflash:${PROFILE_CONTAINER_DIR}/." "${PROFILE_HOST_DIR}/"
find "${PROFILE_HOST_DIR}" -maxdepth 2 -type f | sort
```

For concurrent request profiling, start a new AMD Perf v3 window with a new
label, then run a larger benchmark during that window:

```bash
export PROFILE_LABEL=dflash-concurrent
export PROFILE_CONTAINER_DIR=/tmp/dflash-profile/${PROFILE_LABEL}
export PROFILE_HOST_DIR="${PWD}/profiles/${PROFILE_LABEL}-$(date +%Y%m%d_%H%M%S)"

docker exec -d \
  -e "PROFILE_CONTAINER_DIR=${PROFILE_CONTAINER_DIR}" \
  -e "PROFILE_LABEL=${PROFILE_LABEL}" \
  -e "PROFILE_DELAY_SEC=${PROFILE_DELAY_SEC}" \
  -e "PROFILE_DURATION_SEC=${PROFILE_DURATION_SEC}" \
  sglang-dflash \
  bash -lc '
set -euo pipefail
export PATH="/opt/rocm/bin:${PATH}"
SERVER_PID="$(pgrep -f "sglang.launch_server" | head -n 1)"
rm -rf "${PROFILE_CONTAINER_DIR}"
mkdir -p "${PROFILE_CONTAINER_DIR}"
rocprofv3 \
  --pid "${SERVER_PID}" \
  --sys-trace \
  --stats \
  --summary \
  --summary-output-file "${PROFILE_CONTAINER_DIR}/summary.txt" \
  --output-format pftrace rocpd \
  --output-directory "${PROFILE_CONTAINER_DIR}" \
  --output-file "${PROFILE_LABEL}" \
  --collection-period "${PROFILE_DELAY_SEC}:${PROFILE_DURATION_SEC}:1"
'

sleep "${PROFILE_DELAY_SEC}"

python "${BENCH}" \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --endpoint /v1/completions \
  --dataset-name random \
  --model "${SERVED_MODEL}" \
  --tokenizer "${HOST_MODEL_PATH}" \
  --trust-remote-code \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 240 \
  --request-rate 8 \
  --max-concurrency 32 \
  --ignore-eos \
  --save-result \
  --result-dir "${PROFILE_HOST_DIR}"

sleep "$((PROFILE_DURATION_SEC + 5))"
mkdir -p "${PROFILE_HOST_DIR}"
docker cp "sglang-dflash:${PROFILE_CONTAINER_DIR}/." "${PROFILE_HOST_DIR}/"
find "${PROFILE_HOST_DIR}" -maxdepth 2 -type f | sort
```

`--sys-trace` is broad and can produce large files. If Perfetto becomes slow,
reduce `PROFILE_DURATION_SEC`, `--num-prompts`, `--random-input-len`, or
`--random-output-len`. For a smaller trace, replace `--sys-trace` with a
narrower set such as `--kernel-trace --hip-trace --memory-copy-trace
--marker-trace --rccl-trace`.

## Visualize with Perfetto UI

AMD Perf v3 writes Perfetto-compatible `.pftrace` files when
`--output-format pftrace` is set. Open [Perfetto UI](https://ui.perfetto.dev/),
choose **Open trace file**, and select the `.pftrace` file copied into
`profiles/<profile-label>-<timestamp>/`.

Useful files in each profile directory:

- `*.pftrace`: timeline for Perfetto UI.
- `*.db` or `*.rocpd`: ROCProfiler database output, depending on the installed
  ROCm version.
- `summary.txt`: text summary from `rocprofv3 --summary`.
- benchmark result JSON: request throughput, TTFT, TPOT, ITL, and end-to-end
  latency from `benchmark_serving.py`.

For large or sensitive traces, use Perfetto locally and avoid uploading profile
artifacts outside the machine.

References:

- [ROCProfiler-SDK rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [Perfetto UI](https://ui.perfetto.dev/)
