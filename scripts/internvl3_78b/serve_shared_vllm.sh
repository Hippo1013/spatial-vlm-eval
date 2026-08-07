#!/usr/bin/env bash
# Start the one locked InternVL3-78B vLLM endpoint shared by three benchmarks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
PROFILE="internvl3_78b"
REVISION="3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356"
SERVED_MODEL_NAME="internvl3-78b-three-bench"
RUNTIME_VERSION="0.19.0"

operation="${1:-run}"
if (( $# > 1 )); then
  echo "[three-bench-vllm] expected no arguments, --check, or --dry-run" >&2
  exit 2
fi
case "${operation}" in
  run|--check|--dry-run) ;;
  --help|-h)
    echo "Usage: bash scripts/internvl3_78b/serve_shared_vllm.sh [--check|--dry-run]"
    exit 0
    ;;
  *)
    echo "[three-bench-vllm] unsupported operation: ${operation}" >&2
    exit 2
    ;;
esac

gpu_ids="${INTERNVL3_78B_THREE_BENCH_GPU_IDS:-0,1,2,3}"
port="${INTERNVL3_78B_THREE_BENCH_PORT:-18103}"
model_path="${INTERNVL3_78B_MODEL:-/required/internvl3-78b-model}"
vllm_executable="${INTERNVL3_78B_THREE_BENCH_VLLM:-${SPBENCH_SI_VLLM:-${QSPATIAL_VLLM:-${CVBENCH_VLLM:-vllm}}}}"
if [[ ! "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
  echo "[three-bench-vllm] GPU ids must contain exactly four comma-separated integers" >&2
  exit 2
fi
IFS=',' read -r -a selected_gpus <<<"${gpu_ids}"
unique_gpu_count="$(printf '%s\n' "${selected_gpus[@]}" | awk '!seen[$0]++ {count++} END {print count + 0}')"
if (( unique_gpu_count != 4 )); then
  echo "[three-bench-vllm] GPU ids must be distinct" >&2
  exit 2
fi
if [[ ! "${port}" =~ ^[1-9][0-9]*$ ]] || (( port > 65535 )); then
  echo "[three-bench-vllm] port must be an integer in 1..65535" >&2
  exit 2
fi

args=(
  serve "${model_path}"
  --revision "${REVISION}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 127.0.0.1
  --port "${port}"
  --tensor-parallel-size 4
  --dtype bfloat16
  --max-model-len 32768
  --gpu-memory-utilization 0.90
  --seed 42
  --limit-mm-per-prompt.image 1
  --trust-remote-code
)

if [[ "${operation}" == "--dry-run" ]]; then
  printf '[three-bench-vllm] dry-run:'
  printf ' %q' env "CUDA_VISIBLE_DEVICES=${gpu_ids}" "${vllm_executable}" "${args[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -e "${model_path}" ]]; then
  echo "[three-bench-vllm] INTERNVL3_78B_MODEL is unavailable: ${model_path}" >&2
  exit 2
fi
if [[ "${vllm_executable}" == */* ]]; then
  if [[ ! -x "${vllm_executable}" ]]; then
    echo "[three-bench-vllm] vLLM executable is unavailable: ${vllm_executable}" >&2
    exit 2
  fi
elif ! command -v "${vllm_executable}" >/dev/null 2>&1; then
  echo "[three-bench-vllm] vLLM command is unavailable: ${vllm_executable}" >&2
  exit 2
fi
runtime_output="$("${vllm_executable}" --version 2>&1)" || {
  echo "[three-bench-vllm] cannot read vLLM version" >&2
  exit 2
}
actual_runtime="${runtime_output##* }"
if [[ "${actual_runtime}" != "${RUNTIME_VERSION}" ]]; then
  echo "[three-bench-vllm] vLLM version mismatch: expected ${RUNTIME_VERSION}, got ${actual_runtime}" >&2
  exit 2
fi

python_executable="${LATENT_PYTHON:-${PYTHON:-python}}"
if ! "${python_executable}" - "${port}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
then
  echo "[three-bench-vllm] port ${port} is occupied; existing process was left untouched" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="${gpu_ids}"
if ! MIN_FREE_GPU_MIB=76000 MIN_GPU_COUNT=4 REQUIRE_IDLE_GPU=1 \
  bash "${REPOSITORY_ROOT}/scripts/msmu/gpu_preflight.sh"; then
  echo "[three-bench-vllm] four idle 80GB GPUs are unavailable" >&2
  exit 4
fi

if [[ "${operation}" == "--check" ]]; then
  echo "[three-bench-vllm] CHECK profile=${PROFILE} revision=${REVISION}"
  echo "[three-bench-vllm] CHECK served_model_name=${SERVED_MODEL_NAME}"
  echo "[three-bench-vllm] CHECK runtime=${RUNTIME_VERSION} dtype=bfloat16 tp=4 max_model_len=32768 seed=42 images=1"
  exit 0
fi

exec "${vllm_executable}" "${args[@]}"
