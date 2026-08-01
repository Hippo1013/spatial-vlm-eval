#!/usr/bin/env bash
# Read-only GPU capacity gate. Never terminates or modifies existing processes.

set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu-preflight] nvidia-smi is unavailable; inference must run on the Ubuntu GPU server" >&2
  exit 1
fi

gpu_spec="${CUDA_VISIBLE_DEVICES:-0}"
minimum="${MIN_FREE_GPU_MIB:-20000}"
minimum_gpu_count="${MIN_GPU_COUNT:-1}"
maximum_utilization="${MAX_GPU_UTILIZATION_PERCENT:-10}"
require_idle="${REQUIRE_IDLE_GPU:-1}"
IFS=',' read -r -a gpu_ids <<< "${gpu_spec}"

if [[ ! "${minimum}" =~ ^[0-9]+$ ]]; then
  echo "[gpu-preflight] MIN_FREE_GPU_MIB must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${minimum_gpu_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[gpu-preflight] MIN_GPU_COUNT must be a positive integer" >&2
  exit 2
fi
if [[ ! "${maximum_utilization}" =~ ^[0-9]+$ ]] || (( maximum_utilization > 100 )); then
  echo "[gpu-preflight] MAX_GPU_UTILIZATION_PERCENT must be an integer in [0,100]" >&2
  exit 2
fi
if [[ "${require_idle}" != "0" && "${require_idle}" != "1" ]]; then
  echo "[gpu-preflight] REQUIRE_IDLE_GPU must be 0 or 1" >&2
  exit 2
fi

if (( minimum_gpu_count > 1 )); then
  selected_gpu_count="$(
    printf '%s\n' "${gpu_ids[@]}" \
      | awk '{$1=$1} NF && !seen[$0]++ {count++} END {print count + 0}'
  )"
  if (( selected_gpu_count < minimum_gpu_count )); then
    echo "[gpu-preflight] requires at least ${minimum_gpu_count} selected GPUs; CUDA_VISIBLE_DEVICES provides ${selected_gpu_count}" >&2
    exit 4
  fi

  available_gpu_count="$(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
      | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {count++} END {print count + 0}'
  )"
  if (( available_gpu_count < minimum_gpu_count )); then
    echo "[gpu-preflight] requires at least ${minimum_gpu_count} physical GPUs; nvidia-smi detected ${available_gpu_count}" >&2
    exit 4
  fi
fi

for gpu_id in "${gpu_ids[@]}"; do
  free_mib="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
  utilization="$(
    nvidia-smi --id="${gpu_id}" --query-gpu=utilization.gpu --format=csv,noheader,nounits \
      | head -n 1 \
      | tr -d '[:space:]'
  )"
  compute_pids="$(
    nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits \
      | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); print}'
  )"
  if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
    echo "[gpu-preflight] could not read free memory for GPU ${gpu_id}" >&2
    exit 1
  fi
  if [[ ! "${utilization}" =~ ^[0-9]+$ ]]; then
    echo "[gpu-preflight] could not read utilization for GPU ${gpu_id}" >&2
    exit 1
  fi
  process_count=0
  if [[ -n "${compute_pids}" ]]; then
    process_count="$(printf '%s\n' "${compute_pids}" | wc -l | tr -d '[:space:]')"
  fi
  echo "[gpu-preflight] gpu=${gpu_id} free_mib=${free_mib} required_mib=${minimum} utilization_percent=${utilization} max_utilization_percent=${maximum_utilization} compute_processes=${process_count}"
  if (( free_mib < minimum )); then
    echo "[gpu-preflight] insufficient free memory; existing processes were left untouched" >&2
    exit 3
  fi
  if [[ "${require_idle}" == "1" && -n "${compute_pids}" ]]; then
    echo "[gpu-preflight] GPU ${gpu_id} already has compute process(es): ${compute_pids//$'\n'/,}; existing processes were left untouched" >&2
    exit 4
  fi
  if (( utilization > maximum_utilization )); then
    echo "[gpu-preflight] GPU ${gpu_id} is busy; existing processes were left untouched" >&2
    exit 4
  fi
done
