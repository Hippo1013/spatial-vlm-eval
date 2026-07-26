#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${SPATIALRGPT_PYTHON:-python}}"
: "${SPATIALRGPT_UPSTREAM_ROOT:?Set SPATIALRGPT_UPSTREAM_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH to SpatialRGPT-VILA1.5-8B}"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
MODEL_REVISION_TAG="64df7902f82b5053f5a53455095805e6de3a1f87"
INFERENCE_PROTOCOL="msmu_spatialrgpt_rgb_only_v1"
RUN_NAME="${RUN_NAME:-spatialrgpt-vila1.5-8b}"
source "${SCRIPT_DIR}/_run_paths.sh"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-30000}" "${SCRIPT_DIR}/gpu_preflight.sh"
mkdir -p "$(dirname "${OUTPUT}")"
LOG_PATH="${LOG_PATH:-${OUTPUT%.jsonl}.infer.log}"
args=(--upstream-root "${SPATIALRGPT_UPSTREAM_ROOT}" --model "${MODEL_PATH}" --dataset-root "${DATASET_ROOT}" --output "${OUTPUT}" --conversation-mode llama_3 --device "${DEVICE:-cuda}" --retries "${INFERENCE_RETRIES:-0}")
if [[ -n "${MODEL_BASE:-}" ]]; then args+=(--model-base "${MODEL_BASE}"); fi
if [[ -n "${INDICES:-}" ]]; then args+=(--indices "${INDICES}"); fi
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "${LIMIT}"); fi
if [[ -n "${RUN_METADATA:-}" ]]; then args+=(--metadata "${RUN_METADATA}"); fi
if [[ -n "${JOURNAL:-}" ]]; then args+=(--journal "${JOURNAL}"); fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then args+=(--no-resume); fi
echo "[msmu-spatialrgpt] output=${OUTPUT}" | tee "${LOG_PATH}"
set +e
"${PYTHON}" -m spatial_vlm_eval.models.spatialrgpt.infer "${args[@]}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return "${status}"; fi
exit "${status}"
