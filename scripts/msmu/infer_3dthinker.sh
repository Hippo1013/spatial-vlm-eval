#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${THREEDTHINKER_PYTHON:-python}}"
: "${PROFILE:?Set PROFILE to 3dthinker or 3dthinker_native}"
: "${THREEDTHINKER_UPSTREAM_ROOT:?Set THREEDTHINKER_UPSTREAM_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH to 3DThinker-Mindcube}"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
MODEL_REVISION_TAG="69a70411605f86ec69bada0a625bb96ddee995d9"
case "${PROFILE}" in
  3dthinker) INFERENCE_PROTOCOL="msmu_3dthinker_question_only_v1" ;;
  3dthinker_native) INFERENCE_PROTOCOL="msmu_3dthinker_native_mental3d_native_v1" ;;
  *) echo "Unsupported 3DThinker PROFILE=${PROFILE}" >&2; exit 2 ;;
esac
RUN_NAME="${RUN_NAME:-${PROFILE}-mindcube-stage1}"
source "${SCRIPT_DIR}/_run_paths.sh"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-24000}" "${SCRIPT_DIR}/gpu_preflight.sh"
mkdir -p "$(dirname "${OUTPUT}")"
LOG_PATH="${LOG_PATH:-${OUTPUT%.jsonl}.infer.log}"
args=(--profile "${PROFILE}" --upstream-root "${THREEDTHINKER_UPSTREAM_ROOT}" --model "${MODEL_PATH}" --dataset-root "${DATASET_ROOT}" --output "${OUTPUT}" --device-map "${DEVICE_MAP:-auto}" --retries "${INFERENCE_RETRIES:-0}")
if [[ -n "${INDICES:-}" ]]; then args+=(--indices "${INDICES}"); fi
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "${LIMIT}"); fi
if [[ -n "${RUN_METADATA:-}" ]]; then args+=(--metadata "${RUN_METADATA}"); fi
if [[ -n "${JOURNAL:-}" ]]; then args+=(--journal "${JOURNAL}"); fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then args+=(--no-resume); fi
echo "[msmu-3dthinker] profile=${PROFILE} output=${OUTPUT}" | tee "${LOG_PATH}"
set +e
"${PYTHON}" -m spatial_vlm_eval.models.three_d_thinker.infer "${args[@]}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return "${status}"; fi
exit "${status}"
