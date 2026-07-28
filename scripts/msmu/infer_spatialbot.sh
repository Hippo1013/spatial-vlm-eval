#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${SPATIALBOT_PYTHON:-python}}"
: "${PROFILE:?Set PROFILE to spatialbot or spatialbot_native}"
: "${SPATIALBOT_UPSTREAM_ROOT:?Set SPATIALBOT_UPSTREAM_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH to the accepted/downloaded SpatialBot-3B snapshot}"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
MODEL_REVISION_TAG="41d3b52c642058dfb087885bec0b8e37e0e67f8d"
case "${PROFILE}" in
  spatialbot)
    INFERENCE_PROTOCOL="msmu_spatialbot_rgb_only_v1"
    ;;
  spatialbot_native)
    INFERENCE_PROTOCOL="msmu_spatialbot_native_zoedepth_rgbd_native_v1"
    : "${ZOEDEPTH_ROOT:?Set ZOEDEPTH_ROOT for spatialbot_native}"
    : "${ZOEDEPTH_CHECKPOINT:?Set ZOEDEPTH_CHECKPOINT for spatialbot_native}"
    ;;
  *) echo "Unsupported SpatialBot PROFILE=${PROFILE}" >&2; exit 2 ;;
esac
RUN_NAME="${RUN_NAME:-${PROFILE}-3b}"
source "${SCRIPT_DIR}/_run_paths.sh"
if [[ "${RESOLVE_PATHS_ONLY:-0}" == "1" ]]; then
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
  exit 0
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-24000}" "${SCRIPT_DIR}/gpu_preflight.sh"
mkdir -p "$(dirname "${OUTPUT}")"
LOG_PATH="${LOG_PATH:-${OUTPUT%.jsonl}.infer.log}"
args=(--profile "${PROFILE}" --upstream-root "${SPATIALBOT_UPSTREAM_ROOT}" --model "${MODEL_PATH}" --dataset-root "${DATASET_ROOT}" --output "${OUTPUT}" --model-type phi-2 --conversation-mode bunny --device "${DEVICE:-cuda}" --retries "${INFERENCE_RETRIES:-0}")
if [[ -n "${MODEL_BASE:-}" ]]; then args+=(--model-base "${MODEL_BASE}"); fi
if [[ "${PROFILE}" == "spatialbot_native" ]]; then args+=(--zoedepth-root "${ZOEDEPTH_ROOT}" --zoedepth-revision "${ZOEDEPTH_REVISION:-d87f17b2f5fdcb174cf4fb115491f4a6c60de152}" --zoedepth-checkpoint "${ZOEDEPTH_CHECKPOINT}"); fi
if [[ -n "${INDICES:-}" ]]; then args+=(--indices "${INDICES}"); fi
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "${LIMIT}"); fi
if [[ -n "${RUN_METADATA:-}" ]]; then args+=(--metadata "${RUN_METADATA}"); fi
if [[ -n "${JOURNAL:-}" ]]; then args+=(--journal "${JOURNAL}"); fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then args+=(--no-resume); fi
echo "[msmu-spatialbot] profile=${PROFILE} output=${OUTPUT}" | tee "${LOG_PATH}"
set +e
"${PYTHON}" -m spatial_vlm_eval.models.spatialbot.infer "${args[@]}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return "${status}"; fi
exit "${status}"
