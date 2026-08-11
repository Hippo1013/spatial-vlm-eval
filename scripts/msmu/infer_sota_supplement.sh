#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

: "${PROFILE:?Set PROFILE to one registered MSMU SOTA supplement profile}"
: "${MODEL_PATH:?Set MODEL_PATH to the locked local model snapshot}"
: "${DATASET_ROOT:?Set DATASET_ROOT}"

case "${PROFILE}" in
  robobrain25_8b_nv_rgb)
    PYTHON="${ROBOBRAIN25_PYTHON:?Set ROBOBRAIN25_PYTHON}"
    UPSTREAM_PATH="${ROBOBRAIN25_UPSTREAM_ROOT:?Set ROBOBRAIN25_UPSTREAM_ROOT}"
    MODEL_REVISION_TAG="3d77a19a3ddd8616b3979e03de56096edfb12ff6"
    INFERENCE_PROTOCOL="msmu_robobrain25_8b_nv_rgb_original_first_question_official_general_sampling_t07_top_p08_768_v1"
    ;;
  robobrain25_8b_mt_rgb)
    PYTHON="${ROBOBRAIN25_PYTHON:?Set ROBOBRAIN25_PYTHON}"
    UPSTREAM_PATH="${ROBOBRAIN25_UPSTREAM_ROOT:?Set ROBOBRAIN25_UPSTREAM_ROOT}"
    MODEL_REVISION_TAG="01145b89a0fe49f78f5d677d25af7351088d7c7d"
    INFERENCE_PROTOCOL="msmu_robobrain25_8b_mt_rgb_original_first_question_official_general_sampling_t07_top_p08_768_v1"
    ;;
  hispatial3b_moge2_xyz)
    PYTHON="${HISPATIAL_PYTHON:?Set HISPATIAL_PYTHON}"
    UPSTREAM_PATH="${HISPATIAL_UPSTREAM_ROOT:?Set HISPATIAL_UPSTREAM_ROOT}"
    : "${MOGE2_MODEL:?Set MOGE2_MODEL}"
    : "${MOGE2_UPSTREAM_ROOT:?Set MOGE2_UPSTREAM_ROOT}"
    : "${MOGE2_UTILS3D_ROOT:?Set MOGE2_UTILS3D_ROOT}"
    MODEL_REVISION_TAG="75a5e3d65351d7602c492aa91533f62b8a252604"
    INFERENCE_PROTOCOL="msmu_hispatial3b_same_rgb_moge2_xyz_original_first_question_official_predictor_greedy100_v1"
    ;;
  spatialladder3b_rgb)
    PYTHON="${SPATIALLADDER_PYTHON:?Set SPATIALLADDER_PYTHON}"
    UPSTREAM_PATH="${SPATIALLADDER_UPSTREAM_ROOT:?Set SPATIALLADDER_UPSTREAM_ROOT}"
    MODEL_REVISION_TAG="0819c3adf8827a2ea6c0348d49a23503ecb1f428"
    INFERENCE_PROTOCOL="msmu_spatialladder3b_rgb_original_first_question_direct_flashattn2_leftpad_native_batch_128_v1"
    ;;
  spatialladder3b_thinking)
    PYTHON="${SPATIALLADDER_PYTHON:?Set SPATIALLADDER_PYTHON}"
    UPSTREAM_PATH="${SPATIALLADDER_UPSTREAM_ROOT:?Set SPATIALLADDER_UPSTREAM_ROOT}"
    MODEL_REVISION_TAG="0819c3adf8827a2ea6c0348d49a23503ecb1f428"
    INFERENCE_PROTOCOL="msmu_spatialladder3b_rgb_official_generic_special_thinking_flashattn2_leftpad_native_batch_last_answer_1024_v1"
    ;;
  *)
    echo "Unsupported MSMU SOTA supplement PROFILE=${PROFILE}" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${PROFILE}}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_run_paths.sh"
if [[ "${RESOLVE_PATHS_ONLY:-0}" == "1" ]]; then
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
  exit 0
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-70000}" "${SCRIPT_DIR}/gpu_preflight.sh"

mkdir -p "$(dirname "${OUTPUT}")"
LOG_PATH="${LOG_PATH:-${OUTPUT%.jsonl}.infer.log}"
args=(
  --profile "${PROFILE}"
  --model "${MODEL_PATH}"
  --upstream-root "${UPSTREAM_PATH}"
  --dataset-root "${DATASET_ROOT}"
  --output "${OUTPUT}"
  --batch-size "${BATCH_SIZE:-1}"
  --retries "${INFERENCE_RETRIES:-0}"
)
if [[ "${PROFILE}" == "hispatial3b_moge2_xyz" ]]; then
  args+=(
    --moge-model "${MOGE2_MODEL}"
    --moge-upstream-root "${MOGE2_UPSTREAM_ROOT}"
    --moge-utils3d-root "${MOGE2_UTILS3D_ROOT}"
  )
fi
if [[ "${SOTA_VISION_CANARY:-0}" == "1" ]]; then
  args+=(--vision-canary-report "${RUN_DIR}/vision_canary.json")
fi
if [[ -n "${INDICES:-}" ]]; then args+=(--indices "${INDICES}"); fi
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "${LIMIT}"); fi
if [[ -n "${RUN_METADATA:-}" ]]; then args+=(--metadata "${RUN_METADATA}"); fi
if [[ -n "${JOURNAL:-}" ]]; then args+=(--journal "${JOURNAL}"); fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then args+=(--no-resume); fi

echo "[msmu-sota-supplement] profile=${PROFILE} output=${OUTPUT}" | tee "${LOG_PATH}"
set +e
"${PYTHON}" -m spatial_vlm_eval.models.sota_spatial.cli "${args[@]}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return "${status}"; fi
exit "${status}"
