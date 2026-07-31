#!/usr/bin/env bash
# Run a locked native Qwen2.5-VL, Qwen2.5-VL+PEFT, or Qwen3-VL profile.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${LATENT_PYTHON:-python}}"

: "${BASE_MODEL:?Set BASE_MODEL to a local Qwen-VL model path}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the local MSMU dataset root}"

MODEL_REVISION_TAG="${BASE_MODEL_REVISION:-local-unspecified}"
PROFILE="${PROFILE:-qwen25_vl_7b}"
DEFAULT_RUN_NAME="qwen25-vl-peft"
DEFAULT_BATCH_SIZE=8
case "${PROFILE}" in
  qwen25_vl_7b)
    INFERENCE_PROTOCOL="msmu_qwen25_vl_question_only_deterministic_v1"
    PYTHON_MODULE="spatial_vlm_eval.models.qwen25_vl.peft_infer"
    DEFAULT_IMAGE_MIN_PIXELS=12544
    DEFAULT_IMAGE_MAX_PIXELS=112896
    ;;
  qwen25_vl_32b)
    INFERENCE_PROTOCOL="msmu_qwen25_vl_32b_question_only_deterministic_v1"
    PYTHON_MODULE="spatial_vlm_eval.models.qwen25_vl.peft_infer"
    DEFAULT_IMAGE_MIN_PIXELS=12544
    DEFAULT_IMAGE_MAX_PIXELS=112896
    ;;
  qwen25_vl_72b)
    INFERENCE_PROTOCOL="msmu_qwen25_vl_72b_question_only_deterministic_v1"
    PYTHON_MODULE="spatial_vlm_eval.models.qwen25_vl.peft_infer"
    DEFAULT_IMAGE_MIN_PIXELS=12544
    DEFAULT_IMAGE_MAX_PIXELS=112896
    ;;
  qwen3_vl_2b)
    INFERENCE_PROTOCOL="msmu_qwen3_vl_2b_question_only_deterministic_v1"
    PYTHON_MODULE="spatial_vlm_eval.models.qwen3_vl.infer"
    DEFAULT_IMAGE_MIN_PIXELS=16384
    DEFAULT_IMAGE_MAX_PIXELS=147456
    DEFAULT_RUN_NAME="qwen3-vl-2b"
    ;;
  qwen3_vl_4b)
    INFERENCE_PROTOCOL="msmu_qwen3_vl_4b_question_only_deterministic_v1"
    PYTHON_MODULE="spatial_vlm_eval.models.qwen3_vl.infer"
    DEFAULT_IMAGE_MIN_PIXELS=16384
    DEFAULT_IMAGE_MAX_PIXELS=147456
    DEFAULT_RUN_NAME="qwen3-vl-4b"
    ;;
  qwen3_vl_8b)
    INFERENCE_PROTOCOL="msmu_qwen3_vl_8b_question_only_deterministic_v1"
    PYTHON_MODULE="spatial_vlm_eval.models.qwen3_vl.infer"
    DEFAULT_IMAGE_MIN_PIXELS=16384
    DEFAULT_IMAGE_MAX_PIXELS=147456
    DEFAULT_RUN_NAME="qwen3-vl-8b"
    ;;
  qwen3_vl_32b)
    INFERENCE_PROTOCOL="msmu_qwen3_vl_32b_question_only_deterministic_v1"
    PYTHON_MODULE="spatial_vlm_eval.models.qwen3_vl.infer"
    DEFAULT_IMAGE_MIN_PIXELS=16384
    DEFAULT_IMAGE_MAX_PIXELS=147456
    DEFAULT_RUN_NAME="qwen3-vl-32b"
    DEFAULT_BATCH_SIZE=1
    ;;
  *)
    echo "[msmu-infer] unsupported Qwen profile: ${PROFILE}" >&2
    exit 2
    ;;
esac

if [[ "${PROFILE}" == qwen3_vl_* && ( -n "${CHECKPOINT:-}" || -n "${CHECKPOINT_REVISION:-}" ) ]]; then
  echo "[msmu-infer] Qwen3-VL base supplement does not accept CHECKPOINT variables" >&2
  exit 2
fi

RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
source "${SCRIPT_DIR}/_run_paths.sh"
if [[ "${RESOLVE_PATHS_ONLY:-0}" == "1" ]]; then
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
  exit 0
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$(dirname "${OUTPUT}")"
LOG_PATH="${LOG_PATH:-${OUTPUT%.jsonl}.infer.log}"

args=(
  --profile "${PROFILE}"
  --base-model "${BASE_MODEL}"
  --dataset-root "${DATASET_ROOT}"
  --output "${OUTPUT}"
  --batch-size "${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
  --max-new-tokens "${MAX_NEW_TOKENS:-192}"
  --image-min-pixels "${IMAGE_MIN_PIXELS:-${DEFAULT_IMAGE_MIN_PIXELS}}"
  --image-max-pixels "${IMAGE_MAX_PIXELS:-${DEFAULT_IMAGE_MAX_PIXELS}}"
  --device-map "${DEVICE_MAP:-single}"
  --base-model-revision "${BASE_MODEL_REVISION:-local-unspecified}"
  --retries "${INFERENCE_RETRIES:-0}"
)
if [[ -n "${CHECKPOINT:-}" ]]; then
  args+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${CHECKPOINT_REVISION:-}" ]]; then
  args+=(--checkpoint-revision "${CHECKPOINT_REVISION}")
fi
if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "${LIMIT}")
fi
if [[ -n "${INDICES:-}" ]]; then
  args+=(--indices "${INDICES}")
fi
if [[ -n "${RUN_METADATA:-}" ]]; then
  args+=(--metadata "${RUN_METADATA}")
fi
if [[ -n "${JOURNAL:-}" ]]; then
  args+=(--journal "${JOURNAL}")
fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then
  args+=(--no-resume)
fi
if [[ "${QWEN_VISION_CANARY:-0}" == "1" ]]; then
  args+=(--vision-canary-report "${VISION_CANARY_REPORT:-${RUN_DIR}/vision_canary.json}")
fi

echo "[msmu-infer] profile=${PROFILE} output=${OUTPUT} checkpoint=${CHECKPOINT:-<base-model>}" \
  | tee "${LOG_PATH}"
set +e
"${PYTHON}" -m "${PYTHON_MODULE}" "${args[@]}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e
echo "[msmu-infer] exit_code=${status}" | tee -a "${LOG_PATH}"
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return "${status}"; fi
exit "${status}"
