#!/usr/bin/env bash
# Run deterministic Qwen2.5-VL or Qwen2.5-VL+PEFT inference on MSMU-Bench.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${LATENT_PYTHON:-python}}"

: "${BASE_MODEL:?Set BASE_MODEL to a local Qwen2.5-VL model path}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the local MSMU dataset root}"

MODEL_REVISION_TAG="${BASE_MODEL_REVISION:-local-unspecified}"
INFERENCE_PROTOCOL="msmu_qwen25_vl_question_only_deterministic_v1"
RUN_NAME="${RUN_NAME:-qwen25-vl-peft}"
source "${SCRIPT_DIR}/_run_paths.sh"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$(dirname "${OUTPUT}")"
LOG_PATH="${LOG_PATH:-${OUTPUT%.jsonl}.infer.log}"

args=(
  --base-model "${BASE_MODEL}"
  --dataset-root "${DATASET_ROOT}"
  --output "${OUTPUT}"
  --batch-size "${BATCH_SIZE:-8}"
  --max-new-tokens "${MAX_NEW_TOKENS:-192}"
  --image-min-pixels "${IMAGE_MIN_PIXELS:-12544}"
  --image-max-pixels "${IMAGE_MAX_PIXELS:-112896}"
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

echo "[msmu-infer] output=${OUTPUT} checkpoint=${CHECKPOINT:-<base-model>}" | tee "${LOG_PATH}"
set +e
"${PYTHON}" -m spatial_vlm_eval.models.qwen25_vl.peft_infer "${args[@]}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e
echo "[msmu-infer] exit_code=${status}" | tee -a "${LOG_PATH}"
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return "${status}"; fi
exit "${status}"
