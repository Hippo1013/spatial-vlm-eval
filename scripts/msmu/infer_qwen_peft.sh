#!/usr/bin/env bash
# Run deterministic Qwen2.5-VL or Qwen2.5-VL+PEFT inference on MSMU-Bench.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

: "${BASE_MODEL:?Set BASE_MODEL to a local Qwen2.5-VL model path}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the local MSMU dataset root}"
: "${OUTPUT:?Set OUTPUT to the destination predictions.jsonl}"

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
)
if [[ -n "${CHECKPOINT:-}" ]]; then
  args+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "${LIMIT}")
fi
if [[ -n "${RUN_METADATA:-}" ]]; then
  args+=(--run-metadata "${RUN_METADATA}")
fi

echo "[msmu-infer] output=${OUTPUT} checkpoint=${CHECKPOINT:-<base-model>}" | tee "${LOG_PATH}"
set +e
"${PYTHON}" -m spatial_vlm_eval.models.qwen25_vl.peft_infer "${args[@]}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e
echo "[msmu-infer] exit_code=${status}" | tee -a "${LOG_PATH}"
exit "${status}"
