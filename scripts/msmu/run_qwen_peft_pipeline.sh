#!/usr/bin/env bash
# Orchestrate Qwen/PEFT inference, validation, and optional v3 scoring.
# Start scripts/msmu/serve_local_judge.sh separately before RUN_SCORE=1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${OUTPUT:?Set OUTPUT to the destination predictions.jsonl}"

"${SCRIPT_DIR}/infer_qwen_peft.sh"

PREDICTIONS="${OUTPUT}" \
DATASET_ROOT="${DATASET_ROOT}" \
REPORT="${VALIDATION_REPORT:-$(dirname "${OUTPUT}")/prediction_validation.json}" \
  "${SCRIPT_DIR}/validate_predictions.sh"

if [[ "${RUN_SCORE:-0}" == "1" ]]; then
  : "${SCORE_OUTPUT_DIR:?Set SCORE_OUTPUT_DIR when RUN_SCORE=1}"
  PREDICTIONS="${OUTPUT}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  OUTPUT_DIR="${SCORE_OUTPUT_DIR}" \
    "${SCRIPT_DIR}/score_predictions.sh"
fi
