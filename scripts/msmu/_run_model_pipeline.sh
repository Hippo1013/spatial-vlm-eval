#!/usr/bin/env bash
# Shared validation/scoring orchestration. INFER_SCRIPT must name one family wrapper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${INFER_SCRIPT:?Set INFER_SCRIPT to an MSMU inference wrapper}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the local MSMU dataset root}"

# Source the family wrapper so its derived OUTPUT/RUN_DIR values remain visible
# for validation and scoring. Every inference wrapper also remains directly
# executable and returns here instead of exiting when sourced.
if [[ "${SCORE_ONLY:-0}" == "1" ]]; then
  if [[ "${RUN_SCORE:-0}" != "1" ]]; then
    echo "[msmu-pipeline] SCORE_ONLY=1 requires RUN_SCORE=1" >&2
    exit 2
  fi
  RESOLVE_PATHS_ONLY=1 source "${INFER_SCRIPT}"
else
  RESOLVE_PATHS_ONLY=0 source "${INFER_SCRIPT}"
fi

: "${OUTPUT:?Inference wrapper did not resolve OUTPUT}"

subset=0
if [[ -n "${LIMIT:-}" || -n "${INDICES:-}" ]]; then
  subset=1
fi

PREDICTIONS="${OUTPUT}" \
DATASET_ROOT="${DATASET_ROOT}" \
REPORT="${VALIDATION_REPORT:-$(dirname "${OUTPUT}")/prediction_validation.json}" \
ALLOW_SUBSET="${subset}" \
  "${SCRIPT_DIR}/validate_predictions.sh"

if [[ "${RUN_SCORE:-0}" == "1" ]]; then
  if [[ "${subset}" == "1" ]]; then
    echo "[msmu-pipeline] refusing to score a debug subset" >&2
    exit 2
  fi
  : "${SCORE_OUTPUT_DIR:?Set SCORE_OUTPUT_DIR when RUN_SCORE=1}"
  : "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL when RUN_SCORE=1; do not reuse the inference endpoint}"
  PREDICTIONS="${OUTPUT}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  OUTPUT_DIR="${SCORE_OUTPUT_DIR}" \
  BASE_URL="${JUDGE_BASE_URL}" \
    "${SCRIPT_DIR}/score_predictions.sh"
fi
