#!/usr/bin/env bash
# Source this file after setting RUN_NAME, MODEL_REVISION_TAG, and INFERENCE_PROTOCOL.

SCORER_PROTOCOL="${SCORER_PROTOCOL:-sdvlm_official_compat_local_judge_v3_grounding_split_strict_quant_length}"

if [[ -z "${OUTPUT:-}" ]]; then
  : "${OUTPUT_ROOT:?Set OUTPUT_ROOT when OUTPUT is not provided}"
  : "${RUN_NAME:?Set RUN_NAME when OUTPUT is not provided}"
  : "${MODEL_REVISION_TAG:?Set MODEL_REVISION_TAG when OUTPUT is not provided}"
  : "${INFERENCE_PROTOCOL:?Set INFERENCE_PROTOCOL when OUTPUT is not provided}"
  RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}/${MODEL_REVISION_TAG}/${INFERENCE_PROTOCOL}/${SCORER_PROTOCOL}"
  OUTPUT="${RUN_DIR}/predictions.jsonl"
else
  RUN_DIR="$(dirname "${OUTPUT}")"
fi

VALIDATION_REPORT="${VALIDATION_REPORT:-${RUN_DIR}/prediction_validation.json}"
SCORE_OUTPUT_DIR="${SCORE_OUTPUT_DIR:-${RUN_DIR}/scores/${SCORER_PROTOCOL}}"
export OUTPUT RUN_DIR VALIDATION_REPORT SCORE_OUTPUT_DIR SCORER_PROTOCOL
