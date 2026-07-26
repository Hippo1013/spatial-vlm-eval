#!/usr/bin/env bash
# Validate one predictions.jsonl against the complete official MSMU test split.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${LATENT_PYTHON:-python}}"

: "${PREDICTIONS:?Set PREDICTIONS to predictions.jsonl}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the local MSMU dataset root}"

REPORT="${REPORT:-$(dirname "${PREDICTIONS}")/prediction_validation.json}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  --predictions "${PREDICTIONS}"
  --dataset-root "${DATASET_ROOT}"
  --report "${REPORT}"
)
if [[ "${ALLOW_SUBSET:-0}" == "1" ]]; then
  args+=(--allow-subset)
fi

"${PYTHON}" -m spatial_vlm_eval.benchmarks.msmu.prediction_validation "${args[@]}"
