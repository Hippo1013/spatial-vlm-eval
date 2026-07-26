#!/usr/bin/env bash
# Validate and score one MSMU prediction file with the v3 local-judge protocol.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${LATENT_PYTHON:-python}}"

: "${PREDICTIONS:?Set PREDICTIONS to predictions.jsonl}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the local MSMU dataset root}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a model/protocol-specific score directory}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

mkdir -p "${OUTPUT_DIR}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/score.log}"

echo "[msmu-score] predictions=${PREDICTIONS} output_dir=${OUTPUT_DIR}" | tee "${LOG_PATH}"
set +e
"${PYTHON}" -m spatial_vlm_eval.benchmarks.msmu.scorer \
  --predictions "${PREDICTIONS}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-url "${BASE_URL:-http://127.0.0.1:18080/v1}" \
  --model "${JUDGE_MODEL_NAME:-msmu-judge}" \
  --api-key "${API_KEY:-local}" \
  --workers "${WORKERS:-16}" \
  --retries "${RETRIES:-4}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e
echo "[msmu-score] exit_code=${status}" | tee -a "${LOG_PATH}"
exit "${status}"
