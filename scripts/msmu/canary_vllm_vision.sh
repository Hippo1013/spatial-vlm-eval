#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${LATENT_PYTHON:-python}}"
: "${PROFILE:?Set PROFILE to the running LLaVA-NeXT or InternVL3 profile}"
: "${SERVED_MODEL_NAME:?Set SERVED_MODEL_NAME to the vLLM served name}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
args=(--profile "${PROFILE}" --served-model-name "${SERVED_MODEL_NAME}" --base-url "${INFERENCE_BASE_URL:-http://127.0.0.1:18081/v1}" --timeout "${API_TIMEOUT:-180}")
if [[ -n "${CANARY_REPORT:-}" ]]; then args+=(--output "${CANARY_REPORT}"); fi
exec "${PYTHON}" -m spatial_vlm_eval.models.openai_compatible.vision_canary "${args[@]}"
