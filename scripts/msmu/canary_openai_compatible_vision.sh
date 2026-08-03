#!/usr/bin/env bash
# One paid generation at most: prove an OpenAI-compatible model reads one image.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${LATENT_PYTHON:-python}}"

: "${PROFILE:?Set PROFILE to a locked OpenAI-compatible MSMU profile}"
: "${BACKEND:?Set BACKEND to vllm, openrouter, openai, or google}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
args=(
  --profile "${PROFILE}"
  --backend "${BACKEND}"
  --timeout "${API_TIMEOUT:-180}"
  --metadata-retries "${OPENROUTER_METADATA_RETRIES:-10}"
)
if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then args+=(--served-model-name "${SERVED_MODEL_NAME}"); fi
if [[ -n "${INFERENCE_BASE_URL:-}" ]]; then args+=(--base-url "${INFERENCE_BASE_URL}"); fi
if [[ -n "${API_KEY_ENV:-}" ]]; then args+=(--api-key-env "${API_KEY_ENV}"); fi
if [[ -n "${CANARY_REPORT:-}" ]]; then args+=(--output "${CANARY_REPORT}"); fi

exec "${PYTHON}" -m spatial_vlm_eval.models.openai_compatible.vision_canary "${args[@]}"
