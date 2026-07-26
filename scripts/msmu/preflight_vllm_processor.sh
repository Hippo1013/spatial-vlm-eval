#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${VLLM_PYTHON:-python}}"
: "${PROFILE:?Set PROFILE to a LLaVA-NeXT or InternVL3 profile}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
args=(--profile "${PROFILE}")
if [[ -n "${MODEL_PATH:-}" ]]; then args+=(--model "${MODEL_PATH}"); fi
if [[ -n "${CANARY_IMAGE:-}" ]]; then args+=(--image "${CANARY_IMAGE}"); fi
if [[ -n "${PREFLIGHT_REPORT:-}" ]]; then args+=(--output "${PREFLIGHT_REPORT}"); fi
exec "${PYTHON}" -m spatial_vlm_eval.models.openai_compatible.processor_preflight "${args[@]}"
