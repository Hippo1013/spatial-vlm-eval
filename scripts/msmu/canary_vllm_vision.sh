#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROFILE:?Set PROFILE to the running LLaVA-NeXT or InternVL3 profile}"
: "${SERVED_MODEL_NAME:?Set SERVED_MODEL_NAME to the vLLM served name}"
export BACKEND=vllm
export INFERENCE_BASE_URL="${INFERENCE_BASE_URL:-http://127.0.0.1:18081/v1}"
exec "${SCRIPT_DIR}/canary_openai_compatible_vision.sh"
