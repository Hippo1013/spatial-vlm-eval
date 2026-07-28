#!/usr/bin/env bash
# Run GPT-5, Gemini 3.1 Pro, LLaVA-NeXT, or InternVL through a compatible endpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${LATENT_PYTHON:-python}}"

: "${PROFILE:?Set PROFILE to a locked OpenAI-compatible MSMU profile}"
: "${BACKEND:?Set BACKEND to vllm, openrouter, openai, or google}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the local MSMU dataset root}"

case "${PROFILE}" in
  gpt5)
    MODEL_REVISION_TAG="provider-managed-gpt-5"
    INFERENCE_PROTOCOL="msmu_gpt5_question_only_v1"
    ;;
  gemini31pro)
    MODEL_REVISION_TAG="provider-managed-gemini-3.1-pro-preview"
    INFERENCE_PROTOCOL="msmu_gemini31pro_question_only_v1"
    ;;
  llava_next_mistral_7b)
    MODEL_REVISION_TAG="2424fdd47412fccc66d91719126b420e9fbd7065"
    INFERENCE_PROTOCOL="msmu_llava_next_mistral_7b_question_only_v1"
    ;;
  llava_next_yi_34b)
    MODEL_REVISION_TAG="84e4488fffae48f9da316ec31288b7c03f102ec7"
    INFERENCE_PROTOCOL="msmu_llava_next_yi_34b_question_only_v1"
    ;;
  internvl3_8b)
    MODEL_REVISION_TAG="259a3b64a14623c0ec91a045cb43f7c5af5fa6af"
    INFERENCE_PROTOCOL="msmu_internvl3_8b_question_only_v1"
    ;;
  internvl3_38b)
    MODEL_REVISION_TAG="b2a05c0c325235f7530d8274c313a1d01082e069"
    INFERENCE_PROTOCOL="msmu_internvl3_38b_question_only_v1"
    ;;
  internvl3_78b)
    MODEL_REVISION_TAG="3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356"
    INFERENCE_PROTOCOL="msmu_internvl3_78b_question_only_v1"
    ;;
  *)
    echo "[msmu-infer] unsupported PROFILE=${PROFILE}" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${PROFILE}-${BACKEND}}"
source "${SCRIPT_DIR}/_run_paths.sh"
if [[ "${RESOLVE_PATHS_ONLY:-0}" == "1" ]]; then
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
  exit 0
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$(dirname "${OUTPUT}")"
LOG_PATH="${LOG_PATH:-${OUTPUT%.jsonl}.infer.log}"

args=(
  --profile "${PROFILE}"
  --backend "${BACKEND}"
  --dataset-root "${DATASET_ROOT}"
  --output "${OUTPUT}"
  --workers "${INFERENCE_WORKERS:-1}"
  --retries "${INFERENCE_RETRIES:-2}"
  --timeout "${API_TIMEOUT:-180}"
)
if [[ -n "${INFERENCE_BASE_URL:-}" ]]; then args+=(--base-url "${INFERENCE_BASE_URL}"); fi
if [[ -n "${API_KEY_ENV:-}" ]]; then args+=(--api-key-env "${API_KEY_ENV}"); fi
if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then args+=(--served-model-name "${SERVED_MODEL_NAME}"); fi
if [[ -n "${INDICES:-}" ]]; then args+=(--indices "${INDICES}"); fi
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "${LIMIT}"); fi
if [[ -n "${RUN_METADATA:-}" ]]; then args+=(--metadata "${RUN_METADATA}"); fi
if [[ -n "${JOURNAL:-}" ]]; then args+=(--journal "${JOURNAL}"); fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then args+=(--no-resume); fi

echo "[msmu-infer] profile=${PROFILE} backend=${BACKEND} output=${OUTPUT}" | tee "${LOG_PATH}"
set +e
"${PYTHON}" -m spatial_vlm_eval.models.openai_compatible.infer "${args[@]}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return "${status}"; fi
exit "${status}"
