#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

if [[ -z "${PACKYAPI_API_KEY:-}" ]]; then
  echo "[spbench-si] PACKYAPI_API_KEY is missing; run scripts/spbench_si/set_packyapi_key.sh" >&2
  exit 1
fi
if [[ -z "${https_proxy:-${HTTPS_PROXY:-}}" ]]; then
  echo "[spbench-si] proxy is not enabled in this shell; source the repository-external proxy-on.sh first" >&2
  exit 1
fi

export SPBENCH_SI_GEMINI31PRO_OPENROUTER_NON_ZDR_BACKEND=packyapi
export PACKYAPI_BASE_URL="${PACKYAPI_BASE_URL:-https://www.packyapi.com/v1}"

exec "${SPBENCH_SI_PYTHON}" -m spatial_vlm_eval.benchmarks.spbench_si.inference \
  --stage full \
  --model gemini31pro_openrouter_non_zdr \
  --resume-api-source
