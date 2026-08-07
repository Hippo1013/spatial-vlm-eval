#!/usr/bin/env bash
set -euo pipefail

SPBENCH_SI_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SPBENCH_SI_REPO_ROOT="$(cd "${SPBENCH_SI_SCRIPT_DIR}/../.." && pwd -P)"
SPBENCH_SI_ENV_FILE="${SPBENCH_SI_ENV_FILE:-${SPBENCH_SI_REPO_ROOT}/.env.server}"

# The scheduler supplies these values per job. A shared server env may also
# define defaults, but must not replace the lane's GPU assignment or a
# checkpoint-specific context limit when the child script sources it again.
spbench_si_job_cuda_set="${CUDA_VISIBLE_DEVICES+x}"
spbench_si_job_cuda="${CUDA_VISIBLE_DEVICES-}"
spbench_si_job_max_len_set="${SPBENCH_SI_VLLM_MAX_MODEL_LEN+x}"
spbench_si_job_max_len="${SPBENCH_SI_VLLM_MAX_MODEL_LEN-}"
if [[ -f "${SPBENCH_SI_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SPBENCH_SI_ENV_FILE}"
  set +a
fi
if [[ -n "${spbench_si_job_cuda_set}" ]]; then
  export CUDA_VISIBLE_DEVICES="${spbench_si_job_cuda}"
fi
if [[ -n "${spbench_si_job_max_len_set}" ]]; then
  export SPBENCH_SI_VLLM_MAX_MODEL_LEN="${spbench_si_job_max_len}"
fi
unset spbench_si_job_cuda_set spbench_si_job_cuda
unset spbench_si_job_max_len_set spbench_si_job_max_len
export PYTHONPATH="${SPBENCH_SI_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
SPBENCH_SI_PYTHON="${SPBENCH_SI_PYTHON:-${PYTHON:-${LATENT_PYTHON:-python}}}"
