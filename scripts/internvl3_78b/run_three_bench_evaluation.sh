#!/usr/bin/env bash
# Public entrypoint for one InternVL3-78B service shared by three benchmarks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
INTERNVL3_78B_THREE_BENCH_ENV_FILE="${INTERNVL3_78B_THREE_BENCH_ENV_FILE:-${REPOSITORY_ROOT}/.env.server}"
if [[ -f "${INTERNVL3_78B_THREE_BENCH_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${INTERNVL3_78B_THREE_BENCH_ENV_FILE}"
  set +a
fi
export INTERNVL3_78B_THREE_BENCH_ENV_FILE
export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
THREE_BENCH_PYTHON="${LATENT_PYTHON:-${PYTHON:-python}}"

exec "${THREE_BENCH_PYTHON}" \
  -m spatial_vlm_eval.orchestration.internvl3_78b_three_bench "$@"
