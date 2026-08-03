#!/usr/bin/env bash
# Shared environment loader for CV-Bench public entrypoints.

set -euo pipefail

CVBENCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CVBENCH_REPO_ROOT="$(cd "${CVBENCH_SCRIPT_DIR}/../.." && pwd -P)"
CVBENCH_ENV_FILE="${CVBENCH_ENV_FILE:-${CVBENCH_REPO_ROOT}/.env.server}"
if [[ -f "${CVBENCH_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CVBENCH_ENV_FILE}"
  set +a
fi
export PYTHONPATH="${CVBENCH_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
CVBENCH_PYTHON="${CVBENCH_PYTHON:-${PYTHON:-python}}"
