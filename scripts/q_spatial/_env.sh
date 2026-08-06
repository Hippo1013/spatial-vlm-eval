#!/usr/bin/env bash
# Shared environment loader for Q-Spatial public entrypoints.

set -euo pipefail

QSPATIAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
QSPATIAL_REPO_ROOT="$(cd "${QSPATIAL_SCRIPT_DIR}/../.." && pwd -P)"
QSPATIAL_ENV_FILE="${QSPATIAL_ENV_FILE:-${QSPATIAL_REPO_ROOT}/.env.server}"
if [[ -f "${QSPATIAL_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${QSPATIAL_ENV_FILE}"
  set +a
fi
export PYTHONPATH="${QSPATIAL_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
QSPATIAL_PYTHON="${QSPATIAL_PYTHON:-${PYTHON:-${LATENT_PYTHON:-python}}}"
