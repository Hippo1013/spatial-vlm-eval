#!/usr/bin/env bash
# Run the frozen two-GPU MSMU SOTA supplement and one post-inference judge.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/msmu/run_sota_supplement.sh
  bash scripts/msmu/run_sota_supplement.sh --check
  bash scripts/msmu/run_sota_supplement.sh --status
  bash scripts/msmu/run_sota_supplement.sh --list

The frozen lanes are GPU0: RoboBrain NV -> HiSpatial -> SpatialLadder direct,
and GPU1: RoboBrain MT -> SpatialLadder thinking. Both lanes must complete all
canary, smoke8, and full-987 gates before one local judge is started. Set
MANUAL_DRY_RUN=1 to print the workflow without using GPUs or writing a report.
EOF
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  --list)
    export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    CONTROLLER_PYTHON="${LATENT_PYTHON:-${PYTHON:-python}}"
    exec "${CONTROLLER_PYTHON}" "${SCRIPT_DIR}/_run_sota_supplement.py" --list
    ;;
esac

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/prepare_manual_test.sh"

CONTROLLER_PYTHON="${LATENT_PYTHON:-${PYTHON:-python}}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MSMU_SOTA_MANUAL_OUTPUT_ROOT="${OUTPUT_ROOT}"

exec "${CONTROLLER_PYTHON}" "${SCRIPT_DIR}/_run_sota_supplement.py" "$@"
