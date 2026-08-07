#!/usr/bin/env bash
# Run the frozen two-GPU/API staged Q-Spatial schedule without scoring.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

export QSPATIAL_REPO_ROOT
exec "${QSPATIAL_PYTHON}" -m spatial_vlm_eval.benchmarks.q_spatial.scheduled_batch "$@"
