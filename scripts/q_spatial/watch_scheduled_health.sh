#!/usr/bin/env bash
# Read-only PASS/FAIL/COMPLETE and conservative anomaly watcher for one lane.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

exec "${QSPATIAL_PYTHON}" -m spatial_vlm_eval.benchmarks.q_spatial.scheduled_watcher "$@"
