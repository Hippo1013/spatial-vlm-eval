#!/usr/bin/env bash
# Build the publication-gated 21-track Q-Spatial report.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
exec "${QSPATIAL_PYTHON}" -m spatial_vlm_eval.benchmarks.q_spatial.report "$@"
