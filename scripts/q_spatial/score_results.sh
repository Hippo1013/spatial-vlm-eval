#!/usr/bin/env bash
# Directory-driven, locked scoring for full Q-Spatial prediction tracks.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
exec "${QSPATIAL_PYTHON}" -m spatial_vlm_eval.benchmarks.q_spatial.score_results "$@"
