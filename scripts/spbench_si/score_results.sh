#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
exec "${SPBENCH_SI_PYTHON}" -m spatial_vlm_eval.benchmarks.spbench_si.score_results "$@"
