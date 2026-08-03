#!/usr/bin/env bash
# Score one exact result or all directory-discovered pending full results.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
exec "${CVBENCH_PYTHON}" -m spatial_vlm_eval.benchmarks.cv_bench.score_results "$@"
