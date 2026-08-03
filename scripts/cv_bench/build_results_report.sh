#!/usr/bin/env bash
# Rebuild the publication-gated CV-Bench Markdown report.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
exec "${CVBENCH_PYTHON}" -m spatial_vlm_eval.benchmarks.cv_bench.report "$@"
