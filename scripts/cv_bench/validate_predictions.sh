#!/usr/bin/env bash
# Validate a CV-Bench prediction JSONL against the locked two-Parquet dataset.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
exec "${CVBENCH_PYTHON}" -m spatial_vlm_eval.benchmarks.cv_bench.prediction_validation "$@"
