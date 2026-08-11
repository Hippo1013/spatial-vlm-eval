#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
INFER_SCRIPT="${SCRIPT_DIR}/infer_sota_supplement.sh" exec "${SCRIPT_DIR}/_run_model_pipeline.sh"
