#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_SCRIPT="${SCRIPT_DIR}/infer_spatialbot.sh" exec "${SCRIPT_DIR}/_run_model_pipeline.sh"
