#!/usr/bin/env bash
# Qwen/PEFT inference, subset-aware validation, and optional full-split scoring.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_SCRIPT="${SCRIPT_DIR}/infer_qwen_peft.sh" exec "${SCRIPT_DIR}/_run_model_pipeline.sh"
