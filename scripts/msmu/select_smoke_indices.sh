#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${LATENT_PYTHON:-python}}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the official MSMU dataset root}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
args=(--dataset-root "${DATASET_ROOT}" --format "${FORMAT:-csv}")
if [[ -n "${SMOKE_INDEX_REPORT:-}" ]]; then args+=(--output "${SMOKE_INDEX_REPORT}"); fi
exec "${PYTHON}" -m spatial_vlm_eval.benchmarks.msmu.smoke_indices "${args[@]}"
