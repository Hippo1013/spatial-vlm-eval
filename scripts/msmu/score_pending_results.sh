#!/usr/bin/env bash
# Discover and serially score all pending MSMU stage-three prediction files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

usage() {
  cat <<'EOF'
Usage: bash scripts/msmu/score_pending_results.sh [MODE] [OPTIONS]

Modes:
  --list      list every discovered prediction file and its scoring state
  --check     check paths, the batch lock, and the configured judge model
  --status    print counts for every scoring state
  --help      show this help

Options:
  --results-root ABSOLUTE_PATH  override the stage-three results root
  --predictions ABSOLUTE_PATH   select exactly one predictions.jsonl within that root

With no mode, pending results are scored one at a time in stable path order.
Without --predictions every result below the root is discovered; with it, only
that exact result is classified and scored under the same lock and publication gates.
Set MANUAL_DRY_RUN=1 to print the frozen execution order without calling the
judge or scorer. The default results root is:

  $MANUAL_TEST_OUTPUT_ROOT/03_full987

The script loads the untracked .env.server configuration automatically.
EOF
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
esac

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/prepare_manual_test.sh"

PYTHON="${PYTHON:-${LATENT_PYTHON:-python}}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MSMU_SCORE_RESULTS_ROOT="${OUTPUT_ROOT%/}/03_full987"

exec "${PYTHON}" "${SCRIPT_DIR}/_score_pending_results.py" "$@"
