#!/usr/bin/env bash
# Build one Markdown table from publication-gated MSMU scoring results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

usage() {
  cat <<'EOF'
Usage: bash scripts/msmu/build_results_report.sh [OPTIONS]

Options:
  --list                       list all discovered score summaries
  --results-root ABSOLUTE_PATH override the stage-three results root
  --output ABSOLUTE_PATH       override the Markdown output path
  --profile PROFILE            select one metadata profile; repeatable
  --scorer-protocol PROTOCOL   select one scorer protocol; repeatable
  --help                       show this help

With no scorer protocol filter, the current canonical protocol is selected.
The concise table accepts exactly one scorer protocol. Profile filters may be
repeated or comma-separated.
The default output is:

  $MANUAL_TEST_OUTPUT_ROOT/03_full987/msmu-result.md

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
export MSMU_REPORT_RESULTS_ROOT="${OUTPUT_ROOT%/}/03_full987"

exec "${PYTHON}" "${SCRIPT_DIR}/_build_results_report.py" "$@"
