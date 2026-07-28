#!/usr/bin/env bash
# Source before hand-written commands; the run_manual_stage*.sh wrappers source it automatically.

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "[msmu-prepare] use Bash and run: source scripts/msmu/prepare_manual_test.sh" >&2
  return 2 2>/dev/null || exit 2
fi

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[msmu-prepare] this script must be sourced so its exports remain in the current shell:" >&2
  echo "  source scripts/msmu/prepare_manual_test.sh" >&2
  exit 2
fi

_msmu_prepare_manual_test() {
  local script_dir repository env_file configured_repository manual_output
  local had_allexport=0 source_status=0

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || return 2
  repository="$(cd "${script_dir}/../.." && pwd -P)" || return 2
  env_file="${MSMU_SERVER_ENV:-${repository}/.env.server}"

  if [[ ! -f "${env_file}" ]]; then
    echo "[msmu-prepare] missing untracked server config: ${env_file}" >&2
    echo "[msmu-prepare] copy configs/msmu-server.env.example to .env.server and fill the paths first" >&2
    return 2
  fi

  case "$-" in
    *a*) had_allexport=1 ;;
  esac
  set -a
  # shellcheck disable=SC1090
  source "${env_file}" || source_status=$?
  if [[ "${had_allexport}" == "0" ]]; then
    set +a
  fi
  if [[ "${source_status}" != "0" ]]; then
    echo "[msmu-prepare] could not load ${env_file}" >&2
    return "${source_status}"
  fi

  if [[ -n "${REPO_ROOT:-}" ]]; then
    configured_repository="$(cd "${REPO_ROOT}" 2>/dev/null && pwd -P)" || {
      echo "[msmu-prepare] configured REPO_ROOT does not exist: ${REPO_ROOT}" >&2
      return 2
    }
    if [[ "${configured_repository}" != "${repository}" ]]; then
      echo "[msmu-prepare] REPO_ROOT points to ${configured_repository}, but this script is in ${repository}" >&2
      return 2
    fi
  fi
  export REPO_ROOT="${repository}"
  if [[ -z "${QWEN_BASE_MODEL:-}" && -n "${BASE_MODEL:-}" ]]; then
    export QWEN_BASE_MODEL="${BASE_MODEL}"
  fi
  if [[ -z "${QWEN_BASE_REVISION:-}" && -n "${BASE_MODEL_REVISION:-}" ]]; then
    export QWEN_BASE_REVISION="${BASE_MODEL_REVISION}"
  fi

  if [[ -z "${DATASET_ROOT:-}" ]]; then
    echo "[msmu-prepare] DATASET_ROOT is missing from ${env_file}" >&2
    return 2
  fi
  manual_output="${MANUAL_TEST_OUTPUT_ROOT:-}"
  if [[ -z "${manual_output}" ]]; then
    if [[ -z "${OUTPUT_ROOT:-}" ]]; then
      echo "[msmu-prepare] set MANUAL_TEST_OUTPUT_ROOT or OUTPUT_ROOT in ${env_file}" >&2
      return 2
    fi
    manual_output="${OUTPUT_ROOT%/}/manual-three-stage-v1"
  fi
  if [[ "${manual_output}" != /* ]]; then
    echo "[msmu-prepare] manual output root must be absolute: ${manual_output}" >&2
    return 2
  fi

  export OUTPUT_ROOT="${manual_output%/}"
  export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
  export no_proxy="${NO_PROXY}"
  mkdir -p "${OUTPUT_ROOT}/01_canary" "${OUTPUT_ROOT}/02_smoke8" "${OUTPUT_ROOT}/03_full987" || return 2
  cd "${REPO_ROOT}" || return 2
  set -o pipefail

  echo "[msmu-prepare] repository=${REPO_ROOT}"
  echo "[msmu-prepare] dataset=${DATASET_ROOT}"
  echo "[msmu-prepare] output=${OUTPUT_ROOT}"
  echo "[msmu-prepare] ready; choose one command from the current stage document"
}

_msmu_prepare_manual_test
_msmu_prepare_status=$?
unset -f _msmu_prepare_manual_test
return "${_msmu_prepare_status}"
