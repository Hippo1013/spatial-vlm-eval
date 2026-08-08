#!/usr/bin/env bash
# Securely store the PackyAPI key in the shared untracked server env file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
ENV_FILE="${SPBENCH_SI_ENV_FILE:-${REPO_ROOT}/.env.server}"

printf 'PackyAPI Gemini-slb API key (input hidden): ' >&2
if ! IFS= read -r -s api_key; then
  printf '\n[spbench-si] failed to read API key\n' >&2
  exit 1
fi
printf '\n' >&2

if [[ -z "${api_key}" ]]; then
  echo "[spbench-si] API key must not be empty" >&2
  exit 1
fi

umask 077
env_dir="$(dirname "${ENV_FILE}")"
if [[ ! -d "${env_dir}" ]]; then
  echo "[spbench-si] env directory does not exist: ${env_dir}" >&2
  exit 1
fi

tmp_file="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
cleanup() {
  if [[ -n "${tmp_file:-}" && -e "${tmp_file}" ]]; then
    rm -f -- "${tmp_file}"
  fi
  unset api_key
}
trap cleanup EXIT

if [[ -f "${ENV_FILE}" ]]; then
  awk '!/^[[:space:]]*(export[[:space:]]+)?PACKYAPI_API_KEY=/' \
    "${ENV_FILE}" >"${tmp_file}"
fi
printf '\nPACKYAPI_API_KEY=%q\n' "${api_key}" >>"${tmp_file}"
chmod 600 "${tmp_file}"
mv -f -- "${tmp_file}" "${ENV_FILE}"
tmp_file=""

if ! bash -c 'set -a; source "$1"; set +a; [[ -n "${PACKYAPI_API_KEY:-}" ]]' \
  bash "${ENV_FILE}"; then
  echo "[spbench-si] key was written but could not be loaded from ${ENV_FILE}" >&2
  exit 1
fi

echo "[spbench-si] PackyAPI key saved securely in ${ENV_FILE} (mode 600)"
