#!/usr/bin/env bash
# Run CV-Bench full inference serially while owning each required vLLM service.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cv_bench/run_full_serial.sh --without-internvl78 [--skip-completed]
  bash scripts/cv_bench/run_full_serial.sh --without-internvl78 --dry-run
  bash scripts/cv_bench/run_full_serial.sh --list

Execution owns and rotates vLLM services for general-open profiles, then runs
API and specialized profiles in registry order. It never scores results.
EOF
}

exclude_internvl78=0
skip_completed=0
dry_run=0
list_only=0
while (( $# > 0 )); do
  case "$1" in
    --without-internvl78) exclude_internvl78=1; shift ;;
    --skip-completed) skip_completed=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --list) list_only=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[cv-bench-full-serial] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${list_only}" != "1" && "${exclude_internvl78}" != "1" ]]; then
  echo "[cv-bench-full-serial] execution requires --without-internvl78" >&2
  exit 2
fi

profile_rows="$("${CVBENCH_PYTHON}" - "${exclude_internvl78}" <<'PY'
import sys
from spatial_vlm_eval.benchmarks.cv_bench.profiles import PROFILE_SEQUENCE, get_profile

exclude = sys.argv[1] == "1"
for key in PROFILE_SEQUENCE:
    if exclude and key == "internvl3_78b":
        continue
    profile = get_profile(key)
    print("\t".join([
        profile.key,
        profile.group,
        str(profile.default_tensor_parallel_size),
        profile.served_model_name or "-",
    ]))
PY
)"

if [[ "${list_only}" == "1" ]]; then
  printf '%s\n' "${profile_rows}"
  exit 0
fi
if [[ -z "${CVBENCH_OUTPUT_ROOT:-}" ]]; then
  echo "[cv-bench-full-serial] set CVBENCH_OUTPUT_ROOT" >&2
  exit 2
fi

if [[ "${dry_run}" == "1" ]]; then
  while IFS=$'\t' read -r profile group tp served; do
    echo "[cv-bench-full-serial] dry-run profile=${profile} group=${group} tp=${tp} served=${served}"
  done <<<"${profile_rows}"
  exit 0
fi

service_timeout="${CVBENCH_SERIAL_SERVICE_TIMEOUT_SECONDS:-1800}"
stop_timeout="${CVBENCH_SERIAL_STOP_TIMEOUT_SECONDS:-120}"
gpu_release_timeout="${CVBENCH_SERIAL_GPU_RELEASE_TIMEOUT_SECONDS:-600}"
poll_seconds="${CVBENCH_SERIAL_POLL_SECONDS:-10}"
for value in "${service_timeout}" "${stop_timeout}" "${gpu_release_timeout}" "${poll_seconds}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[cv-bench-full-serial] timeout and poll values must be positive integers" >&2
    exit 2
  fi
done

control_root="${CVBENCH_OUTPUT_ROOT%/}/_serial_full"
logs_root="${control_root}/logs"
mkdir -p "${logs_root}"
exec 8>"${control_root}/lock"
if ! flock -n 8; then
  echo "[cv-bench-full-serial] another run holds ${control_root}/lock" >&2
  exit 4
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
status_file="${control_root}/status.tsv"
controller_log="${logs_root}/${run_id}.controller.log"
exec > >(tee -a "${controller_log}") 2>&1

active_inference_pid=""
active_service_pids=()

group_alive() {
  [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null
}

stop_owned_group() {
  local pid="$1" label="$2" deadline
  [[ -n "${pid}" ]] || return 0
  if group_alive "${pid}"; then
    echo "[cv-bench-full-serial] stopping owned ${label} pid=${pid}"
    kill -TERM -- "-${pid}" 2>/dev/null || true
  fi
  deadline=$(( $(date +%s) + stop_timeout ))
  while group_alive "${pid}" && (( $(date +%s) < deadline )); do
    sleep "${poll_seconds}"
  done
  if group_alive "${pid}"; then
    echo "[cv-bench-full-serial] owned ${label} ignored TERM; sending KILL" >&2
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup_owned() {
  local pid
  stop_owned_group "${active_inference_pid}" "inference"
  active_inference_pid=""
  for pid in "${active_service_pids[@]:-}"; do
    stop_owned_group "${pid}" "vLLM service"
  done
  active_service_pids=()
}

handle_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  cleanup_owned
  exit "${status}"
}
trap handle_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

endpoint_open() {
  "${CVBENCH_PYTHON}" - "$1" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.settimeout(1)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

wait_for_model() {
  local port="$1" expected="$2" pid="$3" deadline
  deadline=$(( $(date +%s) + service_timeout ))
  while true; do
    if ! group_alive "${pid}"; then
      echo "[cv-bench-full-serial] vLLM exited before readiness: ${expected}" >&2
      return 1
    fi
    if "${CVBENCH_PYTHON}" "${SCRIPT_DIR}/../msmu/_probe_openai_models.py" \
      --base-url "http://127.0.0.1:${port}/v1" --expected-model "${expected}" --timeout 5; then
      echo "[cv-bench-full-serial] ready model=${expected} port=${port}"
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      echo "[cv-bench-full-serial] readiness timeout model=${expected} port=${port}" >&2
      return 124
    fi
    sleep "${poll_seconds}"
  done
}

gpu_compute_pids() {
  local gpu
  for gpu in 0 1; do
    nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | awk -v gpu="${gpu}" '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); print gpu ":" $0}'
  done
}

wait_for_gpu_release() {
  local deadline pids
  deadline=$(( $(date +%s) + gpu_release_timeout ))
  while true; do
    pids="$(gpu_compute_pids)"
    [[ -z "${pids}" ]] && return 0
    if (( $(date +%s) >= deadline )); then
      echo "[cv-bench-full-serial] GPU release timeout; processes left untouched: ${pids//$'\n'/,}" >&2
      return 4
    fi
    sleep "${poll_seconds}"
  done
}

profile_env_token() {
  tr '[:lower:]-' '[:upper:]_' <<<"$1" | tr -d '\n'
}

start_vllm() {
  local profile="$1" tp="$2" served="$3" port gpu_ids log pid token
  active_service_pids=()
  token="$(profile_env_token "${profile}")"
  if [[ "${tp}" == "1" ]]; then
    for port in 18101 18102; do
      if endpoint_open "${port}"; then
        echo "[cv-bench-full-serial] port ${port} occupied; existing process left untouched" >&2
        return 4
      fi
    done
    for gpu_ids in 0 1; do
      port=$(( 18101 + gpu_ids ))
      log="${logs_root}/${run_id}.${profile}.gpu${gpu_ids}.vllm.log"
      setsid bash "${SCRIPT_DIR}/serve_vllm_profile.sh" \
        --model "${profile}" --gpu-ids "${gpu_ids}" --port "${port}" >"${log}" 2>&1 &
      active_service_pids+=("$!")
    done
    wait_for_model 18101 "${served}" "${active_service_pids[0]}"
    wait_for_model 18102 "${served}" "${active_service_pids[1]}"
    export "CVBENCH_${token}_BASE_URLS=http://127.0.0.1:18101/v1,http://127.0.0.1:18102/v1"
  else
    port=18101
    if endpoint_open "${port}"; then
      echo "[cv-bench-full-serial] port ${port} occupied; existing process left untouched" >&2
      return 4
    fi
    log="${logs_root}/${run_id}.${profile}.tp${tp}.vllm.log"
    setsid bash "${SCRIPT_DIR}/serve_vllm_profile.sh" \
      --model "${profile}" --gpu-ids "0,1" --port "${port}" >"${log}" 2>&1 &
    pid=$!
    active_service_pids+=("${pid}")
    wait_for_model "${port}" "${served}" "${pid}"
    export "CVBENCH_${token}_BASE_URLS=http://127.0.0.1:18101/v1"
  fi
}

run_profile() {
  local profile="$1" log status
  log="${logs_root}/${run_id}.${profile}.full.log"
  setsid "${CVBENCH_PYTHON}" -m spatial_vlm_eval.benchmarks.cv_bench.inference \
    --stage full --model "${profile}" >"${log}" 2>&1 &
  active_inference_pid=$!
  set +e
  wait "${active_inference_pid}"
  status=$?
  set -e
  active_inference_pid=""
  if (( status != 0 )); then
    echo "[cv-bench-full-serial] profile failed status=${status}: ${profile}; log=${log}" >&2
    tail -80 "${log}" >&2 || true
    return "${status}"
  fi
  echo "[cv-bench-full-serial] profile passed: ${profile}"
}

profile_has_valid_full_predictions() {
  local profile="$1"
  "${CVBENCH_PYTHON}" - "${profile}" "${CVBENCH_OUTPUT_ROOT}" "${CVBENCH_DATASET_ROOT}" <<'PY'
import sys
from pathlib import Path

from spatial_vlm_eval.benchmarks.cv_bench.inference import track_directory
from spatial_vlm_eval.benchmarks.cv_bench.prediction_validation import validate_predictions
from spatial_vlm_eval.benchmarks.cv_bench.profiles import get_profile

profile = get_profile(sys.argv[1])
prediction_path = track_directory(sys.argv[2], profile) / "predictions.jsonl"
if not prediction_path.is_file():
    raise SystemExit(1)
try:
    _, report = validate_predictions(
        prediction_path,
        Path(sys.argv[3]).resolve(),
        allow_subset=False,
        verify_files=True,
    )
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if report.get("passed") is True else 1)
PY
}

printf 'run_id\tprofile\tstate\ttimestamp\n' >"${status_file}"
echo "[cv-bench-full-serial] run_id=${run_id} controller_log=${controller_log}"
while IFS=$'\t' read -r profile group tp served; do
  [[ -n "${profile}" ]] || continue
  if [[ "${skip_completed}" == "1" ]] && profile_has_valid_full_predictions "${profile}"; then
    printf '%s\t%s\tSKIP_COMPLETE\t%s\n' \
      "${run_id}" "${profile}" "$(date -u +%FT%TZ)" >>"${status_file}"
    echo "[cv-bench-full-serial] SKIP_COMPLETE profile=${profile} validator=passed"
    continue
  fi
  printf '%s\t%s\tSTART\t%s\n' "${run_id}" "${profile}" "$(date -u +%FT%TZ)" >>"${status_file}"
  echo "[cv-bench-full-serial] START profile=${profile}"
  if [[ "${group}" == "general_open" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1
    wait_for_gpu_release
    start_vllm "${profile}" "${tp}" "${served}"
  else
    export CUDA_VISIBLE_DEVICES=0
  fi
  if ! run_profile "${profile}"; then
    printf '%s\t%s\tFAIL\t%s\n' "${run_id}" "${profile}" "$(date -u +%FT%TZ)" >>"${status_file}"
    exit 1
  fi
  if [[ "${group}" == "general_open" ]]; then
    cleanup_owned
    wait_for_gpu_release
  fi
  printf '%s\t%s\tPASS\t%s\n' "${run_id}" "${profile}" "$(date -u +%FT%TZ)" >>"${status_file}"
done <<<"${profile_rows}"
echo "[cv-bench-full-serial] COMPLETE"
