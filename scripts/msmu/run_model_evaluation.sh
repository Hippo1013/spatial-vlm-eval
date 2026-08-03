#!/usr/bin/env bash
# Run one registered MSMU model through stage-3 inference, targeted scoring,
# and the global publication-gated report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
STAGE3_SCRIPT="${SCRIPT_DIR}/run_manual_stage3.sh"
SCORING_SCRIPT="${SCRIPT_DIR}/score_pending_results.sh"
REPORT_SCRIPT="${SCRIPT_DIR}/build_results_report.sh"
MODEL_PROBE="${SCRIPT_DIR}/_probe_openai_models.py"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/msmu/run_model_evaluation.sh MODEL
  bash scripts/msmu/run_model_evaluation.sh MODEL --check
  bash scripts/msmu/run_model_evaluation.sh MODEL --status
  bash scripts/msmu/run_model_evaluation.sh --list

The default command runs one registered, stage-3-approved model through:
  full-987 inference + validator -> only that result's scoring -> global report

For a vLLM-backed model the tested-model service is started and stopped
automatically. The local judge is started only when the selected result still
needs scoring. Re-run the same command to resume its inference journal or judge
cache. Set MANUAL_DRY_RUN=1 to print the resolved workflow without using a GPU,
calling an API, starting a service, scoring, or writing a report.

Modes:
  --list    list model names accepted by the shared manual stage entry
  --check   validate registration, paths, commands, locks, and owned endpoints
  --status  show the selected result's scoring state and canonical paths

This entry does not create adapters for unknown models. A new model becomes
eligible after its adapter/profile is registered in run_manual_stage3.sh.
EOF
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  --list)
    "${STAGE3_SCRIPT}" --list | awk '$0 != "judge"'
    exit 0
    ;;
  "")
    usage >&2
    exit 2
    ;;
esac

model="$1"
operation="${2:-run}"
if (( $# > 2 )); then
  usage >&2
  exit 2
fi
case "${operation}" in
  run|--check|--status) ;;
  *)
    echo "[msmu-eval] unsupported operation: ${operation}" >&2
    usage >&2
    exit 2
    ;;
esac

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/prepare_manual_test.sh"

manual_dry_run="${MANUAL_DRY_RUN:-0}"
latent_python="${LATENT_PYTHON:-python}"
stage3_root="${OUTPUT_ROOT%/}/03_full987"
global_report="${stage3_root}/msmu-result.md"
inference_base_url="${MANUAL_INFERENCE_BASE_URL:-http://127.0.0.1:18081/v1}"
judge_base_url="${JUDGE_BASE_URL:-http://127.0.0.1:18080/v1}"
service_timeout="${BATCH_SERVICE_TIMEOUT_SECONDS:-1800}"
gpu_release_timeout="${BATCH_GPU_RELEASE_TIMEOUT_SECONDS:-600}"
process_stop_timeout="${BATCH_PROCESS_STOP_TIMEOUT_SECONDS:-90}"
poll_seconds="${BATCH_POLL_SECONDS:-15}"

fail() {
  echo "[msmu-eval] $*" >&2
  exit 2
}

blocked() {
  echo "[msmu-eval] $*" >&2
  exit 4
}

require_positive_integer() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value == 0 )); then
    fail "${name} must be a positive integer"
  fi
}

require_positive_integer BATCH_SERVICE_TIMEOUT_SECONDS "${service_timeout}"
require_positive_integer BATCH_GPU_RELEASE_TIMEOUT_SECONDS "${gpu_release_timeout}"
require_positive_integer BATCH_PROCESS_STOP_TIMEOUT_SECONDS "${process_stop_timeout}"
require_positive_integer BATCH_POLL_SECONDS "${poll_seconds}"

descriptor_output="$(bash "${STAGE3_SCRIPT}" "${model}" describe)"
descriptor_value() {
  local key="$1"
  awk -F '\t' -v key="${key}" '$1 == "descriptor" && $2 == key {print $3; exit}' \
    <<< "${descriptor_output}"
}

model_kind="$(descriptor_value model_kind)"
profile="$(descriptor_value profile)"
run_slug="$(descriptor_value run_slug)"
served_model_name="$(descriptor_value served_model_name)"
default_devices="$(descriptor_value default_devices)"
stage3_supported="$(descriptor_value stage3_supported)"
stage3_block_reason="$(descriptor_value stage3_block_reason)"

[[ -n "${model_kind}" ]] || fail "model descriptor did not provide model_kind for ${model}"
[[ -n "${profile}" ]] || fail "model descriptor did not provide profile for ${model}"
[[ -n "${run_slug}" ]] || fail "model descriptor did not provide run_slug for ${model}"
if [[ "${stage3_supported}" != "yes" ]]; then
  blocked "${model} is not approved for stage 3: ${stage3_block_reason:-unspecified reason}"
fi
if [[ "${model_kind}" == "vllm" && -z "${served_model_name}" ]]; then
  fail "vLLM descriptor did not provide served_model_name for ${model}"
fi

resolved_output="$(bash "${STAGE3_SCRIPT}" "${model}" resolve)"
resolved_value() {
  local key="$1"
  awk -F '\t' -v key="${key}" '$1 == "resolved" && $2 == key {print $3; exit}' \
    <<< "${resolved_output}"
}

predictions="$(resolved_value output)"
run_dir="$(resolved_value run_dir)"
validation_report="$(resolved_value validation_report)"
score_output_dir="$(resolved_value score_output_dir)"
[[ -n "${predictions}" ]] || fail "path resolver did not provide predictions output"
[[ -n "${run_dir}" ]] || fail "path resolver did not provide run directory"
[[ -n "${validation_report}" ]] || fail "path resolver did not provide validation report"
[[ -n "${score_output_dir}" ]] || fail "path resolver did not provide score directory"
case "${predictions}" in
  "${stage3_root}"/*/predictions.jsonl) ;;
  *) fail "resolved predictions escaped the stage-three result root: ${predictions}" ;;
esac

echo "[msmu-eval] model=${model} profile=${profile} kind=${model_kind}"
echo "[msmu-eval] predictions=${predictions}"
echo "[msmu-eval] score_dir=${score_output_dir}"
echo "[msmu-eval] global_report=${global_report}"

target_listing() {
  bash "${SCORING_SCRIPT}" --list \
    --results-root "${stage3_root}" \
    --predictions "${predictions}"
}

if [[ "${operation}" == "--status" ]]; then
  if [[ ! -f "${predictions}" ]]; then
    echo "missing\t${predictions}"
    exit 0
  fi
  target_listing
  exit 0
fi

endpoint_socket_open() {
  local base_url="$1"
  "${latent_python}" - "${base_url}" <<'PY'
import socket
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
host = parsed.hostname
if not host:
    raise SystemExit(2)
port = parsed.port or (443 if parsed.scheme == "https" else 80)
sock = socket.socket()
sock.settimeout(1.0)
try:
    status = sock.connect_ex((host, port))
finally:
    sock.close()
raise SystemExit(0 if status == 0 else 1)
PY
}

validate_owned_endpoint() {
  local label="$1" base_url="$2" expected_port="$3"
  "${latent_python}" - "${label}" "${base_url}" "${expected_port}" <<'PY'
import sys
from urllib.parse import urlparse

label, value, expected_port = sys.argv[1], sys.argv[2], int(sys.argv[3])
parsed = urlparse(value)
port = parsed.port or (443 if parsed.scheme == "https" else 80)
if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
    raise SystemExit(f"{label} must use a local HTTP endpoint, got {value}")
if port != expected_port or parsed.path.rstrip("/") != "/v1":
    raise SystemExit(f"{label} must be http://127.0.0.1:{expected_port}/v1, got {value}")
PY
}

check_commands_and_endpoints() {
  local command
  for command in bash flock nvidia-smi ps setsid; do
    command -v "${command}" >/dev/null 2>&1 || fail "required command is unavailable: ${command}"
  done
  if [[ "${latent_python}" == */* ]]; then
    [[ -x "${latent_python}" ]] || fail "configured LATENT_PYTHON is unavailable: ${latent_python}"
  else
    command -v "${latent_python}" >/dev/null 2>&1 || \
      fail "configured LATENT_PYTHON command is unavailable: ${latent_python}"
  fi
  [[ -f "${MODEL_PROBE}" ]] || fail "model readiness probe is unavailable: ${MODEL_PROBE}"
  if [[ "${model_kind}" == "vllm" ]]; then
    validate_owned_endpoint "inference endpoint" "${inference_base_url}" 18081
    if endpoint_socket_open "${inference_base_url}"; then
      blocked "inference endpoint is already occupied; existing service was left untouched: ${inference_base_url}"
    fi
  fi
  validate_owned_endpoint "judge endpoint" "${judge_base_url}" 18080
  if endpoint_socket_open "${judge_base_url}"; then
    blocked "judge endpoint is already occupied; existing service was left untouched: ${judge_base_url}"
  fi
  [[ -n "${JUDGE_MODEL:-}" ]] || fail "JUDGE_MODEL is missing from .env.server"
  [[ -n "${JUDGE_MODEL_NAME:-}" ]] || fail "JUDGE_MODEL_NAME is missing from .env.server"
}

if [[ "${manual_dry_run}" == "1" ]]; then
  echo "[msmu-eval] dry-run: bash ${STAGE3_SCRIPT} ${model} infer"
  if [[ "${model_kind}" == "vllm" ]]; then
    echo "[msmu-eval] dry-run: start and own ${STAGE3_SCRIPT} ${model} serve; wait for ${served_model_name}"
  fi
  echo "[msmu-eval] dry-run: start and own ${STAGE3_SCRIPT} judge serve; wait for ${JUDGE_MODEL_NAME:-msmu-judge}"
  echo "[msmu-eval] dry-run: bash ${SCORING_SCRIPT} --results-root ${stage3_root} --predictions ${predictions}"
  echo "[msmu-eval] dry-run: bash ${REPORT_SCRIPT} --results-root ${stage3_root} --output ${global_report}"
  echo "[msmu-eval] dry-run complete; no GPU/API/service/scorer/report action was taken"
  exit 0
fi

check_commands_and_endpoints

mkdir -p "${stage3_root}/_single_model_evaluation/logs" "${stage3_root}/_serial_inference"
exec 8>"${stage3_root}/_serial_inference/batch.lock"
if ! flock -n 8; then
  blocked "another stage-three inference batch holds ${stage3_root}/_serial_inference/batch.lock"
fi

if [[ "${operation}" == "--check" ]]; then
  echo "[msmu-eval] CHECK registration=valid"
  echo "[msmu-eval] CHECK stage3_supported=yes"
  echo "[msmu-eval] CHECK inference_lock=available"
  echo "[msmu-eval] CHECK owned_endpoints=available"
  echo "[msmu-eval] CHECK target_scoring=exact_predictions_path"
  echo "[msmu-eval] CHECK global_report=${global_report}"
  exit 0
fi

run_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
controller_log="${stage3_root}/_single_model_evaluation/logs/${run_timestamp}-${model}.log"
exec > >(tee -a "${controller_log}") 2>&1
echo "[msmu-eval] controller_log=${controller_log}"

active_service_pid=""
active_inference_pid=""
active_judge_pid=""
active_score_pid=""

group_alive() {
  local pid="$1"
  ps -eo pgid=,stat= | awk -v target="${pid}" '
    $1 == target && $2 !~ /^Z/ { found=1 }
    END { exit(found ? 0 : 1) }
  '
}

stop_owned_group() {
  local pid="$1" label="$2"
  [[ -n "${pid}" ]] || return 0
  if ! group_alive "${pid}"; then
    wait "${pid}" 2>/dev/null || true
    return 0
  fi
  echo "[msmu-eval] stopping owned ${label} process group ${pid} with TERM"
  kill -TERM -- "-${pid}" 2>/dev/null || true
  local deadline=$(( $(date +%s) + process_stop_timeout ))
  while group_alive "${pid}" && (( $(date +%s) < deadline )); do
    sleep "${poll_seconds}"
  done
  if group_alive "${pid}"; then
    echo "[msmu-eval] owned ${label} process group ${pid} ignored TERM; sending KILL" >&2
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup_owned_processes() {
  stop_owned_group "${active_score_pid}" "scorer"
  active_score_pid=""
  stop_owned_group "${active_judge_pid}" "judge service"
  active_judge_pid=""
  stop_owned_group "${active_inference_pid}" "inference"
  active_inference_pid=""
  stop_owned_group "${active_service_pid}" "tested-model service"
  active_service_pid=""
}

handle_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  cleanup_owned_processes
  if (( status != 0 )); then
    echo "[msmu-eval] stopped status=${status}; rerun the same command to resume" >&2
  fi
  exit "${status}"
}

handle_signal() {
  local status="$1" signal_name="$2"
  trap - EXIT HUP INT TERM
  cleanup_owned_processes
  echo "[msmu-eval] received ${signal_name}; rerun the same command to resume" >&2
  exit "${status}"
}

trap handle_exit EXIT
trap 'handle_signal 129 HUP' HUP
trap 'handle_signal 130 INT' INT
trap 'handle_signal 143 TERM' TERM

wait_for_served_model() {
  local base_url="$1" expected_model="$2" process_pid="$3" label="$4"
  local deadline=$(( $(date +%s) + service_timeout )) now last_report=0
  while true; do
    if ! group_alive "${process_pid}"; then
      wait "${process_pid}" 2>/dev/null || true
      echo "[msmu-eval] ${label} exited before readiness" >&2
      return 1
    fi
    if "${latent_python}" "${MODEL_PROBE}" \
      --base-url "${base_url}" \
      --expected-model "${expected_model}" \
      --timeout 5; then
      echo "[msmu-eval] ${label} ready endpoint=${base_url} model=${expected_model}"
      return 0
    fi
    now="$(date +%s)"
    if (( now >= deadline )); then
      echo "[msmu-eval] timed out waiting for ${label}: ${expected_model}" >&2
      return 124
    fi
    if (( now - last_report >= 60 )); then
      echo "[msmu-eval] waiting for ${label}: ${expected_model}"
      last_report="${now}"
    fi
    sleep "${poll_seconds}"
  done
}

run_owned_foreground() {
  local pid_variable="$1" label="$2"
  shift 2
  setsid "$@" &
  local pid=$!
  printf -v "${pid_variable}" '%s' "${pid}"
  set +e
  wait "${pid}"
  local status=$?
  set -e
  stop_owned_group "${pid}" "${label}"
  printf -v "${pid_variable}" '%s' ""
  return "${status}"
}

gpu_compute_pids() {
  local devices="$1" gpu_id
  local -a gpu_ids
  IFS=',' read -r -a gpu_ids <<< "${devices}"
  for gpu_id in "${gpu_ids[@]}"; do
    nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | awk -v gpu="${gpu_id}" '/^[[:space:]]*[0-9]+[[:space:]]*$/ {
          gsub(/[[:space:]]/, ""); print gpu ":" $0
        }'
  done
}

wait_for_gpu_release() {
  local devices="$1"
  [[ -n "${devices}" ]] || return 0
  local deadline=$(( $(date +%s) + gpu_release_timeout )) pids now last_report=0
  while true; do
    pids="$(gpu_compute_pids "${devices}")"
    if [[ -z "${pids}" ]]; then
      echo "[msmu-eval] GPUs ${devices} released"
      return 0
    fi
    now="$(date +%s)"
    if (( now >= deadline )); then
      echo "[msmu-eval] timed out waiting for GPUs ${devices}; processes left untouched: ${pids//$'\n'/,}" >&2
      return 4
    fi
    if (( now - last_report >= 60 )); then
      echo "[msmu-eval] waiting for GPUs ${devices}; existing processes left untouched: ${pids//$'\n'/,}"
      last_report="${now}"
    fi
    sleep "${poll_seconds}"
  done
}

inference_artifacts_complete() {
  [[ -f "${predictions}" && -f "${predictions}.metadata.json" ]] || return 1
  if ! PREDICTIONS="${predictions}" DATASET_ROOT="${DATASET_ROOT}" \
    REPORT="${validation_report}" ALLOW_SUBSET=0 \
      bash "${SCRIPT_DIR}/validate_predictions.sh" >/dev/null; then
    return 1
  fi
  "${latent_python}" - "${predictions}" "${profile}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

predictions = Path(sys.argv[1]).resolve()
expected_profile = sys.argv[2]
metadata_path = Path(str(predictions) + ".metadata.json")
try:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit(1)
dataset = metadata.get("dataset") if isinstance(metadata, dict) else None
model = metadata.get("model") if isinstance(metadata, dict) else None
checks = (
    metadata.get("publishable_inference") is True,
    metadata.get("num_predictions") == 987,
    Path(str(metadata.get("output", ""))).resolve() == predictions,
    metadata.get("output_sha256") == hashlib.sha256(predictions.read_bytes()).hexdigest(),
    isinstance(dataset, dict),
    isinstance(model, dict),
)
if not all(checks):
    raise SystemExit(1)
assert isinstance(dataset, dict) and isinstance(model, dict)
dataset_checks = (
    dataset.get("split") == "test",
    dataset.get("official_test_size") == 987,
    dataset.get("num_targets") == 987,
    dataset.get("is_subset") is False,
    dataset.get("target_indices") == list(range(987)),
    model.get("profile") == expected_profile,
)
raise SystemExit(0 if all(dataset_checks) else 1)
PY
}

selected_score_state() {
  [[ -f "${predictions}" ]] || {
    printf 'missing'
    return 0
  }
  target_listing | awk -F '\t' '
    $1 == "new" || $1 == "resume" || $1 == "retry" ||
    $1 == "complete" || $1 == "excluded_protocol" {print $1; exit}
  '
}

score_state="$(selected_score_state)"
if [[ "${score_state}" == "complete" ]]; then
  echo "[msmu-eval] selected result already passed all publication gates; skipping inference and judge"
else
  if inference_artifacts_complete; then
    echo "[msmu-eval] full-987 inference artifacts already validate; skipping model inference"
  else
    if [[ -f "${predictions}" ]]; then
      fail "existing inference artifacts are not safely reusable: ${predictions}"
    fi
    if [[ "${model_kind}" == "vllm" ]]; then
      echo "[msmu-eval] starting tested-model service for ${model}"
      setsid bash "${STAGE3_SCRIPT}" "${model}" serve &
      active_service_pid=$!
      wait_for_served_model "${inference_base_url}" "${served_model_name}" \
        "${active_service_pid}" "tested-model service"
    fi

    echo "[msmu-eval] starting full-987 inference for ${model}"
    run_owned_foreground active_inference_pid "${model} inference" \
      bash "${STAGE3_SCRIPT}" "${model}" infer

    if [[ -n "${active_service_pid}" ]]; then
      stop_owned_group "${active_service_pid}" "${model} service"
      active_service_pid=""
    fi
    devices="${MANUAL_CUDA_VISIBLE_DEVICES:-${default_devices}}"
    if [[ "${model_kind}" != "api" ]]; then
      wait_for_gpu_release "${devices}"
    fi
    if ! inference_artifacts_complete; then
      fail "inference exited successfully but full-987 artifacts are incomplete: ${predictions}"
    fi
  fi

  score_state="$(selected_score_state)"
  if [[ "${score_state}" != "complete" ]]; then
    if endpoint_socket_open "${judge_base_url}"; then
      blocked "judge endpoint became occupied before owned judge startup: ${judge_base_url}"
    fi
    echo "[msmu-eval] starting local judge"
    setsid bash "${STAGE3_SCRIPT}" judge serve &
    active_judge_pid=$!
    wait_for_served_model "${judge_base_url}" "${JUDGE_MODEL_NAME}" \
      "${active_judge_pid}" "judge service"

    echo "[msmu-eval] scoring only selected predictions=${predictions}"
    run_owned_foreground active_score_pid "targeted scoring" \
      bash "${SCORING_SCRIPT}" \
        --results-root "${stage3_root}" \
        --predictions "${predictions}"

    stop_owned_group "${active_judge_pid}" "judge service"
    active_judge_pid=""
  fi
fi

if [[ "$(selected_score_state)" != "complete" ]]; then
  fail "selected result did not pass canonical publication gates: ${predictions}"
fi

echo "[msmu-eval] rebuilding global publication-gated report"
bash "${REPORT_SCRIPT}" \
  --results-root "${stage3_root}" \
  --output "${global_report}"

echo "[msmu-eval] COMPLETE model=${model}"
echo "[msmu-eval] predictions=${predictions}"
echo "[msmu-eval] summary=${score_output_dir}/summary.json"
echo "[msmu-eval] global_report=${global_report}"
