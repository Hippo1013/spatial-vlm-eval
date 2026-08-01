#!/usr/bin/env bash
# Run one locked MSMU stage-3 inference plan sequentially.
#
# This script deliberately performs inference + full validation only. Scoring is
# a separate phase because the text-only judge should be loaded after every
# tested model has released its GPU allocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
STAGE3_SCRIPT="${SCRIPT_DIR}/run_manual_stage3.sh"

legacy_models=(
  llava_next_mistral_7b
  llava_next_yi_34b
  internvl3_8b
  internvl3_38b
  qwen25_vl_base
  qwen25_vl_32b
  ssr
  ssr_native
  spatialrgpt
  3dthinker
  3dthinker_native
  spatialbot
  spatialbot_native
)

qwen3_models=(
  qwen3_vl_2b
  qwen3_vl_4b
  qwen3_vl_8b
  qwen3_vl_32b
)

plan="legacy13"
operation="run"
operation_selected=0
while (( $# > 0 )); do
  case "$1" in
    --qwen3)
      if [[ "${plan}" != "legacy13" ]]; then
        echo "[msmu-batch] only one inference plan may be selected" >&2
        exit 2
      fi
      plan="qwen3"
      ;;
    --help|-h)
      if (( operation_selected == 1 )); then
        echo "[msmu-batch] choose only one of --help, --list, --check, or --status" >&2
        exit 2
      fi
      operation="help"
      operation_selected=1
      ;;
    --list)
      if (( operation_selected == 1 )); then
        echo "[msmu-batch] choose only one of --help, --list, --check, or --status" >&2
        exit 2
      fi
      operation="list"
      operation_selected=1
      ;;
    --status)
      if (( operation_selected == 1 )); then
        echo "[msmu-batch] choose only one of --help, --list, --check, or --status" >&2
        exit 2
      fi
      operation="status"
      operation_selected=1
      ;;
    --check)
      if (( operation_selected == 1 )); then
        echo "[msmu-batch] choose only one of --help, --list, --check, or --status" >&2
        exit 2
      fi
      operation="check"
      operation_selected=1
      ;;
    *)
      echo "[msmu-batch] unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

case "${plan}" in
  legacy13) models=("${legacy_models[@]}") ;;
  qwen3) models=("${qwen3_models[@]}") ;;
  *) echo "[msmu-batch] internal error: unsupported plan ${plan}" >&2; exit 2 ;;
esac

print_plan() {
  printf '%s\n' "${models[@]}"
  if [[ "${plan}" == "legacy13" ]]; then
    cat <<'EOF'
excluded	gpt5	API model
excluded	gemini31pro	API model
excluded	qwen25_vl_72b	70B+ model
excluded	internvl3_78b	separate four-GPU manual supplement
excluded	qwen25_vl_peft	not part of the accepted test plan
EOF
  fi
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/msmu/run_stage3_serial_inference.sh
  bash scripts/msmu/run_stage3_serial_inference.sh --qwen3
  bash scripts/msmu/run_stage3_serial_inference.sh --list
  bash scripts/msmu/run_stage3_serial_inference.sh --check
  bash scripts/msmu/run_stage3_serial_inference.sh --status

The default plan retains the original 13 local tracks. Add --qwen3 to select only
Qwen3-VL 2B, 4B, 8B, and 32B. The selector can be combined with --list, --check,
or --status. Neither plan calls the judge or an API model.

Safety and recovery settings:
  MANUAL_DRY_RUN=1                  print the plan without using GPUs
  BATCH_MODEL_ATTEMPTS=2            attempts per model; journal resumes samples
  BATCH_STALL_TIMEOUT_SECONDS=3600  stop an owned inference process group after
                                    this many seconds without journal/log activity
  BATCH_SERVICE_TIMEOUT_SECONDS=1800
                                    maximum vLLM readiness wait
  BATCH_GPU_WAIT_TIMEOUT_SECONDS=1800
                                    maximum wait for required GPUs to become idle
  BATCH_GPU_RELEASE_TIMEOUT_SECONDS=600
                                    maximum wait after an owned model exits
  BATCH_PROCESS_STOP_TIMEOUT_SECONDS=90
                                    TERM grace before KILL of an owned group
  BATCH_POLL_SECONDS=15             watchdog/readiness polling interval
  BATCH_RETRY_DELAY_SECONDS=30      delay before resuming a failed model
  BATCH_CONTINUE_ON_ERROR=0         stop at first model that exhausts its attempts
  BATCH_SKIP_COMPLETED=1            skip same-commit completion markers

Run this inside tmux. If interrupted, run the exact same command again: completed
models are skipped and the current model resumes from its fsync-backed journal.
EOF
}

case "${operation}" in
  help)
    usage
    exit 0
    ;;
  list)
    print_plan
    exit 0
    ;;
  status)
    status_only=1
    preflight_only=0
    ;;
  check)
    status_only=0
    preflight_only=1
    ;;
  run)
    status_only=0
    preflight_only=0
    ;;
  *) echo "[msmu-batch] internal error: unsupported operation ${operation}" >&2; exit 2 ;;
esac

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/prepare_manual_test.sh"

manual_dry_run="${MANUAL_DRY_RUN:-0}"
model_attempts="${BATCH_MODEL_ATTEMPTS:-2}"
stall_timeout="${BATCH_STALL_TIMEOUT_SECONDS:-3600}"
service_timeout="${BATCH_SERVICE_TIMEOUT_SECONDS:-1800}"
gpu_wait_timeout="${BATCH_GPU_WAIT_TIMEOUT_SECONDS:-1800}"
gpu_release_timeout="${BATCH_GPU_RELEASE_TIMEOUT_SECONDS:-600}"
process_stop_timeout="${BATCH_PROCESS_STOP_TIMEOUT_SECONDS:-90}"
poll_seconds="${BATCH_POLL_SECONDS:-15}"
retry_delay="${BATCH_RETRY_DELAY_SECONDS:-30}"
continue_on_error="${BATCH_CONTINUE_ON_ERROR:-0}"
skip_completed="${BATCH_SKIP_COMPLETED:-1}"

require_nonnegative_integer() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "[msmu-batch] ${name} must be a non-negative integer" >&2
    exit 2
  fi
}

require_positive_integer() {
  local name="$1" value="$2"
  require_nonnegative_integer "${name}" "${value}"
  if (( value == 0 )); then
    echo "[msmu-batch] ${name} must be greater than zero" >&2
    exit 2
  fi
}

require_positive_integer BATCH_MODEL_ATTEMPTS "${model_attempts}"
require_positive_integer BATCH_STALL_TIMEOUT_SECONDS "${stall_timeout}"
require_positive_integer BATCH_SERVICE_TIMEOUT_SECONDS "${service_timeout}"
require_positive_integer BATCH_GPU_WAIT_TIMEOUT_SECONDS "${gpu_wait_timeout}"
require_positive_integer BATCH_GPU_RELEASE_TIMEOUT_SECONDS "${gpu_release_timeout}"
require_positive_integer BATCH_PROCESS_STOP_TIMEOUT_SECONDS "${process_stop_timeout}"
require_positive_integer BATCH_POLL_SECONDS "${poll_seconds}"
require_nonnegative_integer BATCH_RETRY_DELAY_SECONDS "${retry_delay}"
if [[ "${continue_on_error}" != "0" && "${continue_on_error}" != "1" ]]; then
  echo "[msmu-batch] BATCH_CONTINUE_ON_ERROR must be 0 or 1" >&2
  exit 2
fi
if [[ "${skip_completed}" != "0" && "${skip_completed}" != "1" ]]; then
  echo "[msmu-batch] BATCH_SKIP_COMPLETED must be 0 or 1" >&2
  exit 2
fi

serial_root="${OUTPUT_ROOT}/03_full987/_serial_inference"
if [[ "${plan}" == "legacy13" ]]; then
  batch_root="${serial_root}"
else
  batch_root="${serial_root}/qwen3"
fi
completion_dir="${batch_root}/completed"
log_dir="${batch_root}/logs"
active_file="${serial_root}/active_process.env"
plan_file="${batch_root}/plan.env"
status_file="${batch_root}/status.tsv"
mkdir -p "${completion_dir}" "${log_dir}"

repository_sha="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
model_csv="$(IFS=,; printf '%s' "${models[*]}")"

if [[ -f "${plan_file}" ]]; then
  planned_sha="$(awk -F= '$1 == "repository_sha" {print substr($0, index($0, "=") + 1)}' "${plan_file}")"
  planned_models="$(awk -F= '$1 == "models" {print substr($0, index($0, "=") + 1)}' "${plan_file}")"
  if [[ "${planned_sha}" != "${repository_sha}" || "${planned_models}" != "${model_csv}" ]]; then
    echo "[msmu-batch] existing batch state belongs to another commit or model plan:" >&2
    echo "[msmu-batch] state=${plan_file}" >&2
    echo "[msmu-batch] planned_sha=${planned_sha} current_sha=${repository_sha}" >&2
    echo "[msmu-batch] use a new MANUAL_TEST_OUTPUT_ROOT for a changed protocol/code plan" >&2
    exit 2
  fi
else
  plan_temporary="${plan_file}.tmp.$$"
  {
    printf 'repository_sha=%s\n' "${repository_sha}"
    printf 'plan=%s\n' "${plan}"
    printf 'models=%s\n' "${model_csv}"
    if [[ "${plan}" == "legacy13" ]]; then
      printf 'excluded=%s\n' "gpt5,gemini31pro,qwen25_vl_72b,internvl3_78b,qwen25_vl_peft"
    fi
  } > "${plan_temporary}"
  mv "${plan_temporary}" "${plan_file}"
fi

marker_matches_plan() {
  local marker="$1"
  [[ -f "${marker}" ]] || return 1
  grep -Fxq "repository_sha=${repository_sha}" "${marker}"
}

if [[ "${status_only}" == "1" ]]; then
  for model in "${models[@]}"; do
    marker="${completion_dir}/${model}.complete"
    if marker_matches_plan "${marker}"; then
      printf 'complete\t%s\n' "${model}"
    else
      printf 'pending\t%s\n' "${model}"
    fi
  done
  printf 'state\t%s\n' "${batch_root}"
  exit 0
fi

run_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
batch_log="${log_dir}/${run_timestamp}.log"
exec > >(tee -a "${batch_log}") 2>&1

echo "[msmu-batch] repository_sha=${repository_sha}"
echo "[msmu-batch] plan=${plan}"
echo "[msmu-batch] state=${batch_root}"
echo "[msmu-batch] log=${batch_log}"
echo "[msmu-batch] inference tracks=${#models[@]}; scoring is deliberately disabled"
print_plan

configuration_errors=0

check_required_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[msmu-batch] missing required .env.server value: ${name}" >&2
    configuration_errors=$(( configuration_errors + 1 ))
  fi
}

check_required_path() {
  local name="$1" value="${!1:-}"
  if [[ -z "${value}" ]]; then
    echo "[msmu-batch] missing required .env.server path: ${name}" >&2
    configuration_errors=$(( configuration_errors + 1 ))
  elif [[ ! -e "${value}" ]]; then
    echo "[msmu-batch] configured path does not exist: ${name}=${value}" >&2
    configuration_errors=$(( configuration_errors + 1 ))
  fi
}

check_required_executable() {
  local name="$1" value="${!1:-}"
  if [[ -z "${value}" ]]; then
    echo "[msmu-batch] missing required .env.server executable: ${name}" >&2
    configuration_errors=$(( configuration_errors + 1 ))
  elif [[ "${value}" == */* ]]; then
    if [[ ! -x "${value}" ]]; then
      echo "[msmu-batch] configured executable is unavailable: ${name}=${value}" >&2
      configuration_errors=$(( configuration_errors + 1 ))
    fi
  elif ! command -v "${value}" >/dev/null 2>&1; then
    echo "[msmu-batch] configured command is unavailable: ${name}=${value}" >&2
    configuration_errors=$(( configuration_errors + 1 ))
  fi
}

if [[ "${manual_dry_run}" != "1" ]]; then
  for command in git nvidia-smi setsid curl flock ps; do
    if ! command -v "${command}" >/dev/null 2>&1; then
      echo "[msmu-batch] required command is unavailable: ${command}" >&2
      configuration_errors=$(( configuration_errors + 1 ))
    fi
  done
  required_paths=(DATASET_ROOT)
  required_revisions=()
  required_executables=(LATENT_PYTHON)
  if [[ "${plan}" == "legacy13" ]]; then
    required_paths+=(
      LLAVA_MISTRAL_7B_MODEL LLAVA_YI_34B_MODEL
      INTERNVL3_8B_MODEL INTERNVL3_38B_MODEL
      QWEN_BASE_MODEL QWEN_32B_MODEL
      SSR_UPSTREAM_ROOT SSR_DEPTHPRO_ROOT BASE_MODEL SSR_VLM SSR_MIDI
      CLIP_MODEL SIGLIP_MODEL MAMBA_MODEL MIDI_LLM_MODEL DEPTHPRO_CHECKPOINT
      SPATIALRGPT_UPSTREAM_ROOT SPATIALRGPT_MODEL
      THREEDTHINKER_UPSTREAM_ROOT THREEDTHINKER_MODEL
      SPATIALBOT_UPSTREAM_ROOT SPATIALBOT_MODEL
      ZOEDEPTH_ROOT ZOEDEPTH_CHECKPOINT
    )
    required_revisions+=(QWEN_BASE_REVISION QWEN_32B_REVISION)
    required_executables+=(
      VLLM_PYTHON VLLM SSR_PYTHON SPATIALRGPT_PYTHON
      THREEDTHINKER_PYTHON SPATIALBOT_PYTHON
    )
  else
    required_paths+=(
      QWEN3_2B_MODEL QWEN3_4B_MODEL QWEN3_8B_MODEL QWEN3_32B_MODEL
    )
    required_revisions+=(
      QWEN3_2B_REVISION QWEN3_4B_REVISION QWEN3_8B_REVISION QWEN3_32B_REVISION
    )
  fi
  for name in "${required_paths[@]}"; do
    check_required_path "${name}"
  done
  for name in "${required_revisions[@]}"; do
    check_required_value "${name}"
  done
  for name in "${required_executables[@]}"; do
    check_required_executable "${name}"
  done
  if (( configuration_errors > 0 )); then
    echo "[msmu-batch] configuration preflight failed with ${configuration_errors} error(s); no model was started" >&2
    exit 2
  fi
  exec 9>"${serial_root}/batch.lock"
  if ! flock -n 9; then
    echo "[msmu-batch] another serial inference batch already holds ${serial_root}/batch.lock" >&2
    exit 4
  fi
fi

active_service_pid=""
active_inference_pid=""
active_model=""
child_env_cleanup=(
  -u INDICES
  -u JOURNAL
  -u LIMIT
  -u MANUAL_CUDA_VISIBLE_DEVICES
  -u MANUAL_INFERENCE_BASE_URL
  -u MANUAL_RUN_SLUG
  -u MSMU_SMOKE_INDICES
  -u NO_RESUME
  -u OUTPUT
  -u RESOLVE_PATHS_ONLY
  -u RUN_METADATA
  -u RUN_NAME
  -u RUN_SCORE
  -u SCORE_ONLY
  -u SCORE_OUTPUT_DIR
  -u VALIDATION_REPORT
)

group_alive() {
  local pid="$1"
  ps -eo pgid=,stat= | awk -v target="${pid}" '
    $1 == target && $2 !~ /^Z/ { found=1 }
    END { exit(found ? 0 : 1) }
  '
}

write_active_state() {
  if [[ -z "${active_service_pid}" && -z "${active_inference_pid}" ]]; then
    rm -f "${active_file}"
    return 0
  fi
  local temporary="${active_file}.tmp.$$"
  {
    printf 'repository_sha=%s\n' "${repository_sha}"
    printf 'model=%s\n' "${active_model}"
    printf 'service_process_group_id=%s\n' "${active_service_pid}"
    printf 'inference_process_group_id=%s\n' "${active_inference_pid}"
    printf 'updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${temporary}"
  mv "${temporary}" "${active_file}"
}

stop_owned_group() {
  local pid="$1" label="$2"
  [[ -n "${pid}" ]] || return 0
  if ! group_alive "${pid}"; then
    wait "${pid}" 2>/dev/null || true
    return 0
  fi

  echo "[msmu-batch] stopping owned ${label} process group ${pid} with TERM"
  kill -TERM -- "-${pid}" 2>/dev/null || true
  local deadline=$(( $(date +%s) + process_stop_timeout ))
  while group_alive "${pid}" && (( $(date +%s) < deadline )); do
    sleep "${poll_seconds}"
  done
  if group_alive "${pid}"; then
    echo "[msmu-batch] owned ${label} process group ${pid} ignored TERM; sending KILL" >&2
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup_owned_processes() {
  local label="$1"
  if [[ -n "${active_inference_pid}" ]]; then
    stop_owned_group "${active_inference_pid}" "inference"
    active_inference_pid=""
    write_active_state
  fi
  if [[ -n "${active_service_pid}" ]]; then
    stop_owned_group "${active_service_pid}" "vLLM service"
    active_service_pid=""
    write_active_state
  fi
  if [[ -n "${label}" ]]; then
    echo "[msmu-batch] ${label}; rerun the same command to resume" >&2
  fi
}

handle_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  local label=""
  if (( status != 0 )); then
    label="stopped with status=${status}"
  fi
  cleanup_owned_processes "${label}"
  exit "${status}"
}

handle_signal() {
  local status="$1" signal_name="$2"
  trap - EXIT HUP INT TERM
  cleanup_owned_processes "received ${signal_name}"
  exit "${status}"
}

trap handle_exit EXIT
trap 'handle_signal 129 HUP' HUP
trap 'handle_signal 130 INT' INT
trap 'handle_signal 143 TERM' TERM

if [[ -f "${active_file}" ]]; then
  stale_model="$(awk -F= '$1 == "model" {print $2}' "${active_file}")"
  stale_service_pid="$(awk -F= '$1 == "service_process_group_id" {print $2}' "${active_file}")"
  stale_inference_pid="$(awk -F= '$1 == "inference_process_group_id" {print $2}' "${active_file}")"
  stale_alive=()
  for stale_pid in "${stale_service_pid}" "${stale_inference_pid}"; do
    if [[ "${stale_pid}" =~ ^[0-9]+$ ]] && group_alive "${stale_pid}"; then
      stale_alive+=("${stale_pid}")
    fi
  done
  if (( ${#stale_alive[@]} > 0 )); then
    echo "[msmu-batch] a process group recorded by an earlier batch is still alive:" >&2
    echo "[msmu-batch] model=${stale_model} process_group_ids=${stale_alive[*]}" >&2
    echo "[msmu-batch] it was left untouched; inspect it before resuming" >&2
    exit 4
  fi
  echo "[msmu-batch] removing stale inactive process record ${active_file}"
  rm -f "${active_file}"
fi

model_devices() {
  case "$1" in
    llava_next_yi_34b|internvl3_38b) printf '0,1' ;;
    *) printf '0' ;;
  esac
}

model_run_slug() {
  case "$1" in
    llava_next_mistral_7b) printf 'llava-next-mistral-7b-vllm' ;;
    llava_next_yi_34b) printf 'llava-next-yi-34b-vllm' ;;
    internvl3_8b) printf 'internvl3-8b-vllm' ;;
    internvl3_38b) printf 'internvl3-38b-vllm' ;;
    qwen25_vl_base) printf 'qwen25-vl-base' ;;
    qwen25_vl_32b) printf 'qwen25-vl-32b' ;;
    qwen3_vl_2b) printf 'qwen3-vl-2b' ;;
    qwen3_vl_4b) printf 'qwen3-vl-4b' ;;
    qwen3_vl_8b) printf 'qwen3-vl-8b' ;;
    qwen3_vl_32b) printf 'qwen3-vl-32b' ;;
    ssr) printf 'ssr-rgb-only' ;;
    ssr_native) printf 'ssr-native' ;;
    spatialrgpt) printf 'spatialrgpt-rgb-only' ;;
    3dthinker) printf '3dthinker-fair' ;;
    3dthinker_native) printf '3dthinker-native' ;;
    spatialbot) printf 'spatialbot-rgb-only' ;;
    spatialbot_native) printf 'spatialbot-native' ;;
    *) return 2 ;;
  esac
}

served_model_name() {
  case "$1" in
    llava_next_mistral_7b) printf 'llava-next-mistral-7b-msmu' ;;
    llava_next_yi_34b) printf 'llava-next-yi-34b-msmu' ;;
    internvl3_8b) printf 'internvl3-8b-msmu' ;;
    internvl3_38b) printf 'internvl3-38b-msmu' ;;
    *) return 2 ;;
  esac
}

is_vllm_model() {
  case "$1" in
    llava_next_mistral_7b|llava_next_yi_34b|internvl3_8b|internvl3_38b) return 0 ;;
    *) return 1 ;;
  esac
}

inference_port_is_listening() {
  "${LATENT_PYTHON}" -c '
import socket
import sys

sock = socket.socket()
sock.settimeout(1.0)
try:
    status = sock.connect_ex(("127.0.0.1", 18081))
finally:
    sock.close()
sys.exit(0 if status == 0 else 1)
'
}

gpu_compute_pids() {
  local devices="$1" gpu_id
  local -a gpu_ids
  IFS=',' read -r -a gpu_ids <<< "${devices}"
  for gpu_id in "${gpu_ids[@]}"; do
    nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | awk -v gpu="${gpu_id}" '/^[[:space:]]*[0-9]+[[:space:]]*$/ {
          gsub(/[[:space:]]/, "");
          print gpu ":" $0
        }'
  done
}

wait_for_idle_gpus() {
  local devices="$1" timeout_seconds="$2" reason="$3"
  if [[ "${manual_dry_run}" == "1" ]]; then
    echo "[msmu-batch] dry-run: wait for GPUs ${devices} (${reason})"
    return 0
  fi
  local deadline=$(( $(date +%s) + timeout_seconds ))
  local last_report=0 pids now
  while true; do
    pids="$(gpu_compute_pids "${devices}")"
    if [[ -z "${pids}" ]]; then
      echo "[msmu-batch] GPUs ${devices} are idle (${reason})"
      return 0
    fi
    now="$(date +%s)"
    if (( now >= deadline )); then
      echo "[msmu-batch] timed out waiting for GPUs ${devices} (${reason}); processes were left untouched: ${pids//$'\n'/,}" >&2
      return 4
    fi
    if (( now - last_report >= 60 )); then
      echo "[msmu-batch] waiting for GPUs ${devices} (${reason}); existing processes left untouched: ${pids//$'\n'/,}"
      last_report="${now}"
    fi
    sleep "${poll_seconds}"
  done
}

latest_progress_mtime() {
  local directory="$1"
  [[ -d "${directory}" ]] || {
    printf '0'
    return 0
  }
  local latest
  latest="$(
    find "${directory}" -type f \
      \( -name 'predictions.jsonl.journal.jsonl' -o -name 'predictions.infer.log' \
         -o -name 'vllm_serve.log' \) \
      -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1
  )"
  latest="${latest%%.*}"
  printf '%s' "${latest:-0}"
}

start_vllm_service() {
  local model="$1" expected="$2"
  if inference_port_is_listening; then
    echo "[msmu-batch] port 18081 is already occupied; existing service was left untouched" >&2
    return 4
  fi

  echo "[msmu-batch] starting vLLM service for ${model}"
  setsid env "${child_env_cleanup[@]}" \
    bash "${STAGE3_SCRIPT}" "${model}" serve &
  active_service_pid=$!
  active_model="${model}"
  write_active_state

  local deadline=$(( $(date +%s) + service_timeout ))
  local last_report=0 response now
  while true; do
    if ! group_alive "${active_service_pid}"; then
      wait "${active_service_pid}" 2>/dev/null || true
      echo "[msmu-batch] vLLM service for ${model} exited before readiness" >&2
      active_service_pid=""
      write_active_state
      return 1
    fi
    response="$(
      curl --silent --fail --max-time 5 \
        "http://127.0.0.1:18081/v1/models" 2>/dev/null || true
    )"
    if [[ "${response}" == *"${expected}"* ]]; then
      echo "[msmu-batch] vLLM service ready for ${model}: ${expected}"
      return 0
    fi
    now="$(date +%s)"
    if (( now >= deadline )); then
      echo "[msmu-batch] timed out waiting for vLLM service ${model}" >&2
      return 124
    fi
    if (( now - last_report >= 60 )); then
      echo "[msmu-batch] waiting for vLLM service ${model} to become ready"
      last_report="${now}"
    fi
    sleep "${poll_seconds}"
  done
}

wait_for_inference_port_free() {
  local deadline=$(( $(date +%s) + process_stop_timeout ))
  while inference_port_is_listening; do
    if (( $(date +%s) >= deadline )); then
      echo "[msmu-batch] port 18081 remained occupied after the owned vLLM service stopped; no unrelated process was terminated" >&2
      return 4
    fi
    sleep "${poll_seconds}"
  done
  return 0
}

run_inference_with_watchdog() {
  local model="$1" progress_directory="$2"
  echo "[msmu-batch] starting full inference for ${model}"
  setsid env "${child_env_cleanup[@]}" \
    bash "${STAGE3_SCRIPT}" "${model}" infer &
  active_inference_pid=$!
  active_model="${model}"
  write_active_state

  local last_seen last_activity now current_mtime watchdog_status=0 last_report=0
  last_seen="$(latest_progress_mtime "${progress_directory}")"
  last_activity="$(date +%s)"
  while group_alive "${active_inference_pid}"; do
    current_mtime="$(latest_progress_mtime "${progress_directory}")"
    now="$(date +%s)"
    if (( current_mtime > last_seen )); then
      last_seen="${current_mtime}"
      last_activity="${now}"
    fi
    if (( now - last_activity >= stall_timeout )); then
      echo "[msmu-batch] inference watchdog: ${model} had no durable progress for ${stall_timeout}s" >&2
      watchdog_status=124
      stop_owned_group "${active_inference_pid}" "stalled ${model} inference"
      break
    fi
    if (( now - last_report >= 300 )); then
      echo "[msmu-batch] watchdog ${model}: process alive; seconds_since_progress=$(( now - last_activity ))"
      last_report="${now}"
    fi
    sleep "${poll_seconds}"
  done

  local inference_status
  if (( watchdog_status != 0 )); then
    inference_status="${watchdog_status}"
  else
    set +e
    wait "${active_inference_pid}"
    inference_status=$?
    set -e
    stop_owned_group "${active_inference_pid}" "${model} inference"
  fi
  active_inference_pid=""
  write_active_state
  return "${inference_status}"
}

record_status() {
  local model="$1" event="$2" status="$3" attempt="$4"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${repository_sha}" "${model}" \
    "${event}" "${status}" "${attempt}" >> "${status_file}"
}

mark_complete() {
  local model="$1" attempt="$2"
  local marker="${completion_dir}/${model}.complete"
  local temporary="${marker}.tmp.$$"
  {
    printf 'repository_sha=%s\n' "${repository_sha}"
    printf 'model=%s\n' "${model}"
    printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'attempt=%s\n' "${attempt}"
    printf 'validation=passed_full987\n'
  } > "${temporary}"
  mv "${temporary}" "${marker}"
}

if [[ "${preflight_only}" == "1" ]]; then
  if inference_port_is_listening; then
    echo "[msmu-batch] preflight failed: port 18081 is already occupied; existing service was left untouched" >&2
    exit 4
  fi
  required_gpu_devices="0"
  for model in "${models[@]}"; do
    if [[ "$(model_devices "${model}")" == "0,1" ]]; then
      required_gpu_devices="0,1"
      break
    fi
  done
  preflight_pids="$(gpu_compute_pids "${required_gpu_devices}")"
  if [[ -n "${preflight_pids}" ]]; then
    echo "[msmu-batch] preflight failed: GPU compute processes were left untouched: ${preflight_pids//$'\n'/,}" >&2
    exit 4
  fi
  echo "[msmu-batch] preflight passed: plan=${plan}; configuration, executables, lock, port 18081, and GPUs ${required_gpu_devices} are ready"
  exit 0
fi

if [[ "${manual_dry_run}" == "1" ]]; then
  for model in "${models[@]}"; do
    devices="$(model_devices "${model}")"
    echo "[msmu-batch] dry-run: model=${model} devices=${devices}"
    if is_vllm_model "${model}"; then
      echo "[msmu-batch] dry-run: bash ${STAGE3_SCRIPT} ${model} serve"
    fi
    MANUAL_DRY_RUN=1 env "${child_env_cleanup[@]}" \
      bash "${STAGE3_SCRIPT}" "${model}" infer
    echo "[msmu-batch] dry-run: verify GPU release for ${devices}"
  done
  echo "[msmu-batch] dry-run complete; no GPU/API/judge action was taken"
  exit 0
fi

failed_models=()
for model in "${models[@]}"; do
  marker="${completion_dir}/${model}.complete"
  if [[ "${skip_completed}" == "1" ]] && marker_matches_plan "${marker}"; then
    echo "[msmu-batch] skip completed model=${model}"
    record_status "${model}" skipped 0 0
    continue
  fi

  devices="$(model_devices "${model}")"
  run_slug="$(model_run_slug "${model}")"
  progress_directory="${OUTPUT_ROOT}/03_full987/${run_slug}"
  model_succeeded=0

  for (( attempt=1; attempt<=model_attempts; attempt++ )); do
    echo "[msmu-batch] model=${model} attempt=${attempt}/${model_attempts} devices=${devices}"
    record_status "${model}" started 0 "${attempt}"
    if wait_for_idle_gpus "${devices}" "${gpu_wait_timeout}" "before ${model}"; then
      attempt_status=0
    else
      attempt_status=$?
    fi
    if (( attempt_status == 0 )); then
      if is_vllm_model "${model}"; then
        expected_served_name="$(served_model_name "${model}")"
        if start_vllm_service "${model}" "${expected_served_name}"; then
          attempt_status=0
        else
          attempt_status=$?
        fi
      fi

      if (( attempt_status == 0 )); then
        set +e
        run_inference_with_watchdog "${model}" "${progress_directory}"
        attempt_status=$?
        set -e
      fi

      if [[ -n "${active_service_pid}" ]]; then
        stop_owned_group "${active_service_pid}" "${model} vLLM service"
        active_service_pid=""
        write_active_state
        if wait_for_inference_port_free; then
          port_status=0
        else
          port_status=$?
          if (( attempt_status == 0 )); then attempt_status="${port_status}"; fi
        fi
      fi

      if wait_for_idle_gpus "${devices}" "${gpu_release_timeout}" "after ${model}"; then
        release_status=0
      else
        release_status=$?
        if (( attempt_status == 0 )); then attempt_status="${release_status}"; fi
      fi
    fi

    if (( attempt_status == 0 )); then
      mark_complete "${model}" "${attempt}"
      record_status "${model}" completed 0 "${attempt}"
      echo "[msmu-batch] completed model=${model}; full validator passed and GPUs released"
      model_succeeded=1
      break
    fi

    record_status "${model}" failed "${attempt_status}" "${attempt}"
    echo "[msmu-batch] model=${model} attempt=${attempt} failed status=${attempt_status}" >&2
    if (( attempt < model_attempts )); then
      echo "[msmu-batch] retrying ${model} from its journal in ${retry_delay}s"
      sleep "${retry_delay}"
    fi
  done

  if (( model_succeeded == 0 )); then
    failed_models+=("${model}")
    if [[ "${continue_on_error}" != "1" ]]; then
      echo "[msmu-batch] aborting after ${model}; rerun the same command after fixing the cause" >&2
      exit 1
    fi
  fi
done

if (( ${#failed_models[@]} > 0 )); then
  echo "[msmu-batch] completed with failed models: ${failed_models[*]}" >&2
  exit 1
fi

echo "[msmu-batch] all ${#models[@]} local inference tracks completed and passed full validation"
echo "[msmu-batch] GPUs are released; start the separate judge/scoring phase when ready"
