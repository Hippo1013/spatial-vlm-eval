#!/usr/bin/env bash
# Run the locked CV-Bench InternVL3-78B track through test/full inference,
# mandatory validation, exact-target scoring, and the global report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

PROFILE="internvl3_78b"
SERVE_SCRIPT="${SCRIPT_DIR}/serve_vllm_profile.sh"
INFERENCE_SCRIPT="${SCRIPT_DIR}/run_inference.sh"
VALIDATION_SCRIPT="${SCRIPT_DIR}/validate_predictions.sh"
SCORING_SCRIPT="${SCRIPT_DIR}/score_results.sh"
REPORT_SCRIPT="${SCRIPT_DIR}/build_results_report.sh"
MODEL_PROBE="${SCRIPT_DIR}/../msmu/_probe_openai_models.py"
GPU_PREFLIGHT="${SCRIPT_DIR}/../msmu/gpu_preflight.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cv_bench/run_internvl3_78b_evaluation.sh
  bash scripts/cv_bench/run_internvl3_78b_evaluation.sh --check
  bash scripts/cv_bench/run_internvl3_78b_evaluation.sh --status
  bash scripts/cv_bench/run_internvl3_78b_evaluation.sh --dry-run

The default command owns one four-GPU vLLM service and runs:
  current test gate -> full-2638 inference -> full validator
  -> exact InternVL3-78B scoring -> global CV-Bench report

Re-running the command reuses a current test gate, inference journal, complete
prediction, and complete score. Existing ports and GPU processes are never
terminated. Only process groups started by this controller are stopped.

Configuration:
  CVBENCH_INTERNVL3_78B_GPU_IDS   exactly four 80GB GPU ids (default: 0,1,2,3)
  CVBENCH_INTERNVL3_78B_PORT      owned local vLLM port (default: 18101)
  CVBENCH_EVAL_*                  timeout/poll overrides; see the runbook
EOF
}

operation="${1:-run}"
if (( $# > 1 )); then
  usage >&2
  exit 2
fi
case "${operation}" in
  run|--check|--status|--dry-run) ;;
  --help|-h)
    usage
    exit 0
    ;;
  *)
    echo "[cv-bench-78b-eval] unsupported operation: ${operation}" >&2
    usage >&2
    exit 2
    ;;
esac

fail() {
  echo "[cv-bench-78b-eval] $*" >&2
  exit 2
}

blocked() {
  echo "[cv-bench-78b-eval] $*" >&2
  exit 4
}

require_positive_integer() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    fail "${name} must be a positive integer"
  fi
}

[[ -n "${CVBENCH_OUTPUT_ROOT:-}" ]] || fail "set CVBENCH_OUTPUT_ROOT"
gpu_ids="${CVBENCH_INTERNVL3_78B_GPU_IDS:-0,1,2,3}"
port="${CVBENCH_INTERNVL3_78B_PORT:-18101}"
service_timeout="${CVBENCH_EVAL_SERVICE_TIMEOUT_SECONDS:-1800}"
stop_timeout="${CVBENCH_EVAL_STOP_TIMEOUT_SECONDS:-120}"
gpu_release_timeout="${CVBENCH_EVAL_GPU_RELEASE_TIMEOUT_SECONDS:-600}"
poll_seconds="${CVBENCH_EVAL_POLL_SECONDS:-10}"

require_positive_integer CVBENCH_INTERNVL3_78B_PORT "${port}"
if (( port > 65535 )); then
  fail "CVBENCH_INTERNVL3_78B_PORT must be at most 65535"
fi
require_positive_integer CVBENCH_EVAL_SERVICE_TIMEOUT_SECONDS "${service_timeout}"
require_positive_integer CVBENCH_EVAL_STOP_TIMEOUT_SECONDS "${stop_timeout}"
require_positive_integer CVBENCH_EVAL_GPU_RELEASE_TIMEOUT_SECONDS "${gpu_release_timeout}"
require_positive_integer CVBENCH_EVAL_POLL_SECONDS "${poll_seconds}"
if [[ ! "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
  fail "CVBENCH_INTERNVL3_78B_GPU_IDS must contain exactly four comma-separated GPU ids"
fi
IFS=',' read -r -a selected_gpus <<<"${gpu_ids}"
unique_gpu_count="$(printf '%s\n' "${selected_gpus[@]}" | awk '!seen[$0]++ {count++} END {print count + 0}')"
if (( unique_gpu_count != 4 )); then
  fail "CVBENCH_INTERNVL3_78B_GPU_IDS must contain four distinct GPU ids"
fi

export CVBENCH_INTERNVL3_78B_GPU_IDS="${gpu_ids}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
base_url="http://127.0.0.1:${port}/v1"
export CVBENCH_INTERNVL3_78B_BASE_URLS="${base_url}"

resolved="$("${CVBENCH_PYTHON}" - "${CVBENCH_OUTPUT_ROOT}" <<'PY'
import sys

from spatial_vlm_eval.benchmarks.cv_bench.inference import track_directory
from spatial_vlm_eval.benchmarks.cv_bench.profiles import get_profile
from spatial_vlm_eval.benchmarks.cv_bench.scorer import SCORER_PROTOCOL

profile = get_profile("internvl3_78b")
track = track_directory(sys.argv[1], profile)
print("\t".join([
    profile.served_model_name or "",
    str(track),
    str(track / "predictions.jsonl"),
    str(track / "prediction_validation.json"),
    str(track / "scores" / SCORER_PROTOCOL),
]))
PY
)"
IFS=$'\t' read -r served_model track predictions validation_report score_dir <<<"${resolved}"
[[ -n "${served_model}" ]] || fail "registry did not provide the served model name"
report="${CVBENCH_OUTPUT_ROOT%/}/cv-bench-result.md"
control_root="${CVBENCH_OUTPUT_ROOT%/}/_single_model_evaluation"
logs_root="${control_root}/logs"

echo "[cv-bench-78b-eval] profile=${PROFILE} gpu_ids=${gpu_ids}"
echo "[cv-bench-78b-eval] predictions=${predictions}"
echo "[cv-bench-78b-eval] score_dir=${score_dir}"
echo "[cv-bench-78b-eval] global_report=${report}"

score_status() {
  if [[ ! -f "${predictions}" ]]; then
    printf 'missing\t%s\tfull prediction does not exist\n' "${predictions}"
    return 0
  fi
  bash "${SCORING_SCRIPT}" --status --predictions "${predictions}"
}

if [[ "${operation}" == "--status" ]]; then
  bash "${INFERENCE_SCRIPT}" --status --model "${PROFILE}"
  score_status
  if [[ -f "${report}" ]]; then
    echo "report\t${report}"
  else
    echo "missing_report\t${report}"
  fi
  exit 0
fi

if [[ "${operation}" == "--dry-run" ]]; then
  bash "${SERVE_SCRIPT}" --model "${PROFILE}" --gpu-ids "${gpu_ids}" --port "${port}" --dry-run
  bash "${INFERENCE_SCRIPT}" --stage test --model "${PROFILE}" --dry-run
  bash "${INFERENCE_SCRIPT}" --stage full --model "${PROFILE}" --dry-run
  echo "[cv-bench-78b-eval] dry-run: bash ${VALIDATION_SCRIPT} --predictions ${predictions} --report ${validation_report}"
  echo "[cv-bench-78b-eval] dry-run: bash ${SCORING_SCRIPT} --predictions ${predictions}"
  echo "[cv-bench-78b-eval] dry-run: bash ${REPORT_SCRIPT} --check"
  echo "[cv-bench-78b-eval] dry-run: bash ${REPORT_SCRIPT}"
  echo "[cv-bench-78b-eval] dry-run complete; no GPU/service/inference/scoring/report action was taken"
  exit 0
fi

endpoint_open() {
  "${CVBENCH_PYTHON}" - "${port}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(1.0)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

check_readiness() {
  local command model_path output_root_resolved repo_root
  for command in bash flock nvidia-smi ps setsid; do
    command -v "${command}" >/dev/null 2>&1 || fail "required command is unavailable: ${command}"
  done
  if [[ "${CVBENCH_PYTHON}" == */* ]]; then
    [[ -x "${CVBENCH_PYTHON}" ]] || fail "CVBENCH_PYTHON is unavailable: ${CVBENCH_PYTHON}"
  else
    command -v "${CVBENCH_PYTHON}" >/dev/null 2>&1 || \
      fail "CVBENCH_PYTHON command is unavailable: ${CVBENCH_PYTHON}"
  fi
  [[ -f "${MODEL_PROBE}" ]] || fail "model readiness probe is unavailable: ${MODEL_PROBE}"
  [[ -f "${GPU_PREFLIGHT}" ]] || fail "GPU preflight is unavailable: ${GPU_PREFLIGHT}"
  [[ -n "${CVBENCH_DATASET_ROOT:-}" ]] || fail "set CVBENCH_DATASET_ROOT"
  [[ -d "${CVBENCH_DATASET_ROOT}" ]] || fail "CVBENCH_DATASET_ROOT is unavailable: ${CVBENCH_DATASET_ROOT}"
  model_path="${INTERNVL3_78B_MODEL:-}"
  [[ -n "${model_path}" ]] || fail "set INTERNVL3_78B_MODEL"
  [[ -e "${model_path}" ]] || fail "INTERNVL3_78B_MODEL is unavailable: ${model_path}"
  repo_root="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
  output_root_resolved="$("${CVBENCH_PYTHON}" - "${CVBENCH_OUTPUT_ROOT}" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).resolve())
PY
)"
  case "${output_root_resolved}/" in
    "${repo_root}/"*) fail "CVBENCH_OUTPUT_ROOT must stay outside the repository" ;;
  esac
  if endpoint_open; then
    blocked "port ${port} is occupied; existing service was left untouched"
  fi
  MIN_FREE_GPU_MIB=76000 MIN_GPU_COUNT=4 \
    bash "${GPU_PREFLIGHT}"
  "${CVBENCH_PYTHON}" - <<'PY'
import json

from spatial_vlm_eval.benchmarks.cv_bench.inference import inspect_local_gpus
from spatial_vlm_eval.benchmarks.cv_bench.profiles import get_profile

print(json.dumps(inspect_local_gpus(get_profile("internvl3_78b"), "vllm"), sort_keys=True))
PY
  bash "${SERVE_SCRIPT}" --model "${PROFILE}" --gpu-ids "${gpu_ids}" --port "${port}" --dry-run
  bash "${INFERENCE_SCRIPT}" --check --model "${PROFILE}"
}

check_readiness
if [[ "${operation}" == "--check" ]]; then
  echo "[cv-bench-78b-eval] CHECK registration=valid"
  echo "[cv-bench-78b-eval] CHECK four_gpu_preflight=passed"
  echo "[cv-bench-78b-eval] CHECK owned_endpoint=available"
  echo "[cv-bench-78b-eval] CHECK target_scoring=exact_predictions_path"
  echo "[cv-bench-78b-eval] CHECK global_report=${report}"
  exit 0
fi

mkdir -p "${logs_root}"
exec 8>"${control_root}/lock"
if ! flock -n 8; then
  blocked "another InternVL3-78B evaluation holds ${control_root}/lock"
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
controller_log="${logs_root}/${run_id}.controller.log"
exec > >(tee -a "${controller_log}") 2>&1
echo "[cv-bench-78b-eval] controller_log=${controller_log}"

active_service_pid=""
active_step_pid=""

group_alive() {
  local target="$1"
  [[ -n "${target}" ]] || return 1
  ps -eo pgid=,stat= | awk -v target="${target}" '
    $1 == target && $2 !~ /^Z/ { found=1 }
    END { exit(found ? 0 : 1) }
  '
}

stop_owned_group() {
  local pid="$1" label="$2" deadline
  [[ -n "${pid}" ]] || return 0
  if group_alive "${pid}"; then
    echo "[cv-bench-78b-eval] stopping owned ${label} process group ${pid} with TERM"
    kill -TERM -- "-${pid}" 2>/dev/null || true
  fi
  deadline=$(( $(date +%s) + stop_timeout ))
  while group_alive "${pid}" && (( $(date +%s) < deadline )); do
    sleep "${poll_seconds}"
  done
  if group_alive "${pid}"; then
    echo "[cv-bench-78b-eval] owned ${label} ignored TERM; sending KILL" >&2
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup_owned() {
  stop_owned_group "${active_step_pid}" "active step"
  active_step_pid=""
  stop_owned_group "${active_service_pid}" "vLLM service"
  active_service_pid=""
}

handle_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  cleanup_owned
  if (( status != 0 )); then
    echo "[cv-bench-78b-eval] stopped status=${status}; rerun the same command to resume" >&2
  fi
  exit "${status}"
}
trap handle_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

run_step() {
  local label="$1" log="$2" status
  shift 2
  echo "[cv-bench-78b-eval] START ${label}; log=${log}"
  setsid "$@" >"${log}" 2>&1 &
  active_step_pid=$!
  set +e
  wait "${active_step_pid}"
  status=$?
  set -e
  active_step_pid=""
  if (( status != 0 )); then
    echo "[cv-bench-78b-eval] FAIL ${label} status=${status}; log=${log}" >&2
    tail -80 "${log}" >&2 || true
    return "${status}"
  fi
  echo "[cv-bench-78b-eval] PASS ${label}"
}

wait_for_model() {
  local deadline
  deadline=$(( $(date +%s) + service_timeout ))
  while true; do
    if ! group_alive "${active_service_pid}"; then
      echo "[cv-bench-78b-eval] vLLM exited before readiness; log=${service_log}" >&2
      return 1
    fi
    if "${CVBENCH_PYTHON}" "${MODEL_PROBE}" \
      --base-url "${base_url}" --expected-model "${served_model}" --timeout 5; then
      echo "[cv-bench-78b-eval] ready model=${served_model} endpoint=${base_url}"
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      echo "[cv-bench-78b-eval] readiness timeout; log=${service_log}" >&2
      return 124
    fi
    sleep "${poll_seconds}"
  done
}

gpu_compute_pids() {
  local gpu
  for gpu in "${selected_gpus[@]}"; do
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
      echo "[cv-bench-78b-eval] GPU release timeout; processes left untouched: ${pids//$'\n'/,}" >&2
      return 4
    fi
    sleep "${poll_seconds}"
  done
}

prediction_is_reusable() {
  local state precheck_log
  [[ -f "${predictions}" ]] || return 1
  state="$(score_status | awk -F '\t' 'NR == 1 {print $1}')"
  case "${state}" in
    new|retry|complete) ;;
    *) return 1 ;;
  esac
  precheck_log="${logs_root}/${run_id}.${PROFILE}.preexisting-validation.log"
  if bash "${VALIDATION_SCRIPT}" --predictions "${predictions}" \
    --report "${validation_report}" >"${precheck_log}" 2>&1; then
    echo "[cv-bench-78b-eval] SKIP inference; existing full-2638 prediction revalidated"
    return 0
  fi
  echo "[cv-bench-78b-eval] existing prediction is not reusable; inference will resume; log=${precheck_log}"
  return 1
}

if ! prediction_is_reusable; then
  service_log="${logs_root}/${run_id}.${PROFILE}.vllm.log"
  echo "[cv-bench-78b-eval] START owned four-GPU vLLM; log=${service_log}"
  setsid bash "${SERVE_SCRIPT}" --model "${PROFILE}" --gpu-ids "${gpu_ids}" --port "${port}" \
    >"${service_log}" 2>&1 &
  active_service_pid=$!
  wait_for_model
  run_step "test gate" "${logs_root}/${run_id}.${PROFILE}.test.log" \
    bash "${INFERENCE_SCRIPT}" --stage test --model "${PROFILE}"
  run_step "full-2638 inference" "${logs_root}/${run_id}.${PROFILE}.full.log" \
    bash "${INFERENCE_SCRIPT}" --stage full --model "${PROFILE}"
  stop_owned_group "${active_service_pid}" "vLLM service"
  active_service_pid=""
  wait_for_gpu_release
fi

run_step "full validator" "${logs_root}/${run_id}.${PROFILE}.validation.log" \
  bash "${VALIDATION_SCRIPT}" --predictions "${predictions}" --report "${validation_report}"
run_step "exact-target scoring" "${logs_root}/${run_id}.${PROFILE}.scoring.log" \
  bash "${SCORING_SCRIPT}" --predictions "${predictions}"

final_score_state="$(score_status | awk -F '\t' 'NR == 1 {print $1}')"
if [[ "${final_score_state}" != "complete" ]]; then
  fail "target scoring did not reach complete publication gates: ${final_score_state}"
fi

run_step "report preflight" "${logs_root}/${run_id}.report-check.log" \
  bash "${REPORT_SCRIPT}" --check
run_step "global report" "${logs_root}/${run_id}.report.log" \
  bash "${REPORT_SCRIPT}"
[[ -f "${report}" ]] || fail "global report was not created: ${report}"

echo "[cv-bench-78b-eval] COMPLETE profile=${PROFILE} score_state=complete report=${report}"
