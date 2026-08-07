#!/usr/bin/env bash
# Run the locked SPBench-SI InternVL3-78B track through test/full inference,
# mandatory validation, exact-target dual-protocol scoring, and report rebuild.

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
  bash scripts/spbench_si/run_internvl3_78b_evaluation.sh
  bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --check
  bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --status
  bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --dry-run
  bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --faq

The default command owns one four-GPU vLLM service and runs:
  current test gate -> full-1009 inference -> full validator
  -> exact InternVL3-78B main + compatibility scoring -> global report

Re-running reuses a current gate, inference journal, valid prediction, and
complete score. Existing ports and GPU processes are never terminated. Only
process groups started by this controller are stopped.

Configuration:
  SPBENCH_SI_INTERNVL3_78B_GPU_IDS  exactly four 80GB GPU ids (default: 0,1,2,3)
  SPBENCH_SI_INTERNVL3_78B_PORT     owned local vLLM port (default: 18102)
  SPBENCH_SI_EVAL_*                 timeout/poll overrides; see the runbook
EOF
}

faq() {
  cat <<'EOF'
Q: 迁移到四卡服务器后第一条命令是什么？
A: 先运行 --check；它只读检查数据、模型、四卡、端口、环境和绑定。

Q: 如何启动完整评测？
A: 无参数运行本脚本；顺序是 test -> full-1009 -> validator -> score -> report。

Q: 中断后怎么办？
A: 重新运行同一命令；合法 gate、fsync journal、完整 prediction 和 score 会复用。

Q: 脚本会停止 burn、未知服务或别人的 GPU 进程吗？
A: 不会。端口或任一 GPU 忙时 fail closed，只清理脚本自己启动的进程组。

Q: 是否会创建一套模型专属结果目录？
A: 不会。结果写回标准 SPBENCH_SI_OUTPUT_ROOT，并原地把 20/21 报告重建为 21/21。
EOF
}

operation="${1:-run}"
if (( $# > 1 )); then
  usage >&2
  exit 2
fi
case "${operation}" in
  run|--check|--status|--dry-run) ;;
  --faq)
    faq
    exit 0
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  *)
    echo "[spbench-si-78b-eval] unsupported operation: ${operation}" >&2
    usage >&2
    exit 2
    ;;
esac

fail() {
  echo "[spbench-si-78b-eval] $*" >&2
  exit 2
}

blocked() {
  echo "[spbench-si-78b-eval] $*" >&2
  exit 4
}

require_positive_integer() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    fail "${name} must be a positive integer"
  fi
}

[[ -n "${SPBENCH_SI_OUTPUT_ROOT:-}" ]] || fail "set SPBENCH_SI_OUTPUT_ROOT"
gpu_ids="${SPBENCH_SI_INTERNVL3_78B_GPU_IDS:-0,1,2,3}"
port="${SPBENCH_SI_INTERNVL3_78B_PORT:-18102}"
service_timeout="${SPBENCH_SI_EVAL_SERVICE_TIMEOUT_SECONDS:-1800}"
stop_timeout="${SPBENCH_SI_EVAL_STOP_TIMEOUT_SECONDS:-120}"
gpu_release_timeout="${SPBENCH_SI_EVAL_GPU_RELEASE_TIMEOUT_SECONDS:-600}"
poll_seconds="${SPBENCH_SI_EVAL_POLL_SECONDS:-10}"

require_positive_integer SPBENCH_SI_INTERNVL3_78B_PORT "${port}"
if (( port > 65535 )); then
  fail "SPBENCH_SI_INTERNVL3_78B_PORT must be at most 65535"
fi
require_positive_integer SPBENCH_SI_EVAL_SERVICE_TIMEOUT_SECONDS "${service_timeout}"
require_positive_integer SPBENCH_SI_EVAL_STOP_TIMEOUT_SECONDS "${stop_timeout}"
require_positive_integer SPBENCH_SI_EVAL_GPU_RELEASE_TIMEOUT_SECONDS "${gpu_release_timeout}"
require_positive_integer SPBENCH_SI_EVAL_POLL_SECONDS "${poll_seconds}"
if [[ ! "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
  fail "SPBENCH_SI_INTERNVL3_78B_GPU_IDS must contain exactly four comma-separated GPU ids"
fi
IFS=',' read -r -a selected_gpus <<<"${gpu_ids}"
unique_gpu_count="$(printf '%s\n' "${selected_gpus[@]}" | awk '!seen[$0]++ {count++} END {print count + 0}')"
if (( unique_gpu_count != 4 )); then
  fail "SPBENCH_SI_INTERNVL3_78B_GPU_IDS must contain four distinct GPU ids"
fi

export SPBENCH_SI_INTERNVL3_78B_GPU_IDS="${gpu_ids}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
base_url="http://127.0.0.1:${port}/v1"
export SPBENCH_SI_INTERNVL3_78B_BASE_URLS="${base_url}"

resolved="$("${SPBENCH_SI_PYTHON}" - "${SPBENCH_SI_OUTPUT_ROOT}" <<'PY'
import sys

from spatial_vlm_eval.benchmarks.spbench_si.inference import track_directory
from spatial_vlm_eval.benchmarks.spbench_si.profiles import get_profile
from spatial_vlm_eval.benchmarks.spbench_si.scorer import (
    AUDIT_SCORER_PROTOCOL,
    SCORER_PROTOCOL,
)

profile = get_profile("internvl3_78b")
track = track_directory(sys.argv[1], profile)
print("\t".join([
    profile.served_model_name or "",
    str(track),
    str(track / "predictions.jsonl"),
    str(track / "prediction_validation.json"),
    str(track / "scores" / SCORER_PROTOCOL),
    str(track / "scores" / AUDIT_SCORER_PROTOCOL),
]))
PY
)"
IFS=$'\t' read -r served_model track predictions validation_report score_dir audit_score_dir <<<"${resolved}"
[[ -n "${served_model}" ]] || fail "registry did not provide the served model name"
report="${SPBENCH_SI_OUTPUT_ROOT%/}/spbench-si-result.md"
control_root="${SPBENCH_SI_OUTPUT_ROOT%/}/_single_model_evaluation"
logs_root="${control_root}/logs"

echo "[spbench-si-78b-eval] profile=${PROFILE} gpu_ids=${gpu_ids}"
echo "[spbench-si-78b-eval] predictions=${predictions}"
echo "[spbench-si-78b-eval] main_score_dir=${score_dir}"
echo "[spbench-si-78b-eval] audit_score_dir=${audit_score_dir}"
echo "[spbench-si-78b-eval] global_report=${report}"

score_status() {
  "${SPBENCH_SI_PYTHON}" - "${predictions}" <<'PY'
import sys
from spatial_vlm_eval.benchmarks.spbench_si.score_results import score_state

candidate = score_state(sys.argv[1])
print(f"{candidate.state}\t{candidate.predictions}\t{candidate.reason}")
PY
}

if [[ "${operation}" == "--status" ]]; then
  "${SPBENCH_SI_PYTHON}" - "${track}" "${predictions}" "${validation_report}" "${report}" <<'PY'
import json
import sys
from pathlib import Path

track, predictions, validation, report = map(Path, sys.argv[1:])
validation_passed = False
if validation.is_file():
    try:
        validation_passed = json.loads(validation.read_text(encoding="utf-8")).get("passed") is True
    except (OSError, json.JSONDecodeError):
        pass
print("test_gate\t" + ("present" if (track / "test_gate.json").is_file() else "missing"))
print("full_prediction\t" + ("present" if predictions.is_file() else "missing"))
print("full_validator\t" + ("passed" if validation_passed else "missing_or_failed"))
print("report\t" + (str(report) if report.is_file() else "missing"))
PY
  score_status
  exit 0
fi

if [[ "${operation}" == "--dry-run" ]]; then
  echo "[spbench-si-78b-eval] dry-run workflow: test gate -> full-1009 -> validator -> exact dual-protocol score -> report"
  bash "${SERVE_SCRIPT}" --model "${PROFILE}" --gpu-ids "${gpu_ids}" --port "${port}" --dry-run
  bash "${INFERENCE_SCRIPT}" --stage test --model "${PROFILE}" --dry-run
  bash "${INFERENCE_SCRIPT}" --stage full --model "${PROFILE}" --dry-run
  echo "[spbench-si-78b-eval] dry-run: bash ${VALIDATION_SCRIPT} --predictions ${predictions} --report ${validation_report}"
  echo "[spbench-si-78b-eval] dry-run: bash ${SCORING_SCRIPT} --predictions ${predictions}"
  echo "[spbench-si-78b-eval] dry-run: bash ${REPORT_SCRIPT} --check"
  echo "[spbench-si-78b-eval] dry-run: bash ${REPORT_SCRIPT}"
  echo "[spbench-si-78b-eval] dry-run complete; no GPU/service/inference/scoring/report action was taken"
  exit 0
fi

port_available() {
  "${SPBENCH_SI_PYTHON}" - "${port}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
}

check_lock_available() {
  "${SPBENCH_SI_PYTHON}" - "${control_root}/lock" <<'PY'
import fcntl
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(0)
with path.open("r") as handle:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(1)
PY
}

check_readiness() {
  local command model_path output_root_resolved repo_root
  for command in bash flock nvidia-smi ps setsid; do
    command -v "${command}" >/dev/null 2>&1 || fail "required command is unavailable: ${command}"
  done
  if [[ "${SPBENCH_SI_PYTHON}" == */* ]]; then
    [[ -x "${SPBENCH_SI_PYTHON}" ]] || fail "SPBENCH_SI_PYTHON is unavailable: ${SPBENCH_SI_PYTHON}"
  else
    command -v "${SPBENCH_SI_PYTHON}" >/dev/null 2>&1 || \
      fail "SPBENCH_SI_PYTHON command is unavailable: ${SPBENCH_SI_PYTHON}"
  fi
  [[ -f "${MODEL_PROBE}" ]] || fail "model readiness probe is unavailable: ${MODEL_PROBE}"
  [[ -f "${GPU_PREFLIGHT}" ]] || fail "GPU preflight is unavailable: ${GPU_PREFLIGHT}"
  [[ -n "${SPBENCH_SI_PARQUET:-}" ]] || fail "set SPBENCH_SI_PARQUET"
  [[ -f "${SPBENCH_SI_PARQUET}" ]] || fail "SPBENCH_SI_PARQUET is unavailable: ${SPBENCH_SI_PARQUET}"
  [[ -n "${SPBENCH_SI_IMAGES_ARCHIVE:-}" ]] || fail "set SPBENCH_SI_IMAGES_ARCHIVE"
  [[ -f "${SPBENCH_SI_IMAGES_ARCHIVE}" ]] || fail "SPBENCH_SI_IMAGES_ARCHIVE is unavailable: ${SPBENCH_SI_IMAGES_ARCHIVE}"
  model_path="${INTERNVL3_78B_MODEL:-}"
  [[ -n "${model_path}" ]] || fail "set INTERNVL3_78B_MODEL"
  [[ -e "${model_path}" ]] || fail "INTERNVL3_78B_MODEL is unavailable: ${model_path}"
  repo_root="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
  output_root_resolved="$("${SPBENCH_SI_PYTHON}" - "${SPBENCH_SI_OUTPUT_ROOT}" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).resolve())
PY
)"
  case "${output_root_resolved}/" in
    "${repo_root}/"*) fail "SPBENCH_SI_OUTPUT_ROOT must stay outside the repository" ;;
  esac
  if ! port_available; then
    blocked "port ${port} is occupied; existing service was left untouched"
  fi
  if ! check_lock_available; then
    blocked "another InternVL3-78B evaluation holds ${control_root}/lock"
  fi
  MIN_FREE_GPU_MIB=76000 MIN_GPU_COUNT=4 bash "${GPU_PREFLIGHT}"
  "${SPBENCH_SI_PYTHON}" - <<'PY'
import json
from spatial_vlm_eval.benchmarks.spbench_si.inference import inspect_local_gpus
from spatial_vlm_eval.benchmarks.spbench_si.profiles import get_profile

print(json.dumps(inspect_local_gpus(get_profile("internvl3_78b"), "vllm"), sort_keys=True))
PY
  bash "${SERVE_SCRIPT}" --model "${PROFILE}" --gpu-ids "${gpu_ids}" --port "${port}" --dry-run
  bash "${INFERENCE_SCRIPT}" --check --model "${PROFILE}"
}

check_readiness
if [[ "${operation}" == "--check" ]]; then
  echo "[spbench-si-78b-eval] CHECK registration=valid"
  echo "[spbench-si-78b-eval] CHECK four_gpu_preflight=passed"
  echo "[spbench-si-78b-eval] CHECK owned_endpoint=available"
  echo "[spbench-si-78b-eval] CHECK evaluation_lock=available"
  echo "[spbench-si-78b-eval] CHECK target_scoring=exact_predictions_path"
  echo "[spbench-si-78b-eval] CHECK global_report=${report}"
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
echo "[spbench-si-78b-eval] controller_log=${controller_log}"

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
    echo "[spbench-si-78b-eval] stopping owned ${label} process group ${pid} with TERM"
    kill -TERM -- "-${pid}" 2>/dev/null || true
  fi
  deadline=$(( $(date +%s) + stop_timeout ))
  while group_alive "${pid}" && (( $(date +%s) < deadline )); do
    sleep "${poll_seconds}"
  done
  if group_alive "${pid}"; then
    echo "[spbench-si-78b-eval] owned ${label} ignored TERM; sending KILL" >&2
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
    echo "[spbench-si-78b-eval] stopped status=${status}; rerun the same command to resume" >&2
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
  echo "[spbench-si-78b-eval] START ${label}; log=${log}"
  setsid "$@" >"${log}" 2>&1 &
  active_step_pid=$!
  set +e
  wait "${active_step_pid}"
  status=$?
  set -e
  active_step_pid=""
  if (( status != 0 )); then
    echo "[spbench-si-78b-eval] FAIL ${label} status=${status}; log=${log}" >&2
    tail -80 "${log}" >&2 || true
    return "${status}"
  fi
  echo "[spbench-si-78b-eval] PASS ${label}"
}

wait_for_model() {
  local deadline
  deadline=$(( $(date +%s) + service_timeout ))
  while true; do
    if ! group_alive "${active_service_pid}"; then
      echo "[spbench-si-78b-eval] vLLM exited before readiness; log=${service_log}" >&2
      return 1
    fi
    if "${SPBENCH_SI_PYTHON}" "${MODEL_PROBE}" \
      --base-url "${base_url}" --expected-model "${served_model}" --timeout 5; then
      echo "[spbench-si-78b-eval] ready model=${served_model} endpoint=${base_url}"
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      echo "[spbench-si-78b-eval] readiness timeout; log=${service_log}" >&2
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
      echo "[spbench-si-78b-eval] GPU release timeout; processes left untouched: ${pids//$'\n'/,}" >&2
      return 4
    fi
    sleep "${poll_seconds}"
  done
}

prediction_is_reusable() {
  local precheck_log
  [[ -f "${predictions}" ]] || return 1
  precheck_log="${logs_root}/${run_id}.${PROFILE}.preexisting-validation.log"
  if bash "${VALIDATION_SCRIPT}" --predictions "${predictions}" \
    --report "${validation_report}" >"${precheck_log}" 2>&1; then
    echo "[spbench-si-78b-eval] SKIP inference; existing full-1009 prediction revalidated"
    return 0
  fi
  echo "[spbench-si-78b-eval] existing prediction is not reusable; inference will resume; log=${precheck_log}"
  return 1
}

if ! prediction_is_reusable; then
  service_log="${logs_root}/${run_id}.${PROFILE}.vllm.log"
  echo "[spbench-si-78b-eval] START owned four-GPU vLLM; log=${service_log}"
  setsid bash "${SERVE_SCRIPT}" --model "${PROFILE}" --gpu-ids "${gpu_ids}" --port "${port}" \
    >"${service_log}" 2>&1 &
  active_service_pid=$!
  wait_for_model
  run_step "test gate" "${logs_root}/${run_id}.${PROFILE}.test.log" \
    bash "${INFERENCE_SCRIPT}" --stage test --model "${PROFILE}"
  run_step "full-1009 inference" "${logs_root}/${run_id}.${PROFILE}.full.log" \
    bash "${INFERENCE_SCRIPT}" --stage full --model "${PROFILE}"
  stop_owned_group "${active_service_pid}" "vLLM service"
  active_service_pid=""
  wait_for_gpu_release
fi

run_step "full validator" "${logs_root}/${run_id}.${PROFILE}.validation.log" \
  bash "${VALIDATION_SCRIPT}" --predictions "${predictions}" --report "${validation_report}"
run_step "exact-target dual-protocol scoring" "${logs_root}/${run_id}.${PROFILE}.scoring.log" \
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

echo "[spbench-si-78b-eval] COMPLETE profile=${PROFILE} score_state=complete report=${report}"
