#!/usr/bin/env bash
# Complete the locked Q-Spatial InternVL3-78B track on four 80GB GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
export QSPATIAL_PYTHON

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
  bash scripts/q_spatial/run_internvl3_78b_evaluation.sh
  bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --check
  bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --status
  bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --dry-run
  bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --faq

The default command owns one four-GPU vLLM service and runs:
  current test gate -> full-271 inference -> full validator
  -> exact InternVL3-78B scoring -> existing global Q-Spatial report

Formal artifacts stay in the existing QSPATIAL_OUTPUT_ROOT track and report.
Only controller logs use QSPATIAL_OUTPUT_ROOT/_single_model_evaluation/logs.
Re-running reuses a current gate, journal, valid prediction, and complete score.
Existing ports and GPU processes are never terminated; only process groups
started by this controller are stopped.

Configuration:
  QSPATIAL_INTERNVL3_78B_GPU_IDS  exactly four 80GB GPU ids (default: 0,1,2,3)
  QSPATIAL_INTERNVL3_78B_PORT     owned local vLLM port (default: 18101)
  QSPATIAL_EVAL_*                 timeout/poll overrides; see the runbook
EOF
}

faq() {
  cat <<'EOF'
Q: 迁移到四卡服务器后第一步是什么？
A: 先同步现有 20/21 输出根并运行 --check；它只读检查数据、模型、四卡、端口、锁和绑定。

Q: 如何启动完整补测？
A: 无参数运行本脚本；顺序是 test -> full-271 -> validator -> score -> 原报告重建。

Q: 补测会写到新的正式输出目录吗？
A: 不会。prediction/score 写入原 QSPATIAL_OUTPUT_ROOT 的 canonical 轨道，原 q-spatial-result.md 原地更新。

Q: inference、gate 和 scorer 会不会沿用旧协议？
A: 不会。路径、binding 与评分目录都从当前 registry/SCORER_PROTOCOL 解析；旧协议 provenance 不复用。

Q: 中断后怎么办？
A: 重新运行同一命令；合法 gate、fsync journal、完整 prediction 和 complete score 会复用。

Q: 脚本会停止 burn、未知服务或别人的 GPU 进程吗？
A: 不会。请按 GPU burn 手册在运行前人工停、结束后恢复；资源忙时脚本 fail closed。

Q: 怎样确认补测真正完成？
A: 运行 --status；目标是 validator=passed、score=complete、report-completeness=21/21 且 missing 为空。
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
    echo "[q-spatial-78b-eval] unsupported operation: ${operation}" >&2
    usage >&2
    exit 2
    ;;
esac

fail() {
  echo "[q-spatial-78b-eval] $*" >&2
  exit 2
}

blocked() {
  echo "[q-spatial-78b-eval] $*" >&2
  exit 4
}

require_positive_integer() {
  local name="$1" value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    fail "${name} must be a positive integer"
  fi
}

[[ -n "${QSPATIAL_OUTPUT_ROOT:-}" ]] || fail "set QSPATIAL_OUTPUT_ROOT"
gpu_ids="${QSPATIAL_INTERNVL3_78B_GPU_IDS:-0,1,2,3}"
port="${QSPATIAL_INTERNVL3_78B_PORT:-18101}"
service_timeout="${QSPATIAL_EVAL_SERVICE_TIMEOUT_SECONDS:-1800}"
stop_timeout="${QSPATIAL_EVAL_STOP_TIMEOUT_SECONDS:-120}"
gpu_release_timeout="${QSPATIAL_EVAL_GPU_RELEASE_TIMEOUT_SECONDS:-600}"
poll_seconds="${QSPATIAL_EVAL_POLL_SECONDS:-10}"

require_positive_integer QSPATIAL_INTERNVL3_78B_PORT "${port}"
if (( port > 65535 )); then
  fail "QSPATIAL_INTERNVL3_78B_PORT must be at most 65535"
fi
require_positive_integer QSPATIAL_EVAL_SERVICE_TIMEOUT_SECONDS "${service_timeout}"
require_positive_integer QSPATIAL_EVAL_STOP_TIMEOUT_SECONDS "${stop_timeout}"
require_positive_integer QSPATIAL_EVAL_GPU_RELEASE_TIMEOUT_SECONDS "${gpu_release_timeout}"
require_positive_integer QSPATIAL_EVAL_POLL_SECONDS "${poll_seconds}"
if [[ ! "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
  fail "QSPATIAL_INTERNVL3_78B_GPU_IDS must contain exactly four comma-separated GPU ids"
fi
IFS=',' read -r -a selected_gpus <<<"${gpu_ids}"
unique_gpu_count="$(printf '%s\n' "${selected_gpus[@]}" | awk '!seen[$0]++ {count++} END {print count + 0}')"
if (( unique_gpu_count != 4 )); then
  fail "QSPATIAL_INTERNVL3_78B_GPU_IDS must contain four distinct GPU ids"
fi

export QSPATIAL_INTERNVL3_78B_GPU_IDS="${gpu_ids}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
base_url="http://127.0.0.1:${port}/v1"
export QSPATIAL_INTERNVL3_78B_BASE_URLS="${base_url}"

resolved="$("${QSPATIAL_PYTHON}" - "${QSPATIAL_OUTPUT_ROOT}" <<'PY'
import sys

from spatial_vlm_eval.benchmarks.q_spatial.inference import track_directory
from spatial_vlm_eval.benchmarks.q_spatial.profiles import get_profile
from spatial_vlm_eval.benchmarks.q_spatial.scorer import SCORER_PROTOCOL

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
report="${QSPATIAL_OUTPUT_ROOT%/}/q-spatial-result.md"
control_root="${QSPATIAL_OUTPUT_ROOT%/}/_single_model_evaluation"
logs_root="${control_root}/logs"

echo "[q-spatial-78b-eval] profile=${PROFILE} gpu_ids=${gpu_ids}"
echo "[q-spatial-78b-eval] canonical_track=${track}"
echo "[q-spatial-78b-eval] predictions=${predictions}"
echo "[q-spatial-78b-eval] score_dir=${score_dir}"
echo "[q-spatial-78b-eval] existing_global_report=${report}"

score_status() {
  if [[ ! -f "${predictions}" ]]; then
    printf 'missing\t%s\tfull prediction does not exist\n' "${predictions}"
    return 0
  fi
  env QSPATIAL_ENV_FILE=/dev/null \
    bash "${SCORING_SCRIPT}" --status --predictions "${predictions}"
}

report_status() {
  "${QSPATIAL_PYTHON}" - "${QSPATIAL_OUTPUT_ROOT}" "${report}" <<'PY'
import sys
from pathlib import Path

from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILE_SEQUENCE
from spatial_vlm_eval.benchmarks.q_spatial.report import discover_results

root, report = Path(sys.argv[1]), Path(sys.argv[2])
try:
    present = {item.profile for item in discover_results(root)}
except Exception as error:  # status must remain read-only and informative
    print(f"report-completeness\terror\t{type(error).__name__}: {error}")
else:
    missing = [key for key in PROFILE_SEQUENCE if key not in present]
    print(f"report-completeness\t{len(present)}/21\tmissing={','.join(missing)}")
print(f"report\t{'present' if report.is_file() else 'missing'}\t{report}")
PY
}

if [[ "${operation}" == "--status" ]]; then
  "${QSPATIAL_PYTHON}" - "${track}" "${predictions}" "${validation_report}" <<'PY'
import json
import sys
from pathlib import Path

track, predictions, validation = map(Path, sys.argv[1:])

def passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("passed") is True
    except (OSError, json.JSONDecodeError, AttributeError):
        return False

print("test_gate\t" + ("passed" if passed(track / "test_gate.json") else "missing_or_failed"))
print("full_prediction\t" + ("present" if predictions.is_file() else "missing"))
print("full_validator\t" + ("passed" if passed(validation) else "missing_or_failed"))
PY
  score_status
  report_status
  exit 0
fi

if [[ "${operation}" == "--dry-run" ]]; then
  env QSPATIAL_ENV_FILE=/dev/null \
    bash "${SERVE_SCRIPT}" --model "${PROFILE}" --gpu-ids "${gpu_ids}" --port "${port}" --dry-run
  env QSPATIAL_ENV_FILE=/dev/null \
    bash "${INFERENCE_SCRIPT}" --stage test --model "${PROFILE}" --dry-run
  env QSPATIAL_ENV_FILE=/dev/null \
    bash "${INFERENCE_SCRIPT}" --stage full --model "${PROFILE}" --dry-run
  echo "[q-spatial-78b-eval] dry-run: bash ${VALIDATION_SCRIPT} --predictions ${predictions} --report ${validation_report}"
  echo "[q-spatial-78b-eval] dry-run: bash ${SCORING_SCRIPT} --predictions ${predictions}"
  echo "[q-spatial-78b-eval] dry-run: bash ${REPORT_SCRIPT} --check"
  echo "[q-spatial-78b-eval] dry-run: bash ${REPORT_SCRIPT}"
  echo "[q-spatial-78b-eval] dry-run complete; no GPU/service/inference/scoring/report action was taken"
  exit 0
fi

port_available() {
  "${QSPATIAL_PYTHON}" - "${port}" <<'PY'
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
  "${QSPATIAL_PYTHON}" - "${control_root}/lock" <<'PY'
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

verify_existing_result_base() {
  "${QSPATIAL_PYTHON}" - "${QSPATIAL_OUTPUT_ROOT}" <<'PY'
import json
import sys

from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILE_SEQUENCE
from spatial_vlm_eval.benchmarks.q_spatial.report import discover_results

present = {item.profile for item in discover_results(sys.argv[1])}
missing = [key for key in PROFILE_SEQUENCE if key not in present]
if missing not in (["internvl3_78b"], []):
    raise SystemExit(
        "QSPATIAL_OUTPUT_ROOT must contain the existing publishable result base; "
        f"expected only internvl3_78b missing, got {missing}"
    )
print(json.dumps({
    "existing_publishable_results": len(present),
    "missing": missing,
    "policy": "append the canonical internvl3_78b result and rebuild the existing report",
}, ensure_ascii=False, sort_keys=True))
PY
}

check_readiness() {
  local command model_path output_root_resolved repo_root vllm_executable
  for command in bash flock nvidia-smi ps setsid awk tee; do
    command -v "${command}" >/dev/null 2>&1 || fail "required command is unavailable: ${command}"
  done
  if [[ "${QSPATIAL_PYTHON}" == */* ]]; then
    [[ -x "${QSPATIAL_PYTHON}" ]] || fail "QSPATIAL_PYTHON is unavailable: ${QSPATIAL_PYTHON}"
  else
    command -v "${QSPATIAL_PYTHON}" >/dev/null 2>&1 || \
      fail "QSPATIAL_PYTHON command is unavailable: ${QSPATIAL_PYTHON}"
  fi
  vllm_executable="${QSPATIAL_VLLM:-vllm}"
  if [[ "${vllm_executable}" == */* ]]; then
    [[ -x "${vllm_executable}" ]] || fail "QSPATIAL_VLLM is unavailable: ${vllm_executable}"
  else
    command -v "${vllm_executable}" >/dev/null 2>&1 || \
      fail "QSPATIAL_VLLM command is unavailable: ${vllm_executable}"
  fi
  [[ -f "${MODEL_PROBE}" ]] || fail "model readiness probe is unavailable: ${MODEL_PROBE}"
  [[ -f "${GPU_PREFLIGHT}" ]] || fail "GPU preflight is unavailable: ${GPU_PREFLIGHT}"
  [[ -n "${QSPATIAL_PARQUET_ROOT:-}" ]] || fail "set QSPATIAL_PARQUET_ROOT"
  [[ -d "${QSPATIAL_PARQUET_ROOT}" ]] || \
    fail "QSPATIAL_PARQUET_ROOT is unavailable: ${QSPATIAL_PARQUET_ROOT}"
  [[ -n "${QSPATIAL_SCANNET_RGB_ROOT:-}" ]] || fail "set QSPATIAL_SCANNET_RGB_ROOT"
  [[ -d "${QSPATIAL_SCANNET_RGB_ROOT}" ]] || \
    fail "QSPATIAL_SCANNET_RGB_ROOT is unavailable: ${QSPATIAL_SCANNET_RGB_ROOT}"
  model_path="${INTERNVL3_78B_MODEL:-}"
  [[ -n "${model_path}" ]] || fail "set INTERNVL3_78B_MODEL"
  [[ -e "${model_path}" ]] || fail "INTERNVL3_78B_MODEL is unavailable: ${model_path}"
  [[ -d "${QSPATIAL_OUTPUT_ROOT}" ]] || \
    fail "synchronize the existing QSPATIAL_OUTPUT_ROOT before this supplemental run"
  [[ -w "${QSPATIAL_OUTPUT_ROOT}" ]] || fail "QSPATIAL_OUTPUT_ROOT is not writable"
  repo_root="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
  output_root_resolved="$("${QSPATIAL_PYTHON}" - "${QSPATIAL_OUTPUT_ROOT}" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).resolve())
PY
)"
  case "${output_root_resolved}/" in
    "${repo_root}/"*) fail "QSPATIAL_OUTPUT_ROOT must stay outside the repository" ;;
  esac
  verify_existing_result_base
  if ! port_available; then
    blocked "port ${port} is occupied; existing service was left untouched"
  fi
  if ! check_lock_available; then
    blocked "another InternVL3-78B evaluation holds ${control_root}/lock"
  fi
  MIN_FREE_GPU_MIB=76000 MIN_GPU_COUNT=4 REQUIRE_IDLE_GPU=1 \
    bash "${GPU_PREFLIGHT}"
  "${QSPATIAL_PYTHON}" - <<'PY'
import json

from spatial_vlm_eval.benchmarks.q_spatial.inference import inspect_local_gpus
from spatial_vlm_eval.benchmarks.q_spatial.profiles import get_profile

print(json.dumps(inspect_local_gpus(get_profile("internvl3_78b"), "vllm"), sort_keys=True))
PY
  env QSPATIAL_ENV_FILE=/dev/null \
    bash "${SERVE_SCRIPT}" --model "${PROFILE}" --gpu-ids "${gpu_ids}" --port "${port}" --dry-run
  env QSPATIAL_ENV_FILE=/dev/null \
    bash "${INFERENCE_SCRIPT}" --check --model "${PROFILE}"
}

check_readiness
if [[ "${operation}" == "--check" ]]; then
  echo "[q-spatial-78b-eval] CHECK registration=valid"
  echo "[q-spatial-78b-eval] CHECK existing_result_base=20/21_or_21/21"
  echo "[q-spatial-78b-eval] CHECK four_gpu_preflight=passed"
  echo "[q-spatial-78b-eval] CHECK owned_endpoint=available"
  echo "[q-spatial-78b-eval] CHECK evaluation_lock=available"
  echo "[q-spatial-78b-eval] CHECK target_scoring=exact_predictions_path"
  echo "[q-spatial-78b-eval] CHECK existing_global_report=${report}"
  exit 0
fi

mkdir -p "${logs_root}"
exec 8>"${control_root}/lock"
if ! flock -n 8; then
  blocked "another InternVL3-78B evaluation holds ${control_root}/lock"
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ).$$"
controller_log="${logs_root}/${run_id}.controller.log"
exec > >(tee -a "${controller_log}") 2>&1
echo "[q-spatial-78b-eval] controller_log=${controller_log}"

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
    echo "[q-spatial-78b-eval] stopping owned ${label} process group ${pid} with TERM"
    kill -TERM -- "-${pid}" 2>/dev/null || true
  fi
  deadline=$(( $(date +%s) + stop_timeout ))
  while group_alive "${pid}" && (( $(date +%s) < deadline )); do
    sleep "${poll_seconds}"
  done
  if group_alive "${pid}"; then
    echo "[q-spatial-78b-eval] owned ${label} ignored TERM; sending KILL" >&2
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
    echo "[q-spatial-78b-eval] stopped status=${status}; rerun the same command to resume" >&2
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
  echo "[q-spatial-78b-eval] START ${label}; log=${log}"
  setsid "$@" >"${log}" 2>&1 &
  active_step_pid=$!
  set +e
  wait "${active_step_pid}"
  status=$?
  set -e
  active_step_pid=""
  if (( status != 0 )); then
    echo "[q-spatial-78b-eval] FAIL ${label} status=${status}; log=${log}" >&2
    tail -80 "${log}" >&2 || true
    return "${status}"
  fi
  echo "[q-spatial-78b-eval] PASS ${label}"
}

wait_for_model() {
  local deadline
  deadline=$(( $(date +%s) + service_timeout ))
  while true; do
    if ! group_alive "${active_service_pid}"; then
      echo "[q-spatial-78b-eval] vLLM exited before readiness; log=${service_log}" >&2
      return 1
    fi
    if "${QSPATIAL_PYTHON}" "${MODEL_PROBE}" \
      --base-url "${base_url}" --expected-model "${served_model}" --timeout 5; then
      echo "[q-spatial-78b-eval] ready model=${served_model} endpoint=${base_url}"
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      echo "[q-spatial-78b-eval] readiness timeout; log=${service_log}" >&2
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
      echo "[q-spatial-78b-eval] GPU release timeout; processes left untouched: ${pids//$'\n'/,}" >&2
      return 4
    fi
    sleep "${poll_seconds}"
  done
}

prediction_is_reusable() {
  local state provenance_log precheck_log
  [[ -f "${predictions}" ]] || return 1
  provenance_log="${logs_root}/${run_id}.${PROFILE}.preexisting-provenance.log"
  if ! "${QSPATIAL_PYTHON}" - "${QSPATIAL_OUTPUT_ROOT}" >"${provenance_log}" 2>&1 <<'PY'
import json
import os
import sys
from pathlib import Path

from spatial_vlm_eval.benchmarks.q_spatial.data import QSpatialTestContract
from spatial_vlm_eval.benchmarks.q_spatial.inference import track_directory
from spatial_vlm_eval.benchmarks.q_spatial.profiles import get_profile
from spatial_vlm_eval.benchmarks.q_spatial.scheduled_batch import complete_result_errors
from spatial_vlm_eval.benchmarks.q_spatial.scorer import SCORER_PROTOCOL

root = Path(sys.argv[1]).resolve()
profile = get_profile("internvl3_78b")
contract = QSpatialTestContract(
    os.environ["QSPATIAL_PARQUET_ROOT"],
    os.environ["QSPATIAL_SCANNET_RGB_ROOT"],
)
errors = complete_result_errors(profile, contract, root)
metadata_path = track_directory(root, profile) / "predictions.jsonl.metadata.json"
try:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    metadata = {}
if metadata.get("scorer_protocol") != SCORER_PROTOCOL:
    errors.append("metadata scorer protocol is not the current protocol")
if errors:
    print("\n".join(errors))
    raise SystemExit(1)
print("current inference, scorer declaration, binding, gate, dataset, revision, and hashes passed")
PY
  then
    echo "[q-spatial-78b-eval] existing prediction does not match every current protocol/binding; inference will resume; log=${provenance_log}"
    return 1
  fi
  state="$(score_status | awk -F '\t' 'NR == 1 {print $1}')"
  case "${state}" in
    new|retry|complete) ;;
    *) return 1 ;;
  esac
  precheck_log="${logs_root}/${run_id}.${PROFILE}.preexisting-validation.log"
  if env QSPATIAL_ENV_FILE=/dev/null \
    bash "${VALIDATION_SCRIPT}" --predictions "${predictions}" \
      --report "${validation_report}" >"${precheck_log}" 2>&1; then
    echo "[q-spatial-78b-eval] SKIP inference; existing full-271 prediction revalidated"
    return 0
  fi
  echo "[q-spatial-78b-eval] existing prediction is not reusable; inference will resume; log=${precheck_log}"
  return 1
}

if ! prediction_is_reusable; then
  service_log="${logs_root}/${run_id}.${PROFILE}.vllm.log"
  echo "[q-spatial-78b-eval] START owned four-GPU vLLM; log=${service_log}"
  setsid env QSPATIAL_ENV_FILE=/dev/null \
    bash "${SERVE_SCRIPT}" --model "${PROFILE}" --gpu-ids "${gpu_ids}" --port "${port}" \
    >"${service_log}" 2>&1 &
  active_service_pid=$!
  wait_for_model
  run_step "test gate" "${logs_root}/${run_id}.${PROFILE}.test.log" \
    env QSPATIAL_ENV_FILE=/dev/null \
    bash "${INFERENCE_SCRIPT}" --stage test --model "${PROFILE}"
  run_step "full-271 inference" "${logs_root}/${run_id}.${PROFILE}.full.log" \
    env QSPATIAL_ENV_FILE=/dev/null \
    bash "${INFERENCE_SCRIPT}" --stage full --model "${PROFILE}"
  stop_owned_group "${active_service_pid}" "vLLM service"
  active_service_pid=""
  wait_for_gpu_release
fi

run_step "full validator" "${logs_root}/${run_id}.${PROFILE}.validation.log" \
  env QSPATIAL_ENV_FILE=/dev/null \
  bash "${VALIDATION_SCRIPT}" --predictions "${predictions}" --report "${validation_report}"
run_step "exact-target scoring" "${logs_root}/${run_id}.${PROFILE}.scoring.log" \
  env QSPATIAL_ENV_FILE=/dev/null \
  bash "${SCORING_SCRIPT}" --predictions "${predictions}"

final_score_state="$(score_status | awk -F '\t' 'NR == 1 {print $1}')"
if [[ "${final_score_state}" != "complete" ]]; then
  fail "target scoring did not reach complete publication gates: ${final_score_state}"
fi

run_step "report preflight" "${logs_root}/${run_id}.report-check.log" \
  env QSPATIAL_ENV_FILE=/dev/null bash "${REPORT_SCRIPT}" --check
run_step "existing global report rebuild" "${logs_root}/${run_id}.report.log" \
  env QSPATIAL_ENV_FILE=/dev/null bash "${REPORT_SCRIPT}"

"${QSPATIAL_PYTHON}" - "${QSPATIAL_OUTPUT_ROOT}" "${report}" <<'PY'
import sys
from pathlib import Path

from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILE_SEQUENCE
from spatial_vlm_eval.benchmarks.q_spatial.report import discover_results

root, report = Path(sys.argv[1]), Path(sys.argv[2])
present = {item.profile for item in discover_results(root)}
missing = [key for key in PROFILE_SEQUENCE if key not in present]
if missing or len(present) != len(PROFILE_SEQUENCE):
    raise SystemExit(f"global report source is incomplete: present={len(present)}/21 missing={missing}")
if not report.is_file():
    raise SystemExit(f"global report was not created: {report}")
text = report.read_text(encoding="utf-8")
if "全轨完整度：21/21" not in text or "缺失 profile：无" not in text:
    raise SystemExit("global report does not declare 21/21 completeness with no missing profile")
PY

echo "[q-spatial-78b-eval] COMPLETE profile=${PROFILE} score_state=complete report=21/21:${report}"
