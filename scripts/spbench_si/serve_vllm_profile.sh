#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

usage() {
  echo "Usage: bash scripts/spbench_si/serve_vllm_profile.sh --model PROFILE --gpu-ids IDS --port PORT [--dry-run]"
}

profile=""
gpu_ids=""
port=""
dry_run=0
while (( $# > 0 )); do
  case "$1" in
    --model) profile="${2:-}"; shift 2 ;;
    --gpu-ids) gpu_ids="${2:-}"; shift 2 ;;
    --port) port="${2:-}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[spbench-si-vllm] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "${profile}" || -z "${gpu_ids}" || -z "${port}" ]]; then usage >&2; exit 2; fi
if [[ ! "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || [[ ! "${port}" =~ ^[1-9][0-9]*$ ]] || (( port > 65535 )); then
  echo "[spbench-si-vllm] invalid --gpu-ids or --port" >&2; exit 2
fi

profile_fields="$("${SPBENCH_SI_PYTHON}" - "${profile}" <<'PY'
import sys
from spatial_vlm_eval.benchmarks.spbench_si.profiles import get_profile
p = get_profile(sys.argv[1])
if p.group != "general_open" or p.default_backend != "vllm":
    raise SystemExit(f"Profile {p.key} is not an SPBench-SI vLLM track")
print("\t".join([
    p.model_path_env, p.revision, p.served_model_name or "",
    str(p.default_tensor_parallel_size), str(p.min_free_gpu_mib),
    "1" if p.family == "internvl3" else "0", str(p.decoding.get("seed", 42)),
]))
PY
)"
IFS=$'\t' read -r model_path_env revision served_model tensor_parallel min_free trust_remote_code seed <<<"${profile_fields}"
model_path="${!model_path_env:-}"
if [[ -z "${model_path}" ]]; then echo "[spbench-si-vllm] set ${model_path_env}" >&2; exit 2; fi
IFS=',' read -r -a selected_gpus <<<"${gpu_ids}"
if (( ${#selected_gpus[@]} != tensor_parallel )); then echo "[spbench-si-vllm] ${profile} requires TP=${tensor_parallel}" >&2; exit 2; fi

args=(
  serve "${model_path}" --revision "${revision}" --served-model-name "${served_model}"
  --host 127.0.0.1 --port "${port}" --tensor-parallel-size "${tensor_parallel}"
  --dtype bfloat16 --max-model-len "${SPBENCH_SI_VLLM_MAX_MODEL_LEN:-32768}"
  --gpu-memory-utilization "${SPBENCH_SI_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
  --seed "${seed}" --limit-mm-per-prompt.image 1
)
if [[ "${trust_remote_code}" == "1" ]]; then args+=(--trust-remote-code); fi
vllm_executable="${SPBENCH_SI_VLLM:-vllm}"
if [[ "${dry_run}" == "1" ]]; then printf '[spbench-si-vllm] dry-run:'; printf ' %q' env "CUDA_VISIBLE_DEVICES=${gpu_ids}" "${vllm_executable}" "${args[@]}"; printf '\n'; exit 0; fi
if [[ -z "${SPBENCH_SI_VLLM_RUNTIME_VERSION:-}" ]]; then
  echo "[spbench-si-vllm] set SPBENCH_SI_VLLM_RUNTIME_VERSION to the audited exact 0.19.x version" >&2
  exit 2
fi
runtime_output="$("${vllm_executable}" --version 2>&1)" || {
  echo "[spbench-si-vllm] cannot read vLLM version" >&2
  exit 2
}
runtime_version="${runtime_output##* }"
if [[ ! "${runtime_version}" =~ ^0\.19([.]|$) ]] || [[ "${runtime_version}" != "${SPBENCH_SI_VLLM_RUNTIME_VERSION}" ]]; then
  echo "[spbench-si-vllm] vLLM version mismatch: expected ${SPBENCH_SI_VLLM_RUNTIME_VERSION}, got ${runtime_version}" >&2
  exit 2
fi
if ! "${SPBENCH_SI_PYTHON}" - "${port}" <<'PY'
import socket, sys
s = socket.socket()
try: s.bind(("127.0.0.1", int(sys.argv[1])))
finally: s.close()
PY
then echo "[spbench-si-vllm] port ${port} is occupied; existing process left untouched" >&2; exit 4; fi
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
MIN_FREE_GPU_MIB="${min_free}" MIN_GPU_COUNT="${tensor_parallel}" "${SCRIPT_DIR}/../msmu/gpu_preflight.sh"
exec "${vllm_executable}" "${args[@]}"
