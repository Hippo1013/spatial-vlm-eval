#!/usr/bin/env bash
# Start one locked Q-Spatial vLLM endpoint after read-only GPU and port gates.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/q_spatial/serve_vllm_profile.sh --model PROFILE --gpu-ids IDS --port PORT [--dry-run]

TP=1 profiles require two independently started endpoints for the inference controller.
This command starts exactly one endpoint and never adopts or terminates an existing service.
EOF
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
    *) echo "[q-spatial-vllm] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${profile}" || -z "${gpu_ids}" || -z "${port}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "${port}" =~ ^[1-9][0-9]*$ ]] || (( port > 65535 )); then
  echo "[q-spatial-vllm] --port must be an integer in 1..65535" >&2
  exit 2
fi
if [[ ! "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[q-spatial-vllm] --gpu-ids must be a comma-separated integer list" >&2
  exit 2
fi

profile_fields="$("${QSPATIAL_PYTHON}" - "${profile}" <<'PY'
import sys
from spatial_vlm_eval.benchmarks.q_spatial.profiles import get_profile

profile = get_profile(sys.argv[1])
if profile.group != "general_open" or profile.default_backend != "vllm":
    raise SystemExit(f"Profile {profile.key} is not a Q-Spatial open-model vLLM track")
trust_remote_code = "1" if profile.family == "internvl3" else "0"
print("\t".join([
    profile.model_path_env,
    profile.revision,
    profile.served_model_name or "",
    str(profile.default_tensor_parallel_size),
    str(profile.min_free_gpu_mib),
    trust_remote_code,
    str(profile.decoding.get("seed", 42)),
]))
PY
)"
IFS=$'\t' read -r model_path_env revision served_model tensor_parallel min_free trust_remote_code seed \
  <<<"${profile_fields}"
model_path="${!model_path_env:-}"
if [[ -z "${model_path}" ]]; then
  echo "[q-spatial-vllm] set ${model_path_env} in ${QSPATIAL_ENV_FILE}" >&2
  exit 2
fi
IFS=',' read -r -a selected_gpus <<<"${gpu_ids}"
if (( ${#selected_gpus[@]} != tensor_parallel )); then
  echo "[q-spatial-vllm] ${profile} requires exactly ${tensor_parallel} selected GPU(s)" >&2
  exit 2
fi

args=(
  serve "${model_path}"
  --revision "${revision}"
  --served-model-name "${served_model}"
  --host 127.0.0.1
  --port "${port}"
  --tensor-parallel-size "${tensor_parallel}"
  --dtype bfloat16
  --max-model-len "${QSPATIAL_VLLM_MAX_MODEL_LEN:-32768}"
  --gpu-memory-utilization "${QSPATIAL_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
  --seed "${seed}"
  --limit-mm-per-prompt.image 1
)
if [[ "${trust_remote_code}" == "1" ]]; then
  args+=(--trust-remote-code)
fi
vllm_executable="${QSPATIAL_VLLM:-vllm}"
if [[ "${dry_run}" == "1" ]]; then
  printf '[q-spatial-vllm] dry-run:'
  printf ' %q' env "CUDA_VISIBLE_DEVICES=${gpu_ids}" "${vllm_executable}" "${args[@]}"
  printf '\n'
  exit 0
fi

if ! "${QSPATIAL_PYTHON}" - "${port}" <<'PY'
import socket
import sys

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    sock.close()
PY
then
  echo "[q-spatial-vllm] port ${port} is occupied; existing process left untouched" >&2
  exit 4
fi
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
MIN_FREE_GPU_MIB="${min_free}" MIN_GPU_COUNT="${tensor_parallel}" \
  "${SCRIPT_DIR}/../msmu/gpu_preflight.sh"
exec "${vllm_executable}" "${args[@]}"
