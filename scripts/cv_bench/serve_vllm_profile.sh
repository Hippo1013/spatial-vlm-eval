#!/usr/bin/env bash
# Serve one locked CV-Bench general-open profile endpoint with vLLM 0.19.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cv_bench/serve_vllm_profile.sh --model PROFILE --gpu-ids IDS --port PORT [--dry-run]

Examples:
  bash scripts/cv_bench/serve_vllm_profile.sh --model qwen3_vl_8b --gpu-ids 0 --port 18101
  bash scripts/cv_bench/serve_vllm_profile.sh --model qwen3_vl_32b --gpu-ids 0,1 --port 18101
EOF
}

profile=""
gpu_ids=""
port=""
dry_run=0
while (( $# > 0 )); do
  case "$1" in
    --model)
      profile="${2:-}"
      shift 2
      ;;
    --gpu-ids)
      gpu_ids="${2:-}"
      shift 2
      ;;
    --port)
      port="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[cv-bench-vllm] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${profile}" || -z "${gpu_ids}" || -z "${port}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "${port}" =~ ^[1-9][0-9]*$ ]] || (( port > 65535 )); then
  echo "[cv-bench-vllm] --port must be an integer in 1..65535" >&2
  exit 2
fi
if [[ ! "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[cv-bench-vllm] --gpu-ids must be a comma-separated integer list" >&2
  exit 2
fi

profile_fields="$("${CVBENCH_PYTHON}" - "${profile}" <<'PY'
import sys

from spatial_vlm_eval.benchmarks.cv_bench.profiles import get_profile

profile = get_profile(sys.argv[1])
if profile.group != "general_open" or profile.default_backend != "vllm":
    raise SystemExit(f"Profile {profile.key} is not a CV-Bench general-open vLLM track")
trust_remote_code = "1" if profile.family == "internvl3" else "0"
print(
    "\t".join(
        [
            profile.model_path_env,
            profile.revision,
            profile.served_model_name or "",
            str(profile.default_tensor_parallel_size),
            str(profile.min_free_gpu_mib),
            trust_remote_code,
        ]
    )
)
PY
)"
IFS=$'\t' read -r model_path_env revision served_model tensor_parallel min_free trust_remote_code \
  <<<"${profile_fields}"
model_path="${!model_path_env:-}"
if [[ -z "${model_path}" ]]; then
  echo "[cv-bench-vllm] set ${model_path_env} in ${CVBENCH_ENV_FILE}" >&2
  exit 2
fi

IFS=',' read -r -a selected_gpus <<<"${gpu_ids}"
if (( ${#selected_gpus[@]} != tensor_parallel )); then
  echo "[cv-bench-vllm] ${profile} requires exactly ${tensor_parallel} selected GPU(s)" >&2
  exit 2
fi

args=(
  serve "${model_path}"
  --revision "${revision}"
  --served-model-name "${served_model}"
  --host "127.0.0.1"
  --port "${port}"
  --tensor-parallel-size "${tensor_parallel}"
  --dtype "bfloat16"
  --max-model-len "${CVBENCH_VLLM_MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization "${CVBENCH_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
  --seed 42
  --limit-mm-per-prompt.image 1
)
if [[ "${trust_remote_code}" == "1" ]]; then
  args+=(--trust-remote-code)
fi

vllm_executable="${CVBENCH_VLLM:-vllm}"
if [[ "${dry_run}" == "1" ]]; then
  printf '[cv-bench-vllm] dry-run:'
  printf ' %q' env "CUDA_VISIBLE_DEVICES=${gpu_ids}" "${vllm_executable}" "${args[@]}"
  printf '\n'
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${gpu_ids}"
MIN_FREE_GPU_MIB="${min_free}" \
MIN_GPU_COUNT="${tensor_parallel}" \
  "${SCRIPT_DIR}/../msmu/gpu_preflight.sh"
exec "${vllm_executable}" "${args[@]}"
