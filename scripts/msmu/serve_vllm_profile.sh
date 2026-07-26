#!/usr/bin/env bash
# Serve one locked LLaVA-NeXT/InternVL profile with vLLM 0.19.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROFILE:?Set PROFILE to a LLaVA-NeXT or InternVL profile}"
: "${MODEL_PATH:?Set MODEL_PATH to the local checkpoint or cache snapshot}"

trust_remote_code=0
case "${PROFILE}" in
  llava_next_mistral_7b)
    locked_revision="2424fdd47412fccc66d91719126b420e9fbd7065"
    served="llava-next-mistral-7b-msmu"
    tp=1
    default_devices="0"
    min_free=30000
    ;;
  llava_next_yi_34b)
    locked_revision="84e4488fffae48f9da316ec31288b7c03f102ec7"
    served="llava-next-yi-34b-msmu"
    tp=2
    default_devices="0,1"
    min_free=60000
    ;;
  internvl3_8b)
    locked_revision="259a3b64a14623c0ec91a045cb43f7c5af5fa6af"
    served="internvl3-8b-msmu"
    tp=1
    default_devices="0"
    min_free=30000
    trust_remote_code=1
    ;;
  internvl3_38b)
    locked_revision="b2a05c0c325235f7530d8274c313a1d01082e069"
    served="internvl3-38b-msmu"
    tp=2
    default_devices="0,1"
    min_free=60000
    trust_remote_code=1
    ;;
  internvl3_78b)
    locked_revision="3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356"
    served="internvl3-78b-msmu"
    tp=2
    default_devices="0,1"
    min_free=76000
    trust_remote_code=1
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      echo "[vllm-serve] InternVL3-78B BF16 is not approved for forced loading on 2x80GB; use DRY_RUN=1 for config validation" >&2
      exit 4
    fi
    ;;
  *)
    echo "[vllm-serve] unsupported PROFILE=${PROFILE}" >&2
    exit 2
    ;;
esac

if [[ -n "${MODEL_REVISION:-}" && "${MODEL_REVISION}" != "${locked_revision}" ]]; then
  echo "[vllm-serve] revision mismatch: expected ${locked_revision}" >&2
  exit 2
fi

args=(
  serve "${MODEL_PATH}"
  --revision "${locked_revision}"
  --served-model-name "${SERVED_MODEL_NAME:-${served}}"
  --host "${HOST:-127.0.0.1}"
  --port "${PORT:-18081}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-${tp}}"
  --dtype "${DTYPE:-bfloat16}"
  --max-model-len "${MAX_MODEL_LEN:-4096}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --limit-mm-per-prompt.image 1
)
if [[ "${trust_remote_code}" == "1" ]]; then
  args+=(--trust-remote-code)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[vllm-serve] dry-run:'
  printf ' %q' "${VLLM:-vllm}" "${args[@]}"
  printf '\n'
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${default_devices}}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-${min_free}}" "${SCRIPT_DIR}/gpu_preflight.sh"
exec "${VLLM:-vllm}" "${args[@]}"
