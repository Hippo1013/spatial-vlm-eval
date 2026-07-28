#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${SSR_PYTHON:-python}}"
: "${PROFILE:?Set PROFILE to ssr or ssr_native}"
: "${SSR_UPSTREAM_ROOT:?Set SSR_UPSTREAM_ROOT}"
: "${BASE_MODEL:?Set BASE_MODEL to Qwen2.5-VL-7B-Instruct}"
: "${SSR_VLM:?Set SSR_VLM to the locked SSR-VLM-7B snapshot}"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
case "${PROFILE}" in
  ssr)
    MODEL_REVISION_TAG="7bcb4636f1396325f27f7fbb2f2df121128931bf"
    INFERENCE_PROTOCOL="msmu_ssr_rgb_only_v1"
    min_free=30000
    ;;
  ssr_native)
    MODEL_REVISION_TAG="7bcb4636f1396325f27f7fbb2f2df121128931bf-8ed878fa16e3e440741ed8c1fedfcfe40710258d"
    INFERENCE_PROTOCOL="msmu_ssr_native_depthpro_midi_tor10_native_v1"
    min_free=60000
    : "${SSR_MIDI:?Set SSR_MIDI for ssr_native}"
    : "${CLIP_MODEL:?Set CLIP_MODEL for ssr_native}"
    : "${SIGLIP_MODEL:?Set SIGLIP_MODEL for ssr_native}"
    : "${MAMBA_MODEL:?Set MAMBA_MODEL for ssr_native}"
    : "${MIDI_LLM_MODEL:?Set MIDI_LLM_MODEL to the local Qwen2.5-7B component for ssr_native}"
    : "${SSR_DEPTHPRO_ROOT:?Set SSR_DEPTHPRO_ROOT to the locked SSR DepthPro checkout}"
    : "${DEPTHPRO_CHECKPOINT:?Set DEPTHPRO_CHECKPOINT for ssr_native}"
    ;;
  *) echo "Unsupported SSR PROFILE=${PROFILE}" >&2; exit 2 ;;
esac
RUN_NAME="${RUN_NAME:-${PROFILE}}"
source "${SCRIPT_DIR}/_run_paths.sh"
if [[ "${RESOLVE_PATHS_ONLY:-0}" == "1" ]]; then
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
  exit 0
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-${min_free}}" "${SCRIPT_DIR}/gpu_preflight.sh"
mkdir -p "$(dirname "${OUTPUT}")"
LOG_PATH="${LOG_PATH:-${OUTPUT%.jsonl}.infer.log}"
args=(
  --profile "${PROFILE}" --upstream-root "${SSR_UPSTREAM_ROOT}"
  --base-model "${BASE_MODEL}" --base-model-revision "${BASE_MODEL_REVISION:-cc594898137f460bfe9f0759e9844b3ce807cfb5}"
  --ssr-vlm "${SSR_VLM}" --dataset-root "${DATASET_ROOT}" --output "${OUTPUT}"
  --device "${DEVICE:-cuda}" --retries "${INFERENCE_RETRIES:-0}"
)
if [[ "${PROFILE}" == "ssr_native" ]]; then
  args+=(
    --ssr-midi "${SSR_MIDI}"
    --clip-model "${CLIP_MODEL}"
    --clip-model-revision "${CLIP_MODEL_REVISION:-32bd64288804d66eefd0ccbe215aa642df71cc41}"
    --siglip-model "${SIGLIP_MODEL}"
    --siglip-model-revision "${SIGLIP_MODEL_REVISION:-9fdffc58afc957d1a03a25b10dba0329ab15c2a3}"
    --mamba-model "${MAMBA_MODEL}"
    --mamba-model-revision "${MAMBA_MODEL_REVISION:-1e76775f628fbf1350fbe4dbb3d971ba64af25a1}"
    --midi-llm-model "${MIDI_LLM_MODEL}"
    --midi-llm-model-revision "${MIDI_LLM_MODEL_REVISION:-d149729398750b98c0af14eb82c78cfe92750796}"
    --depthpro-root "${SSR_DEPTHPRO_ROOT}"
    --depthpro-checkpoint "${DEPTHPRO_CHECKPOINT}"
    --depthpro-checkpoint-sha256 "${DEPTHPRO_CHECKPOINT_SHA256:-3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce}"
    --depthpro-revision "${DEPTHPRO_REVISION:-edb23bbab37cfc4d3fe1048a2f126ca7c590ab64}"
  )
fi
if [[ -n "${INDICES:-}" ]]; then args+=(--indices "${INDICES}"); fi
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "${LIMIT}"); fi
if [[ -n "${RUN_METADATA:-}" ]]; then args+=(--metadata "${RUN_METADATA}"); fi
if [[ -n "${JOURNAL:-}" ]]; then args+=(--journal "${JOURNAL}"); fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then args+=(--no-resume); fi
echo "[msmu-ssr] profile=${PROFILE} output=${OUTPUT}" | tee "${LOG_PATH}"
set +e
"${PYTHON}" -m spatial_vlm_eval.models.ssr.infer "${args[@]}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return "${status}"; fi
exit "${status}"
