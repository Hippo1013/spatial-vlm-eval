#!/usr/bin/env bash
# Serve the local text-only MSMU judge through vLLM's OpenAI-compatible API.

set -euo pipefail

: "${JUDGE_MODEL:?Set JUDGE_MODEL to a local instruction-model path}"

VLLM="${VLLM:-vllm}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-msmu-judge}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "${VLLM}" serve "${JUDGE_MODEL}" \
  --served-model-name "${JUDGE_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE:-bfloat16}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
