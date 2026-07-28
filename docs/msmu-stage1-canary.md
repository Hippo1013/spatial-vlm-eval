# MSMU 阶段一：接口与图像链路检查

## 目标

本阶段只回答一个问题：模型接口能否正确加载并接收图片？它不产生 benchmark 分数。

固定输出目录：

```text
/media/datasets/tangzecong/latent_reasoning/msmu-outputs/manual-three-stage-v1/01_canary/
```

每个新终端先执行一行准备命令，然后只选择当前要测试的模型：

```bash
source /media/datasets/tangzecong/latent_reasoning/spatial-vlm-eval/scripts/msmu/prepare_manual_test.sh
```

## A. LLaVA-NeXT 与 InternVL3

这类模型要做三件事：processor preflight、启动 vLLM、红/蓝合成图 canary。启动服务的命令会持续
占用终端 A；在另一个终端 B 中执行 canary。终端 B 也先执行同一条准备命令。

### LLaVA-NeXT Mistral 7B

终端 A：

```bash
CHECK_DIR="${OUTPUT_ROOT}/01_canary/llava-next-mistral-7b-vllm"
mkdir -p "${CHECK_DIR}"
PROFILE=llava_next_mistral_7b MODEL_PATH="${LLAVA_MISTRAL_7B_MODEL}" \
PREFLIGHT_REPORT="${CHECK_DIR}/processor_preflight.json" \
  bash scripts/msmu/preflight_vllm_processor.sh

PROFILE=llava_next_mistral_7b MODEL_PATH="${LLAVA_MISTRAL_7B_MODEL}" \
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/msmu/serve_llava_next.sh 2>&1 | tee "${CHECK_DIR}/vllm_serve.log"
```

终端 B：

```bash
CHECK_DIR="${OUTPUT_ROOT}/01_canary/llava-next-mistral-7b-vllm"
PROFILE=llava_next_mistral_7b SERVED_MODEL_NAME=llava-next-mistral-7b-msmu \
INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
CANARY_REPORT="${CHECK_DIR}/vision_canary.json" \
  bash scripts/msmu/canary_vllm_vision.sh
```

### LLaVA-NeXT Yi 34B

终端 A：

```bash
CHECK_DIR="${OUTPUT_ROOT}/01_canary/llava-next-yi-34b-vllm"
mkdir -p "${CHECK_DIR}"
PROFILE=llava_next_yi_34b MODEL_PATH="${LLAVA_YI_34B_MODEL}" \
PREFLIGHT_REPORT="${CHECK_DIR}/processor_preflight.json" \
  bash scripts/msmu/preflight_vllm_processor.sh

PROFILE=llava_next_yi_34b MODEL_PATH="${LLAVA_YI_34B_MODEL}" \
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/msmu/serve_llava_next.sh 2>&1 | tee "${CHECK_DIR}/vllm_serve.log"
```

终端 B：

```bash
CHECK_DIR="${OUTPUT_ROOT}/01_canary/llava-next-yi-34b-vllm"
PROFILE=llava_next_yi_34b SERVED_MODEL_NAME=llava-next-yi-34b-msmu \
INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
CANARY_REPORT="${CHECK_DIR}/vision_canary.json" \
  bash scripts/msmu/canary_vllm_vision.sh
```

### InternVL3 8B

终端 A：

```bash
CHECK_DIR="${OUTPUT_ROOT}/01_canary/internvl3-8b-vllm"
mkdir -p "${CHECK_DIR}"
PROFILE=internvl3_8b MODEL_PATH="${INTERNVL3_8B_MODEL}" \
PREFLIGHT_REPORT="${CHECK_DIR}/processor_preflight.json" \
  bash scripts/msmu/preflight_vllm_processor.sh

PROFILE=internvl3_8b MODEL_PATH="${INTERNVL3_8B_MODEL}" CUDA_VISIBLE_DEVICES=0 \
  bash scripts/msmu/serve_internvl3.sh 2>&1 | tee "${CHECK_DIR}/vllm_serve.log"
```

终端 B：

```bash
CHECK_DIR="${OUTPUT_ROOT}/01_canary/internvl3-8b-vllm"
PROFILE=internvl3_8b SERVED_MODEL_NAME=internvl3-8b-msmu \
INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
CANARY_REPORT="${CHECK_DIR}/vision_canary.json" \
  bash scripts/msmu/canary_vllm_vision.sh
```

### InternVL3 38B

终端 A：

```bash
CHECK_DIR="${OUTPUT_ROOT}/01_canary/internvl3-38b-vllm"
mkdir -p "${CHECK_DIR}"
PROFILE=internvl3_38b MODEL_PATH="${INTERNVL3_38B_MODEL}" \
PREFLIGHT_REPORT="${CHECK_DIR}/processor_preflight.json" \
  bash scripts/msmu/preflight_vllm_processor.sh

PROFILE=internvl3_38b MODEL_PATH="${INTERNVL3_38B_MODEL}" CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/msmu/serve_internvl3.sh 2>&1 | tee "${CHECK_DIR}/vllm_serve.log"
```

终端 B：

```bash
CHECK_DIR="${OUTPUT_ROOT}/01_canary/internvl3-38b-vllm"
PROFILE=internvl3_38b SERVED_MODEL_NAME=internvl3-38b-msmu \
INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
CANARY_REPORT="${CHECK_DIR}/vision_canary.json" \
  bash scripts/msmu/canary_vllm_vision.sh
```

### InternVL3 78B：仅静态检查

```bash
CHECK_DIR="${OUTPUT_ROOT}/01_canary/internvl3-78b-vllm"
mkdir -p "${CHECK_DIR}"
PROFILE=internvl3_78b MODEL_PATH="${INTERNVL3_78B_MODEL}" \
PREFLIGHT_REPORT="${CHECK_DIR}/processor_preflight.json" \
  bash scripts/msmu/preflight_vllm_processor.sh

PROFILE=internvl3_78b MODEL_PATH="${INTERNVL3_78B_MODEL}" DRY_RUN=1 \
  bash scripts/msmu/serve_internvl3.sh | tee "${CHECK_DIR}/vllm_config_dry_run.log"
```

不要为 78B 启动服务或继续后续阶段。

## B. GPT-5 与 Gemini API

API 没有本地 processor。本阶段发送 2 条真实 MSMU 图像请求，验证 key、provider、图片计数和
generation metadata。默认使用 OpenRouter：

先从未跟踪配置加载 key，并检查它确实存在：

```bash
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in the untracked environment}"
```

GPT-5：

```bash
env -u INDICES -u INFERENCE_BASE_URL \
LIMIT=2 RUN_NAME="01_canary/gpt5-openrouter" RUN_SCORE=0 \
PROFILE=gpt5 BACKEND=openrouter \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

Gemini 3.1 Pro：

```bash
env -u INDICES -u INFERENCE_BASE_URL \
LIMIT=2 RUN_NAME="01_canary/gemini31pro-openrouter" RUN_SCORE=0 \
PROFILE=gemini31pro BACKEND=openrouter \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

## C. Qwen 与空间专用模型

这些 adapter 当前没有独立合成图入口。本阶段用 `LIMIT=1` 验证环境、权重、processor 和一次真实
generation。所有结果仍是 debug subset。

### Qwen2.5-VL base

```bash
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" MIN_FREE_GPU_MIB=30000 \
  bash scripts/msmu/gpu_preflight.sh

env -u INDICES \
LIMIT=1 RUN_NAME="01_canary/qwen25-vl-base" RUN_SCORE=0 \
BASE_MODEL="${QWEN_BASE_MODEL}" BASE_MODEL_REVISION="${QWEN_BASE_REVISION}" \
  bash scripts/msmu/run_qwen_peft_pipeline.sh
```

### Qwen2.5-VL PEFT

```bash
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" MIN_FREE_GPU_MIB=30000 \
  bash scripts/msmu/gpu_preflight.sh

env -u INDICES \
LIMIT=1 RUN_NAME="01_canary/qwen25-vl-peft" RUN_SCORE=0 \
BASE_MODEL="${QWEN_BASE_MODEL}" BASE_MODEL_REVISION="${QWEN_BASE_REVISION}" \
CHECKPOINT="${QWEN_PEFT_CHECKPOINT}" CHECKPOINT_REVISION="${QWEN_PEFT_REVISION:-}" \
  bash scripts/msmu/run_qwen_peft_pipeline.sh
```

### SSR fair

```bash
env -u INDICES \
LIMIT=1 RUN_NAME="01_canary/ssr-rgb-only" RUN_SCORE=0 PROFILE=ssr \
  bash scripts/msmu/run_ssr_pipeline.sh
```

### SSR native

```bash
env -u INDICES \
LIMIT=1 RUN_NAME="01_canary/ssr-native" RUN_SCORE=0 PROFILE=ssr_native \
  bash scripts/msmu/run_ssr_pipeline.sh
```

### SpatialRGPT

```bash
env -u INDICES \
LIMIT=1 RUN_NAME="01_canary/spatialrgpt-rgb-only" RUN_SCORE=0 \
MODEL_PATH="${SPATIALRGPT_MODEL}" \
  bash scripts/msmu/run_spatialrgpt_pipeline.sh
```

### 3DThinker fair

```bash
env -u INDICES \
LIMIT=1 RUN_NAME="01_canary/3dthinker-fair" RUN_SCORE=0 PROFILE=3dthinker \
MODEL_PATH="${THREEDTHINKER_MODEL}" \
  bash scripts/msmu/run_3dthinker_pipeline.sh
```

### 3DThinker native

```bash
env -u INDICES \
LIMIT=1 RUN_NAME="01_canary/3dthinker-native" RUN_SCORE=0 PROFILE=3dthinker_native \
MODEL_PATH="${THREEDTHINKER_MODEL}" \
  bash scripts/msmu/run_3dthinker_pipeline.sh
```

### SpatialBot fair

```bash
env -u INDICES \
LIMIT=1 RUN_NAME="01_canary/spatialbot-rgb-only" RUN_SCORE=0 PROFILE=spatialbot \
MODEL_PATH="${SPATIALBOT_MODEL}" \
  bash scripts/msmu/run_spatialbot_pipeline.sh
```

### SpatialBot native

```bash
env -u INDICES \
LIMIT=1 RUN_NAME="01_canary/spatialbot-native" RUN_SCORE=0 PROFILE=spatialbot_native \
MODEL_PATH="${SPATIALBOT_MODEL}" \
  bash scripts/msmu/run_spatialbot_pipeline.sh
```

SSR native 还需要 `.env.server` 中的 MIDI/DepthPro/CLIP/SigLIP/Mamba/Qwen 组件；SpatialBot 需要已
接受 gated license，native 轨还需要 ZoeDepth。

## 通过标准

- LLaVA/InternVL：`processor_preflight.json` 与 `vision_canary.json` 都包含 `"passed": true`；
- API/直接加载：深层结果目录中存在 `predictions.jsonl`、metadata 和
  `prediction_validation.json`，validator 为 subset pass；
- 失败时停止当前模型，不进入阶段二。

完成后继续：[阶段二：八类 8 条小量测试](msmu-stage2-smoke8.md)。
