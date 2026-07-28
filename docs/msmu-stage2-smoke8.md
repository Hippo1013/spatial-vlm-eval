# MSMU 阶段二：八类 8 条小量测试

## 目标

从 MSMU 的八个 official type 中各选一条，共运行 8 条，验证真实数据的端到端流水线。本阶段禁止
评分，结果不可发布。

固定输出目录：

```text
/media/datasets/tangzecong/latent_reasoning/msmu-outputs/manual-three-stage-v1/02_smoke8/
```

只有阶段一通过的模型才能执行本阶段。

## 1. 准备固定的 8 个 index

每个新终端先运行准备脚本，再生成固定 index：

```bash
source /media/datasets/tangzecong/latent_reasoning/spatial-vlm-eval/scripts/msmu/prepare_manual_test.sh

export MSMU_SMOKE_INDICES="$(
  DATASET_ROOT="${DATASET_ROOT}" \
  SMOKE_INDEX_REPORT="${OUTPUT_ROOT}/02_smoke8/selected_indices.json" \
    bash scripts/msmu/select_smoke_indices.sh
)"
printf 'smoke indices: %s\n' "${MSMU_SMOKE_INDICES}"
```

不要手工改这 8 个 index。下面只选择当前模型的一条命令执行。

## 2. API 模型

GPT-5：

```bash
env -u LIMIT -u INFERENCE_BASE_URL \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/gpt5-openrouter" RUN_SCORE=0 \
PROFILE=gpt5 BACKEND=openrouter \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

Gemini 3.1 Pro：

```bash
env -u LIMIT -u INFERENCE_BASE_URL \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/gemini31pro-openrouter" RUN_SCORE=0 \
PROFILE=gemini31pro BACKEND=openrouter \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

## 3. vLLM 模型

先保持阶段一对应的 vLLM 服务运行。如果已经停止，重新执行阶段一中该模型的启动服务命令。

LLaVA-NeXT Mistral 7B：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/llava-next-mistral-7b-vllm" RUN_SCORE=0 \
PROFILE=llava_next_mistral_7b BACKEND=vllm \
SERVED_MODEL_NAME=llava-next-mistral-7b-msmu INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

LLaVA-NeXT Yi 34B：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/llava-next-yi-34b-vllm" RUN_SCORE=0 \
PROFILE=llava_next_yi_34b BACKEND=vllm \
SERVED_MODEL_NAME=llava-next-yi-34b-msmu INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

InternVL3 8B：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/internvl3-8b-vllm" RUN_SCORE=0 \
PROFILE=internvl3_8b BACKEND=vllm \
SERVED_MODEL_NAME=internvl3-8b-msmu INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

InternVL3 38B：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/internvl3-38b-vllm" RUN_SCORE=0 \
PROFILE=internvl3_38b BACKEND=vllm \
SERVED_MODEL_NAME=internvl3-38b-msmu INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

InternVL3 78B（`internvl3_78b`）当前禁止执行本阶段。

## 4. Qwen2.5-VL

Qwen wrapper 不内置 GPU preflight，先执行：

```bash
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" MIN_FREE_GPU_MIB=30000 \
  bash scripts/msmu/gpu_preflight.sh
```

Qwen base：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/qwen25-vl-base" RUN_SCORE=0 \
BASE_MODEL="${QWEN_BASE_MODEL}" BASE_MODEL_REVISION="${QWEN_BASE_REVISION}" \
  bash scripts/msmu/run_qwen_peft_pipeline.sh
```

Qwen PEFT：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/qwen25-vl-peft" RUN_SCORE=0 \
BASE_MODEL="${QWEN_BASE_MODEL}" BASE_MODEL_REVISION="${QWEN_BASE_REVISION}" \
CHECKPOINT="${QWEN_PEFT_CHECKPOINT}" CHECKPOINT_REVISION="${QWEN_PEFT_REVISION:-}" \
  bash scripts/msmu/run_qwen_peft_pipeline.sh
```

多个 PEFT checkpoint 要在 slug 中加入唯一 checkpoint 名。

## 5. 空间专用模型

SSR fair：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/ssr-rgb-only" RUN_SCORE=0 PROFILE=ssr \
  bash scripts/msmu/run_ssr_pipeline.sh
```

SSR native：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/ssr-native" RUN_SCORE=0 PROFILE=ssr_native \
  bash scripts/msmu/run_ssr_pipeline.sh
```

SpatialRGPT：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/spatialrgpt-rgb-only" RUN_SCORE=0 \
MODEL_PATH="${SPATIALRGPT_MODEL}" \
  bash scripts/msmu/run_spatialrgpt_pipeline.sh
```

3DThinker fair：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/3dthinker-fair" RUN_SCORE=0 \
PROFILE=3dthinker MODEL_PATH="${THREEDTHINKER_MODEL}" \
  bash scripts/msmu/run_3dthinker_pipeline.sh
```

3DThinker native：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/3dthinker-native" RUN_SCORE=0 \
PROFILE=3dthinker_native MODEL_PATH="${THREEDTHINKER_MODEL}" \
  bash scripts/msmu/run_3dthinker_pipeline.sh
```

SpatialBot fair：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/spatialbot-rgb-only" RUN_SCORE=0 \
PROFILE=spatialbot MODEL_PATH="${SPATIALBOT_MODEL}" \
  bash scripts/msmu/run_spatialbot_pipeline.sh
```

SpatialBot native：

```bash
env -u LIMIT \
INDICES="${MSMU_SMOKE_INDICES}" RUN_NAME="02_smoke8/spatialbot-native" RUN_SCORE=0 \
PROFILE=spatialbot_native MODEL_PATH="${SPATIALBOT_MODEL}" \
  bash scripts/msmu/run_spatialbot_pipeline.sh
```

## 6. 通过标准

在当前模型的深层目录中检查：

- `predictions.jsonl` 恰好 8 行；
- `prediction_validation.json` 中 `passed: true`、`allow_subset: true`、
  `num_prediction_rows: 8`、`num_unique_indices: 8`；
- `predictions.jsonl.metadata.json` 中 `publishable_inference: false`；
- 没有正式 `summary.json`，因为 subset 禁止评分。

查看当前阶段产物：

```bash
find "${OUTPUT_ROOT}/02_smoke8" -type f \
  \( -name 'prediction_validation.json' -o -name 'predictions.jsonl.metadata.json' \) \
  -print | sort
```

完成后继续：[阶段三：完整 987 条推理与评分](msmu-stage3-full-eval.md)。
