# MSMU 阶段三：完整 987 条推理与评分

## 目标

运行 official `test` split 全部 987 条，先通过完整 validator，再用独立 local judge 评分。只有本阶段
通过 publication gates 的 summary 才能进入结果表。

固定输出目录：

```text
/media/datasets/tangzecong/latent_reasoning/msmu-outputs/manual-three-stage-v1/03_full987/
```

只有阶段二通过的模型才能执行本阶段。

## 1. 开始前必须执行

```bash
source /media/datasets/tangzecong/latent_reasoning/spatial-vlm-eval/scripts/msmu/prepare_manual_test.sh

# 这是正式全量的关键：清除所有 subset 参数。
unset LIMIT INDICES MSMU_SMOKE_INDICES

# judge endpoint 必须和被测 vLLM endpoint 分开。
export JUDGE_BASE_URL=http://127.0.0.1:18080/v1
export JUDGE_MODEL_NAME=msmu-judge
```

下面命令默认 `RUN_SCORE=1`，要求 judge 已经 ready。如果 GPU 不够同时运行被测模型和 judge：

1. 先把当前模型命令中的 `RUN_SCORE=1` 改为 `RUN_SCORE=0`，完成 987 条推理与 validator；
2. 停止被测模型，启动 judge；
3. 使用完全相同的 `RUN_NAME` 重跑，改回 `RUN_SCORE=1`。runner 会 resume，不会重推 987 条。

## 2. API 模型

GPT-5：

```bash
env -u LIMIT -u INDICES -u INFERENCE_BASE_URL \
RUN_NAME="03_full987/gpt5-openrouter" RUN_SCORE=1 \
PROFILE=gpt5 BACKEND=openrouter \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

Gemini 3.1 Pro：

```bash
env -u LIMIT -u INDICES -u INFERENCE_BASE_URL \
RUN_NAME="03_full987/gemini31pro-openrouter" RUN_SCORE=1 \
PROFILE=gemini31pro BACKEND=openrouter \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

正式全量前应确认 API 预算。本文默认 OpenRouter；如果使用首方 API，必须改 backend、key 和 run slug。

## 3. vLLM 模型

先启动当前模型的 vLLM 服务。被测服务使用 `18081`，不能把它当作 `18080` 的 judge。

LLaVA-NeXT Mistral 7B：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/llava-next-mistral-7b-vllm" RUN_SCORE=1 \
PROFILE=llava_next_mistral_7b BACKEND=vllm \
SERVED_MODEL_NAME=llava-next-mistral-7b-msmu INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

LLaVA-NeXT Yi 34B：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/llava-next-yi-34b-vllm" RUN_SCORE=1 \
PROFILE=llava_next_yi_34b BACKEND=vllm \
SERVED_MODEL_NAME=llava-next-yi-34b-msmu INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

InternVL3 8B：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/internvl3-8b-vllm" RUN_SCORE=1 \
PROFILE=internvl3_8b BACKEND=vllm \
SERVED_MODEL_NAME=internvl3-8b-msmu INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

InternVL3 38B：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/internvl3-38b-vllm" RUN_SCORE=1 \
PROFILE=internvl3_38b BACKEND=vllm \
SERVED_MODEL_NAME=internvl3-38b-msmu INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

InternVL3 78B（`internvl3_78b`）当前没有获准的 full-run 命令。

## 4. Qwen2.5-VL

先做 GPU preflight：

```bash
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" MIN_FREE_GPU_MIB=30000 \
  bash scripts/msmu/gpu_preflight.sh
```

Qwen base：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/qwen25-vl-base" RUN_SCORE=1 \
BASE_MODEL="${QWEN_BASE_MODEL}" BASE_MODEL_REVISION="${QWEN_BASE_REVISION}" \
  bash scripts/msmu/run_qwen_peft_pipeline.sh
```

Qwen PEFT：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/qwen25-vl-peft" RUN_SCORE=1 \
BASE_MODEL="${QWEN_BASE_MODEL}" BASE_MODEL_REVISION="${QWEN_BASE_REVISION}" \
CHECKPOINT="${QWEN_PEFT_CHECKPOINT}" CHECKPOINT_REVISION="${QWEN_PEFT_REVISION:-}" \
  bash scripts/msmu/run_qwen_peft_pipeline.sh
```

多个 PEFT checkpoint 必须使用不同 slug。

## 5. 空间专用模型

SSR fair：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/ssr-rgb-only" RUN_SCORE=1 PROFILE=ssr \
  bash scripts/msmu/run_ssr_pipeline.sh
```

SSR native：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/ssr-native" RUN_SCORE=1 PROFILE=ssr_native \
  bash scripts/msmu/run_ssr_pipeline.sh
```

SpatialRGPT：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/spatialrgpt-rgb-only" RUN_SCORE=1 \
MODEL_PATH="${SPATIALRGPT_MODEL}" \
  bash scripts/msmu/run_spatialrgpt_pipeline.sh
```

3DThinker fair：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/3dthinker-fair" RUN_SCORE=1 \
PROFILE=3dthinker MODEL_PATH="${THREEDTHINKER_MODEL}" \
  bash scripts/msmu/run_3dthinker_pipeline.sh
```

3DThinker native：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/3dthinker-native" RUN_SCORE=1 \
PROFILE=3dthinker_native MODEL_PATH="${THREEDTHINKER_MODEL}" \
  bash scripts/msmu/run_3dthinker_pipeline.sh
```

SpatialBot fair：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/spatialbot-rgb-only" RUN_SCORE=1 \
PROFILE=spatialbot MODEL_PATH="${SPATIALBOT_MODEL}" \
  bash scripts/msmu/run_spatialbot_pipeline.sh
```

SpatialBot native：

```bash
env -u LIMIT -u INDICES \
RUN_NAME="03_full987/spatialbot-native" RUN_SCORE=1 \
PROFILE=spatialbot_native MODEL_PATH="${SPATIALBOT_MODEL}" \
  bash scripts/msmu/run_spatialbot_pipeline.sh
```

## 6. 正式通过标准

在当前模型的深层目录中确认：

- `predictions.jsonl` 恰好 987 行，index 精确覆盖 `0..986`；
- `predictions.jsonl.metadata.json` 中 `num_predictions: 987`、dataset `num_targets: 987`、
  `publishable_inference: true`；
- `prediction_validation.json` 中 `passed: true`、`allow_subset: false`、
  `num_prediction_rows: 987`、`num_unique_indices: 987`；
- `scores/<scorer-protocol>/summary.json` 中 `num_samples: 987`、八类齐全、
  `num_judge_failures: 0`、`publishable: true`。

查看正式产物：

```bash
find "${OUTPUT_ROOT}/03_full987" -type f \
  \( -name 'prediction_validation.json' -o -name 'predictions.jsonl.metadata.json' \
     -o -name 'summary.json' \) \
  -print | sort
```

需要我检查时，只需提供 SSH 连接方式；不要发送 API key。
