# MSMU 阶段一：接口与图像链路检查

## 目标

确认模型能正确加载并收到图片。本阶段不产生 benchmark 分数，固定写入：

```text
/media/datasets/tangzecong/latent_reasoning/msmu-outputs/manual-three-stage-v1/01_canary/
```

脚本会自动加载 `.env.server`。先查看可用模型名：

```bash
bash scripts/msmu/run_manual_stage1.sh --list
```

## LLaVA-NeXT 与 InternVL3

这四个模型需要两个终端：

```text
llava_next_mistral_7b
llava_next_yi_34b
internvl3_8b
internvl3_38b
```

终端 A 启动 processor preflight 和 vLLM 服务：

```bash
bash scripts/msmu/run_manual_stage1.sh MODEL serve
```

看到服务 ready 后，终端 B 运行红/蓝合成图 canary：

```bash
bash scripts/msmu/run_manual_stage1.sh MODEL check
```

例如：

```bash
# 终端 A
bash scripts/msmu/run_manual_stage1.sh llava_next_mistral_7b serve

# 终端 B
bash scripts/msmu/run_manual_stage1.sh llava_next_mistral_7b check
```

脚本默认给 7B/8B 使用 GPU `0`，给 34B/38B 使用 GPU `0,1`。只有在已经协调好其他 GPU 时才覆盖：

```bash
MANUAL_CUDA_VISIBLE_DEVICES=2,3 bash scripts/msmu/run_manual_stage1.sh internvl3_38b serve
```

InternVL3 78B 只执行 processor 与 vLLM 配置静态检查，不会启动服务：

```bash
bash scripts/msmu/run_manual_stage1.sh internvl3_78b check
```

脚本禁止对 78B 使用 `serve`。

## API、Qwen 与空间专用模型

这些模型只需要一个终端，省略 action 时默认执行 `run`：

```bash
bash scripts/msmu/run_manual_stage1.sh MODEL
```

例如：

```bash
bash scripts/msmu/run_manual_stage1.sh qwen25_vl_base
bash scripts/msmu/run_manual_stage1.sh qwen25_vl_32b
bash scripts/msmu/run_manual_stage1.sh qwen25_vl_72b
bash scripts/msmu/run_manual_stage1.sh qwen3_vl_2b
bash scripts/msmu/run_manual_stage1.sh qwen3_vl_4b
bash scripts/msmu/run_manual_stage1.sh qwen3_vl_8b
bash scripts/msmu/run_manual_stage1.sh qwen3_vl_32b
bash scripts/msmu/run_manual_stage1.sh ssr_native
bash scripts/msmu/run_manual_stage1.sh spatialbot_native
```

`qwen25_vl_base` 是 7B。32B 默认使用 GPU 0，72B 默认使用 GPU `0,1` 并做 balanced 加载；
两种大模型固定 batch size 1。直接加载的本地模型自动运行 GPU preflight，并生成 1 条真实结果。
当前 Qwen3-VL 四条补测轨均默认使用 GPU 0，其中 32B 固定 batch size 1。stage 1 必须同时核对
`vision_canary.json` 中红/蓝图语义检查通过，以及 metadata 中无 system message、恰好一张 RGB、
pixel `16384..147456` 和 `num_model_image_tensors: 1`。
GPT-5/Gemini 默认通过 OpenRouter
生成 2 条结果，运行前先在当前终端导出 key：

```bash
export OPENROUTER_API_KEY='provided-out-of-band'
bash scripts/msmu/run_manual_stage1.sh gpt5
bash scripts/msmu/run_manual_stage1.sh gemini31pro
```

如改用首方 API：

```bash
MANUAL_API_BACKEND=openai bash scripts/msmu/run_manual_stage1.sh gpt5
MANUAL_API_BACKEND=google bash scripts/msmu/run_manual_stage1.sh gemini31pro
```

对应 key 分别是 `OPENAI_API_KEY` 和 `GEMINI_API_KEY`。

## 推荐 tmux 名称

三个阶段共用 session `msmu`。vLLM 模型使用两个窗口，例如：

```text
llava-m7b-srv
llava-m7b-check
```

其他模型使用一个窗口；旧 Qwen 窗口为 `12-qwen-base`、`21-qwen32b`、`22-qwen72b`，Qwen3
补测建议用 `23-qwen3-2b`、`24-qwen3-4b`、`25-qwen3-8b`、`26-qwen3-32b`。

## 通过标准

- LLaVA/InternVL：当前模型目录中的 `processor_preflight.json` 与 `vision_canary.json` 都包含
  `"passed": true`；
- API/直接加载：深层结果目录中存在 `predictions.jsonl`、metadata 和
  `prediction_validation.json`，validator 为 subset pass；
- Qwen：同目录 `vision_canary.json` 还必须包含 `"passed": true`；
- 失败时停止当前模型，不进入阶段二。

完成后继续：[阶段二：八类 8 条小量测试](msmu-stage2-smoke8.md)。
