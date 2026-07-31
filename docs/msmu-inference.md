# MSMU 多模型推理与验收手册

本手册只覆盖 inference/deployment；canonical scoring 语义仍以
[`docs/benchmarks/msmu/protocol.md`](benchmarks/msmu/protocol.md) 为准。

## 1. 运行前边界

1. 先运行 `git status --short`，保留无关用户修改。
2. 先运行 `conda env list`，不得因默认 Python 缺包直接安装。
3. GPU/模型测试只能在已授权的 Ubuntu 服务器执行。没有 SSH 连接信息时停止在本地 contract/mock
   验收，不猜主机名或账户。
4. 每次 GPU 运行前执行 `scripts/msmu/gpu_preflight.sh`。默认同时要求显存达标、利用率不高于
   10% 且没有现存 compute process；不满足就退出，不 kill 现有进程。只有已协调独占/共用资源时，
   才可显式调整 `REQUIRE_IDLE_GPU` 或 `MAX_GPU_UTILIZATION_PERCENT`。
5. API key 只通过未跟踪环境变量提供；CLI 不接收 key 值。
6. SpatialBot gated license 未接受时停止，不绕过权限。

建议服务器路径模板在 `configs/msmu-server.env.example`。其中 repo、dataset、models 默认分别是：

```text
/media/datasets/tangzecong/latent_reasoning/spatial-vlm-eval
/media/datasets/tangzecong/huggingface/latent_reasoning/MSMU
/media/datasets/tangzecong/huggingface/models
```

这些路径只存在于配置示例，不写入 Python 源码。
模板还给每个隔离环境提供独立 interpreter 变量；脚本会按 family 自动选择，显式 `PYTHON=...`
仍具有最高优先级。

## 2. 环境隔离

先检查服务器已有环境与关键包：

```bash
conda env list
conda run -n latent python -c 'import torch, datasets; print(torch.__version__, datasets.__version__)'
conda run -n vllm019 python -c 'import vllm; print(vllm.__version__)'
```

预期分工：

| Environment | Use |
|---|---|
| `latent` | 本仓库、dataset、validator/scorer、API client |
| `vllm019` | LLaVA-NeXT / InternVL3 vLLM 0.19 serving |
| `msmu-ssr` | SSR 官方 Transformers + DepthPro/MIDI |
| `msmu-spatialrgpt` | VILA/SpatialRGPT 上游依赖 |
| `msmu-3dthinker` | 上游修改版 Transformers |
| `msmu-spatialbot` | Bunny/SpatialBot + ZoeDepth |

后四个环境依赖冲突明显，不能覆盖共享环境包。PyTorch/CUDA 安装命令必须在看过
`nvidia-smi`、driver、已有 torch build 和上游 requirements 后确定，因此本仓库不提供一个会盲装
CUDA wheel 的通用脚本。每个可用环境只接入本仓库本身：

```bash
conda run -n ENV python -m pip install -e "$REPO_ROOT" --no-deps
```

若 Conda launcher 本身损坏、但既有环境的绝对 Python 仍可运行，不要向共享环境写包，也不要把
launcher 故障误判为环境不可用。可以用该解释器创建 task-specific `uv venv
--system-site-packages` overlay；必须把 base interpreter、base torch/CUDA、overlay 状态和
`pip freeze` 保存到未跟踪的 `environment-manifests/`。上游源码与 revision 如下：

| Component | Locked commit |
|---|---|
| SSR | `52a21a14a84a98f07575721dd3200f76c11930d8` |
| SSR DepthPro fork | `edb23bbab37cfc4d3fe1048a2f126ca7c590ab64` |
| SpatialRGPT | `16715d4f1419997da18926c6ce574802d1eb3a37` |
| SpatialRGPT `s2wrapper` | `bfshi/scaling_on_scales@9c008a37540e761f53574b488979db6e49a64312` |
| 3DThinker | `c9469e01b719310b0eaecc1133317e4ecfc74d8c` |
| SpatialBot | `775ad8cf2f9251261dcd70b2639133d506ff583f` |
| ZoeDepth | `d87f17b2f5fdcb174cf4fb115491f4a6c60de152` |

若路径是 Git checkout，adapter 会 fail-closed 检查 HEAD；源码 archive 会在 metadata 中标成未验证。

## 3. 通用 debug subset

用 benchmark-owned selector 选每个 official type 的第一条，共 8 条：

```bash
export INDICES="$(DATASET_ROOT="$DATASET_ROOT" bash scripts/msmu/select_smoke_indices.sh)"
```

也可手工用 `INDICES=0,7,10-14` 或 `LIMIT=2`。这些只用于 debug；pipeline 会运行
`--allow-subset` validator，并在 `RUN_SCORE=1` 时直接拒绝。

正式 987 条前：

```bash
unset INDICES LIMIT
```

## 4. vLLM：LLaVA-NeXT 与 InternVL3

先做 processor 静态检查：

```bash
PROFILE=internvl3_8b \
MODEL_PATH="$MODEL_ROOT/OpenGVLab--InternVL3-8B-hf/snapshots/259a3b64a14623c0ec91a045cb43f7c5af5fa6af" \
PREFLIGHT_REPORT=/absolute/path/to/processor_preflight.json \
  bash scripts/msmu/preflight_vllm_processor.sh
```

报告必须满足：LLaVA 一个 `<image>`；InternVL 一个 `<IMG_CONTEXT>`；`pixel_values_numel > 0`。

服务启动示例：

```bash
PROFILE=internvl3_38b \
MODEL_PATH="$MODEL_ROOT/OpenGVLab--InternVL3-38B-hf/snapshots/b2a05c0c325235f7530d8274c313a1d01082e069" \
  bash scripts/msmu/serve_internvl3.sh
```

7B/8B 默认 TP=1/GPU 0；34B/38B 默认 TP=2/GPU `0,1`。服务固定 revision、served name、BF16、
`--limit-mm-per-prompt.image 1`。InternVL3-78B 只允许：

```bash
PROFILE=internvl3_78b MODEL_PATH=/locked/snapshot DRY_RUN=1 \
  bash scripts/msmu/serve_internvl3.sh
```

本阶段不在 2×80GB 上强行加载其约 147GB BF16 权重。

服务 ready 后必须先用非 MSMU 红/蓝图确认视觉输入真的被读取：

```bash
PROFILE=internvl3_38b \
SERVED_MODEL_NAME=internvl3-38b-msmu \
INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
CANARY_REPORT=/absolute/path/to/vision_canary.json \
  bash scripts/msmu/canary_vllm_vision.sh
```

然后运行 8 条 subset：

```bash
PROFILE=internvl3_38b \
BACKEND=vllm \
SERVED_MODEL_NAME=internvl3-38b-msmu \
INFERENCE_BASE_URL=http://127.0.0.1:18081/v1 \
INDICES="$INDICES" \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

## 5. GPT-5 与 Gemini 3.1 Pro

OpenRouter：

```bash
export OPENROUTER_API_KEY='provided-out-of-band'
unset INFERENCE_BASE_URL
PROFILE=gpt5 BACKEND=openrouter LIMIT=2 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

```bash
PROFILE=gemini31pro BACKEND=openrouter LIMIT=2 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

OpenRouter 请求固定首方 provider 并 fail-closed；每条成功 journal 包含 generation id、canonical model、
provider、upstream id、finish reason、reasoning/output tokens、cost 和 latency。metadata 查询失败不会写
prediction success。

首方 API：

```bash
export OPENAI_API_KEY='provided-out-of-band'
PROFILE=gpt5 BACKEND=openai LIMIT=2 bash scripts/msmu/run_openai_compatible_pipeline.sh

export GEMINI_API_KEY='provided-out-of-band'
PROFILE=gemini31pro BACKEND=google LIMIT=2 bash scripts/msmu/run_openai_compatible_pipeline.sh
```

GPT-5 不发送 temperature；Gemini 轨按本项目锁定为 temperature 0；两者 low reasoning、192 completion
tokens。先核对两条 live smoke 的 provider、图片计数、generation metadata 与费用，再批准全量。

## 6. Qwen 与空间专用模型

### Qwen3-VL 2B / 4B / 8B / 32B（当前补测）

四款模型复用现有 Qwen pipeline，只改变 `PROFILE`、锁定模型路径和 revision：

```bash
PROFILE=qwen3_vl_2b \
BASE_MODEL="$QWEN3_2B_MODEL" \
BASE_MODEL_REVISION="$QWEN3_2B_REVISION" \
LIMIT=1 \
  bash scripts/msmu/run_qwen_peft_pipeline.sh

PROFILE=qwen3_vl_32b \
BASE_MODEL="$QWEN3_32B_MODEL" \
BASE_MODEL_REVISION="$QWEN3_32B_REVISION" \
BATCH_SIZE=1 \
  bash scripts/msmu/run_qwen_peft_pipeline.sh
```

日常三阶段测试优先使用：

```bash
bash scripts/msmu/run_manual_stage1.sh qwen3_vl_2b
bash scripts/msmu/run_manual_stage1.sh qwen3_vl_4b
bash scripts/msmu/run_manual_stage1.sh qwen3_vl_8b
bash scripts/msmu/run_manual_stage1.sh qwen3_vl_32b
```

`.env.server` 分别提供 `QWEN3_2B_MODEL`/`QWEN3_2B_REVISION` 到
`QWEN3_32B_MODEL`/`QWEN3_32B_REVISION`。adapter 使用官方
`Qwen3VLForConditionalGeneration`、`AutoProcessor`、checkpoint 原生 structured-image chat
template 和 BF16/SDPA；要求 Transformers 至少包含原生 Qwen3-VL 支持。四条轨不发送 system
message，只输入一张 MSMU RGB 与清洗后的第一条问题。

stage 1 还会先用同一 processor/model 分别识别纯红图和纯蓝图，并写入 `vision_canary.json`。该
诊断不读取 MSMU 数据、不进入 prediction/journal，也不参与评分。

本项目固定 greedy、`num_beams=1`、192 tokens 和 pixel `16384..147456`。该像素范围按 Qwen3-VL
32-pixel spatial factor 保持 16..144 个 merged visual token，与旧 Qwen2.5-VL 的 token budget 对齐；
不是官方更大默认分辨率或推荐 sampling profile。2B/4B/8B 默认单卡 batch size 8，32B 默认单卡
batch size 1；真实 batch 上限仍须在 stage 1 后根据 A800 显存验证。

### Qwen2.5-VL 7B / 32B / 72B

手工测试统一使用：

```bash
bash scripts/msmu/run_manual_stage1.sh qwen25_vl_base  # 7B
bash scripts/msmu/run_manual_stage1.sh qwen25_vl_32b   # 单卡，batch size 1
bash scripts/msmu/run_manual_stage1.sh qwen25_vl_72b   # 双卡 balanced，batch size 1
```

`.env.server` 分别提供 `QWEN_BASE_MODEL`/`QWEN_BASE_REVISION`、
`QWEN_32B_MODEL`/`QWEN_32B_REVISION` 和 `QWEN_72B_MODEL`/`QWEN_72B_REVISION`。
72B 必须同时通过两张 GPU 的空闲检查，不允许退化成 CPU/disk offload；三种参数量使用相同的
structured image、原生 chat template、greedy/192-token/pixel 设置，但使用独立 inference protocol
和输出目录。`qwen25_vl_peft` 只对应 7B base。旧 adapter、PEFT 和历史产物继续保留用于复现，
但不属于当前 Qwen3-VL 四模型补测计划。

### SSR

公平 RGB-only：

```bash
PROFILE=ssr \
BASE_MODEL=/locked/Qwen2.5-VL-7B-Instruct \
SSR_VLM=/locked/SSR-VLM-7B/snapshots/7bcb4636f1396325f27f7fbb2f2df121128931bf \
  bash scripts/msmu/run_ssr_pipeline.sh
```

原生轨还必须设置 `SSR_MIDI`、`CLIP_MODEL`、`SIGLIP_MODEL`、`MAMBA_MODEL`、
`MIDI_LLM_MODEL`（MIDI 内部 `Qwen/Qwen2.5-7B@d1497293…`，hidden size 3584）和
`DEPTHPRO_CHECKPOINT`：

```bash
PROFILE=ssr_native bash scripts/msmu/run_ssr_pipeline.sh
```

公平轨明确不插入 TOR、不运行 MIDI/DepthPro；原生轨固定 10 TOR，depth 只来自当前 RGB。
与上游 `infer.py` 一致，VLM 看到 256×256 view，而原生轨的 CLIP 与 DepthPro 从同一张
原始分辨率 RGB 计算；不能先把辅助输入缩成 256×256。
SSR 的 Qwen2.5-VL base、CLIP、SigLIP、Mamba 与 MIDI 内部 Qwen revision 也会 fail-closed 校验。
`SSR-MIDI-7B` 的 `tor_proj` 输出维度为 3584，不能替换成 Qwen2.5-3B。
DepthPro 源码 commit 与 `depth_pro.pt` 文件分别校验；当前锁定 checkpoint SHA-256 为
`3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce`。

### SpatialRGPT

```bash
MODEL_PATH=/locked/SpatialRGPT-VILA1.5-8B/snapshots/64df7902f82b5053f5a53455095805e6de3a1f87 \
  bash scripts/msmu/run_spatialrgpt_pipeline.sh
```

只走 `llava/eval/model_vqa.py` 等价 RGB path 和 `llama_3` conversation；不传 region、mask 或 depth。

### 3DThinker

```bash
PROFILE=3dthinker \
MODEL_PATH=/locked/3DThinker-Mindcube/snapshots/69a70411605f86ec69bada0a625bb96ddee995d9 \
  bash scripts/msmu/run_3dthinker_pipeline.sh

PROFILE=3dthinker_native bash scripts/msmu/run_3dthinker_pipeline.sh
```

结果名必须写 “MindCube-trained stage-1 checkpoint”。fair 轨 192 tokens；native 轨使用上游 begin-position
mental-3D prompt 和 2048 tokens，从最后一个完整 answer tag 抽取 prediction。
公开 snapshot 把原生 Qwen chat template 保存在 `tokenizer_config.json`，没有独立的 processor
`chat_template.json`；在新版 Transformers 未自动暴露该字段时，adapter 会把同一份 tokenizer template
挂到 processor 后再调用 `apply_chat_template`，不会替换或改写模板。上游修改版 Transformers 当前
默认使用 slow image processor；adapter 显式锁定 `use_fast=False`，避免未来默认值变化造成像素轨漂移。

### SpatialBot

```bash
PROFILE=spatialbot \
MODEL_PATH=/accepted/SpatialBot-3B/snapshots/41d3b52c642058dfb087885bec0b8e37e0e67f8d \
  bash scripts/msmu/run_spatialbot_pipeline.sh

PROFILE=spatialbot_native \
ZOEDEPTH_CHECKPOINT=/local/ZoeD_M12_NK.pt \
  bash scripts/msmu/run_spatialbot_pipeline.sh
```

若 merged checkpoint 不存在，adapter 明确报告 gated 权限并退出。native 轨把 ZoeDepth 米制输出四舍五入、
截断为 uint16 毫米，再按上游三通道编码传入 RGB-D 两图 token；不使用 GT depth。

## 7. 正式 987 条与评分

每个 profile 依次执行：

1. 合成图/processor canary；
2. 八类 8 条 subset inference；
3. `prediction_validation.json` 中 `passed=true`、`allow_subset=true`；
4. 资源允许时清除 subset 参数，完整 inference；
5. 检查 metadata：987 targets、`publishable_inference=true`、无 missing index；
6. 正式 validator；
7. 单独启动 local judge，再运行目录驱动的串行评分入口。

人工执行优先使用 `run_manual_stage1.sh`、`run_manual_stage2.sh` 和
`run_manual_stage3.sh`。阶段三评分统一使用 `score_pending_results.sh`。

本轮阶段三排除 GPT-5、Gemini、Qwen PEFT、Qwen2.5-VL-72B 和 InternVL3-78B。获准本地轨的唯一
操作名单由阶段三 runbook 和串行脚本 `--list` 给出，并通过同一入口完成部署、完整推理、validator
和 GPU 释放：

```bash
MANUAL_DRY_RUN=1 bash scripts/msmu/run_stage3_serial_inference.sh
bash scripts/msmu/run_stage3_serial_inference.sh --check
bash scripts/msmu/run_stage3_serial_inference.sh
bash scripts/msmu/run_stage3_serial_inference.sh --status
```

该入口不评分。它使用独占锁、vLLM readiness timeout、逐样本 journal 活动 watchdog、每模型两次
可恢复尝试、自有进程组信号清理和 GPU 释放门禁。任何非本批次 GPU 进程或已存在的 inference
endpoint 都只会导致有界等待/退出，不会被终止。完整参数见
[阶段三文档](msmu-stage3-full-eval.md)。

上述 13 轨是默认历史计划。Qwen3-VL 2B/4B/8B/32B 用同一串行脚本的 `--qwen3` 参数补测；
评分仍由 `score_pending_results.sh` 从结果目录发现。

服务器上由本项目协作者管理的 burn 只按 [GPU burn 启停手册](server-gpu-burn-runbook.md)操作固定
pane。全部获准轨的完整 validator 通过后，可按阶段三文档在正式评分前生成一次固定样本答案抽查。

评分命令：

```bash
bash scripts/msmu/run_manual_stage3.sh judge serve

# 另一个终端
bash scripts/msmu/score_pending_results.sh --list
bash scripts/msmu/score_pending_results.sh --check
bash scripts/msmu/score_pending_results.sh
bash scripts/msmu/score_pending_results.sh --status
```

正式 summary 必须是 987 条、八类齐全、`publishable=true`、`num_judge_failures=0`。metadata、summary
与结果目录必须同时保存 `inference_protocol` 和 scorer protocol，并区分 official-compatible internal
score 与 strict official score。精简展示表可在逐行验证这些 provenance、一次只选择一个 scorer
protocol，并在模型名称中直接标明实际输入或提示配置后省略 protocol 列。完整命令见
[阶段三串行评分指令](msmu-stage3-scoring-commands.md)。

评分完成后先用 `build_results_report.sh --list` 枚举所有 scorer protocol 下的 summary，再按
metadata profile 和一个可选 scorer protocol 生成 Markdown 表。无筛选时使用当前 canonical
scorer protocol 并收录全部通过完整 publication gates 的评分；生成器不维护模型名单，也不会把
不完整 summary 静默加入结果表。表格使用中文精简列，专用模型在模型名称中直接标明 `RGB`、
`RGB + 深度估计` 或 `RGB + Mental-3D 提示词`；SpatialRGPT 保持模型原名。

## 8. 产物与故障恢复

输出目录自动包含 run/model revision/inference protocol/scorer protocol。核心文件：

```text
predictions.jsonl.journal.jsonl   # 可恢复逐样本事件
predictions.jsonl                 # 全部 target 成功后才原子出现
predictions.jsonl.metadata.json   # 完整 provenance/runtime
prediction_validation.json
predictions.infer.log
scores/<scorer-protocol>/
```

失败后使用完全相同的参数重跑即可 resume。不要编辑 journal；signature 不同、重复 success 或 target
不一致会 hard fail。若要开始另一个 protocol/run，使用新的输出目录。

## 9. 上游参考

- [vLLM 0.19 multimodal OpenAI chat example](https://docs.vllm.ai/en/v0.19.0/examples/online_serving/openai_chat_completion_client_for_multimodal/)
- [Qwen3-VL official repository](https://github.com/QwenLM/Qwen3-VL)、
  [Qwen3-VL-2B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [OpenRouter image input](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding)、
  [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)、
  [generation metadata](https://openrouter.ai/docs/api/api-reference/generations/get-generation)
- [SSR](https://github.com/yliu-cs/SSR)、
  [SpatialRGPT](https://github.com/AnjieCheng/SpatialRGPT)、
  [3DThinker](https://github.com/zhangquanchen/3DThinker)、
  [SpatialBot](https://github.com/BAAI-DCAI/SpatialBot)
