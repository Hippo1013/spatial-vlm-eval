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

7B/8B 默认 TP=1/GPU 0；34B/38B 默认 TP=2/GPU `0,1`；InternVL3-78B 固定 TP=4/GPU
`0,1,2,3`。服务固定 revision、served name、BF16、`--limit-mm-per-prompt.image 1`。78B 启动前还会
同时枚举选中的 GPU 与 `nvidia-smi` 物理 GPU，任一数量少于四张即拒绝；四张卡也必须逐卡通过空闲
和显存 preflight。底层启动命令为：

```bash
PROFILE=internvl3_78b \
MODEL_PATH="$MODEL_ROOT/InternVL3-78B-hf" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/msmu/serve_internvl3.sh
```

日常人工测试优先使用三阶段统一入口，由入口自动设置同样的四卡配置。2×80GB 仍不允许加载其约
147GB BF16 权重；`TENSOR_PARALLEL_SIZE` 也不能覆盖为 4 以外的值。

服务 ready 后必须先用非 MSMU 组合图确认视觉输入真的被读取：固定 512×512 抗锯齿白底图的左上角
为红圆、右下角为蓝方块；图像由 4× 超采样后 LANCZOS 缩小确定性生成，模型必须同时答对颜色、
形状和位置：

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

三阶段入口的 stage 1 会在两个真实 MSMU 样本前自动运行同一组合视觉 canary，写入当前模型
`01_canary/RUN_SLUG/vision_canary.json`。该 canary 只做 1 次 generation，不使用 MSMU 数据、不进入
journal 或评分；语义答案、OpenRouter 首方 provider/model 或 `num_media_prompt==1` 任一不合法都会
fail closed，且不会继续真实样本。单独只跑 canary 时使用：

```bash
PROFILE=gpt5_openrouter_non_zdr BACKEND=openrouter \
API_KEY_ENV=OPENROUTER_API_KEY CANARY_REPORT=/absolute/path/to/vision_canary.json \
  bash scripts/msmu/canary_openai_compatible_vision.sh
```

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
prediction success。请求 alias `openai/gpt-5` / `google/gemini-3.1-pro-preview` 分别锁定返回的
catalog canonical revision `openai/gpt-5-2025-08-07` /
`google/gemini-3.1-pro-preview-20260219`；其他 revision 会被拒绝。
OpenRouter 的 generation metadata 可能在 completion 返回后短暂 404；client 默认只重试同一个
generation id 10 次（指数退避、单次最多 2 秒，累计约 16 秒），不会因此重发付费 completion。
必要时可用 `OPENROUTER_METADATA_RETRIES` 同时覆盖 canary 和 inference wrapper。
API inference 首轮遍历全部目标后，若仍有网络失败留下的缺失 index，wrapper 会固定只对这些缺失项
再执行一轮；已经成功并写入 journal 的样本不会重复请求。补跑后仍缺失时保持 incomplete、拒绝生成
正式 prediction，之后重跑同一命令仍只会从 journal 续跑缺失项。

标准 `gpt5` / `gemini31pro` OpenRouter profile 还要求请求级 ZDR。若首方 endpoint 没有可用 ZDR
路由，只有在用户明确同意数据策略例外后才能改用独立 non-ZDR profile：

```bash
PROFILE=gpt5_openrouter_non_zdr BACKEND=openrouter LIMIT=2 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh

PROFILE=gemini31pro_openrouter_non_zdr BACKEND=openrouter LIMIT=2 \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

两条例外轨仍固定 OpenAI / Google AI Studio 首方 provider、`allow_fallbacks=false`、
`require_parameters=true` 和 `data_collection=deny`，只把 `zdr` 设为 false。它们使用独立 protocol、
run slug 和输出目录；三阶段统一入口分别是 `gpt5_openrouter_non_zdr` 与
`gemini31pro_openrouter_non_zdr`。两条 v3 run slug 分别是
`gpt5-openrouter-non-zdr-medium-16384-v3` 与
`gemini31pro-openrouter-non-zdr-medium-16384-v3`，不得与原 ZDR 或 low/512 v2 journal 混用。

首方 API：

```bash
export OPENAI_API_KEY='provided-out-of-band'
PROFILE=gpt5 BACKEND=openai LIMIT=2 bash scripts/msmu/run_openai_compatible_pipeline.sh

export GEMINI_API_KEY='provided-out-of-band'
PROFILE=gemini31pro BACKEND=google LIMIT=2 bash scripts/msmu/run_openai_compatible_pipeline.sh
```

GPT-5 不发送 temperature；Gemini 轨按本项目锁定为 temperature 0。标准 ZDR/direct profile 保持 low
reasoning、192 completion tokens。non-ZDR live 结果确认 completion 上限同时计算 hidden reasoning：
GPT-5 v1 的 192 和 v2 的 512 都出现过 hidden reasoning 耗尽预算的空文本，v2 还出现可见回答截断。
因此当前两条 non-ZDR v3 能力轨均锁定 medium reasoning 和 16384 total completion tokens；GPT-5
配置对齐 EASI 使用相同 `gpt-5-2025-08-07` revision 的正式空间能力评测设置。先核对两条 live smoke
的 provider、图片计数、generation metadata、可见输出与费用，再批准全量。

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

## 7. 正式 987 条、评分与报告

本手册只保留 inference/deployment 边界，不复制阶段三操作者命令。每个 profile 仍须依次通过：

1. 合成图/processor canary；
2. 八类 8 条 subset inference 与 subset validator；
3. 清除 subset 参数后的 full-987 inference；
4. metadata 中 987 targets、`publishable_inference=true` 且无 missing index；
5. 正式 validator；
6. 独立 local judge 和目录驱动评分；
7. 完整 publication gates。

已经通过 stage 1/2 的任一注册且获准 stage 3 模型，可用统一入口完成后三步：

```bash
bash scripts/msmu/run_model_evaluation.sh MODEL
```

该命令默认只运行 stage 3；不会重新运行 canary/smoke。它按共享注册信息自动区分 API、直接加载和
vLLM 模型，必要时管理被测模型服务，推理完成后释放 GPU、启动独立 judge，只评分刚解析出的精确
`predictions.jsonl`，随后重建全局 `03_full987/msmu-result.md`。新模型仍须先实现合法 adapter/profile
并接入 `run_manual_stage3.sh`；一键入口本身不猜 processor、chat template 或 revision。

执行前可用 `MODEL --check`，查看产物可用 `MODEL --status`，完整支持范围用 `--list`；
`MANUAL_DRY_RUN=1` 只解析命令和路径，不启动 GPU/API/service/scorer 或写报告。中断后重跑同一命令会
复用相同 journal/cache。入口只清理自己创建的进程组，绝不终止已有 GPU 进程或占用端口的服务。

实际名单、串行恢复、GPU 释放、InternVL3-78B 四卡补测和答案抽查只查
[阶段三 full-987 runbook](msmu-stage3-full-eval.md)；评分命令只查
[阶段三串行评分指令](msmu-stage3-scoring-commands.md)。服务器 burn 只按
[GPU burn 启停手册](server-gpu-burn-runbook.md)操作。

正式完成以 metadata、validator 和 summary 为准：987 条、八类齐全、`publishable=true`、
`num_judge_failures=0`。结果必须同时保存 inference/scorer protocol，并区分 official-compatible
internal score 与 strict official score。精简展示表只有在逐行校验 provenance、一次只选择一个 scorer
protocol，并在模型名称中标明实际输入或提示配置后才可省略 protocol 列；报告发现和筛选规则以
scorer protocol 与对应评分 runbook 为准。

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
