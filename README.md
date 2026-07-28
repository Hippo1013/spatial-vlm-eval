# Spatial VLM Evaluation

用于可复现地评测通用与空间专用视觉语言模型的多 benchmark 工作区。仓库当前实现
MSMU-Bench official test 987 条的受限输入合同、统一可恢复推理、严格 prediction validator 和本地
judge v3 scorer。模型适配与 benchmark 评分分层，任何模型都不能收到 reference、类型标签、其他 QA
或同图历史。

## 当前状态

- Benchmark：MSMU-Bench official `test` split（987 条）。
- Qwen：Qwen2.5-VL 7B base/PEFT、32B、72B deterministic profile；72B 在两张 A800 上均衡加载。
- 新增适配：GPT-5、Gemini 3.1 Pro、2 个 LLaVA-NeXT、3 个 InternVL3，以及 SSR、
  SpatialRGPT、3DThinker、SpatialBot；连同 3 个 Qwen 参数量共注册 17 个 inference profile。
- 主指标：八个 official type accuracy 的非加权平均 `official_macro8_accuracy`。
- scorer protocol 保持为
  `sdvlm_official_compat_local_judge_v3_grounding_split_strict_quant_length`。
- 当前分数性质：official-compatible internal score，不是 GPT-4-Turbo strict official score。
- adapter、contract/mock 回归、静态 processor/provenance 验证和运行编排已完成；各模型的 live
  阶段状态以模型矩阵和固定输出目录中的 validator 为准，不能把“adapter available”写成
  “evaluation completed”。

完整 profile、权重 revision 和部署状态见 [模型矩阵](docs/model-matrix.md)，命令与服务器验收顺序见
[MSMU 多模型推理手册](docs/msmu-inference.md)。人工测试从[三阶段入口](docs/msmu-all-model-test-commands.md)
开始，依次执行[阶段一](docs/msmu-stage1-canary.md)、[阶段二](docs/msmu-stage2-smoke8.md)和
[阶段三](docs/msmu-stage3-full-eval.md)。每个阶段均有统一模型入口脚本，会自动加载 `.env.server`；
无需逐行复制各 adapter 的环境变量和 pipeline 命令。

## 仓库结构

```text
src/spatial_vlm_eval/
├── benchmarks/msmu/          # 数据所有权、validator、smoke selector、v3 scorer
└── models/
    ├── common/               # 输入审计、journal、resume、原子 finalization
    ├── openai_compatible/    # OpenRouter/OpenAI/Google/vLLM
    ├── qwen25_vl/
    ├── ssr/
    ├── spatialrgpt/
    ├── three_d_thinker/
    └── spatialbot/
scripts/msmu/                 # 环境、GPU preflight、服务与 pipeline 编排
tests/                        # 协议不变量和 bug 回归
docs/                         # canonical 协议、模型矩阵、部署说明与来源记录
```

模型、dataset、checkpoint、API key、prediction、judge cache、论文 PDF 和服务器环境 manifest 均不
进入 Git。

## 环境准备

安装前必须先查看已有 Conda 环境：

```bash
conda env list
```

本地非 CUDA 开发环境只需 contract、API mock 和测试依赖：

```bash
conda create -n spatial-vlm-eval-dev python=3.10 -y
conda activate spatial-vlm-eval-dev
python -m pip install -r requirements/local-dev.txt
python -m pip install -e . --no-deps
```

GPU 推理栈不能在 macOS 或默认 Python 中盲装。服务器应先复用已有 `latent` 与 `vllm019`，SSR、
SpatialRGPT、3DThinker、SpatialBot 使用互相隔离的环境；PyTorch 安装源必须依据服务器 CUDA/驱动
选择。环境完成后保存但不提交；Conda 环境保存 explicit manifest，基于现有解释器的隔离 overlay
则保存 base interpreter/build 说明：

```bash
mkdir -p environment-manifests
conda list --explicit > environment-manifests/ENV-NAME.explicit.txt
python -m pip freeze > environment-manifests/ENV-NAME.pip-freeze.txt
```

## 最短运行路径

复制模板并从未跟踪文件加载：

```bash
cp configs/msmu-server.env.example .env.server
set -a
source .env.server
set +a
```

模板中的 family-specific interpreter 会被相应脚本自动选用；显式设置 `PYTHON=...` 仍可覆盖。

选择覆盖八类的调试 subset：

```bash
INDICES="$(DATASET_ROOT="$DATASET_ROOT" bash scripts/msmu/select_smoke_indices.sh)"
export INDICES
```

以 vLLM LLaVA 7B 为例，先做 processor 静态检查、服务启动和非 MSMU 视觉 canary：

```bash
PROFILE=llava_next_mistral_7b MODEL_PATH="$MODEL_ROOT/llava-v1.6-mistral-7b-hf" \
  bash scripts/msmu/preflight_vllm_processor.sh

PROFILE=llava_next_mistral_7b MODEL_PATH="$MODEL_ROOT/llava-v1.6-mistral-7b-hf" \
  bash scripts/msmu/serve_llava_next.sh

PROFILE=llava_next_mistral_7b SERVED_MODEL_NAME=llava-next-mistral-7b-msmu \
  bash scripts/msmu/canary_vllm_vision.sh
```

另一个 shell 运行 subset inference + subset validator：

```bash
PROFILE=llava_next_mistral_7b \
BACKEND=vllm \
SERVED_MODEL_NAME=llava-next-mistral-7b-msmu \
INDICES="$INDICES" \
  bash scripts/msmu/run_openai_compatible_pipeline.sh
```

指定 `INDICES` 或 `LIMIT` 时 pipeline 会启用 validator 的 `--allow-subset`，并拒绝评分。正式运行必须
先 `unset INDICES LIMIT`，完整生成 `0..986`，validator 通过后才能设置 `RUN_SCORE=1`；此时还必须
显式提供与推理服务不同的 `JUDGE_BASE_URL`。

## 严格输出合同

每行 `predictions.jsonl` 始终只有：

```text
index, raw_type, task_family, question, reference, prediction
```

前五项由 benchmark-owned test row 重新附着，模型只产生 `prediction`。逐样本 journal 保存 index、
清洗后题干、图片尺寸与像素 SHA-256、profile/protocol 和 generation metadata，不保存 API key 或图片
base64。只有所有目标 index 成功后才原子生成排序 JSONL；网络错误不能伪装成空答案，模型真实返回
空文本则保留并告警。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

协议细节见 [MSMU canonical protocol](docs/benchmarks/msmu/protocol.md)，分层边界见
[架构说明](docs/architecture.md)。协作者和 coding agent 修改前必须阅读 [AGENTS.md](AGENTS.md)。
