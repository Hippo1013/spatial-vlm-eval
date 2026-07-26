# Spatial VLM Evaluation

用于可复现地评测通用与空间专用视觉语言模型的多 benchmark 工作区。仓库当前完整实现
MSMU-Bench 的 Qwen2.5-VL/PEFT 推理、预测校验和本地 judge v3 评分；后续 benchmark 与模型
通过独立适配器扩展，不复制或改写公共评测逻辑。

## 当前状态

- Benchmark：MSMU-Bench official test split（987 条）。
- 模型适配器：Qwen2.5-VL base model 与 PEFT adapter。
- 主要指标：八个 official type accuracy 的非加权平均 `official_macro8_accuracy`。
- 当前协议：`sdvlm_official_compat_local_judge_v3_grounding_split_strict_quant_length`。
- 结果性质：official-compatible internal score，不是 GPT-4-Turbo strict official score。
- Git 不追踪模型、数据集、checkpoint、运行输出或 `benchmark_paper/*.pdf`。

## 仓库结构

```text
.
├── src/spatial_vlm_eval/
│   ├── benchmarks/msmu/          # MSMU 数据接口、校验器、v3 scorer
│   └── models/qwen25_vl/         # Qwen2.5-VL / PEFT 推理适配器
├── scripts/msmu/                 # Ubuntu 服务器上的可执行封装
├── tests/benchmarks/msmu/        # 协议与回归测试
├── docs/
│   ├── architecture.md
│   ├── model-matrix.md
│   ├── source-provenance.json
│   └── benchmarks/msmu/protocol.md
├── requirements/                 # 已验证环境版本与开发依赖
├── benchmark_paper/              # 本地论文目录；PDF 不入 Git
├── .env.example
├── DEVLOG.md                      # 服务器实验报错与解决方法（精简）
└── AGENTS.md
```

## 环境准备

目标运行环境是带 NVIDIA GPU 的 Ubuntu 服务器。先检查已有 Conda 环境；如果已有满足依赖的
环境，应直接复用，不要仅因系统 Python 缺包就在默认解释器中安装依赖。

服务器上已验证的推理环境：

```text
Python 3.10.20
PyTorch 2.5.1+cu124
Transformers 5.8.1
PEFT 0.19.1
Datasets 4.8.5
NumPy 2.2.6
TQDM 4.67.3
```

本地 judge 使用独立 vLLM 环境，已验证版本为 `vLLM 0.19.0`。确认环境满足依赖后，可安装仓库
本身而不改动依赖：

```bash
python -m pip install -e . --no-deps
```

## 快速开始：Qwen2.5-VL / PEFT × MSMU

复制配置模板并填写服务器绝对路径：

```bash
cp .env.example .env
set -a
source .env
set +a
```

### 1. 启动本地 judge

judge 应单独占用一张 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 \
JUDGE_MODEL=/absolute/path/to/Qwen2.5-14B-Instruct \
VLLM=/absolute/path/to/vllm \
  bash scripts/msmu/serve_local_judge.sh
```

确认接口：

```bash
curl -s http://127.0.0.1:18080/v1/models
```

### 2. 推理、校验和评分

PEFT checkpoint：

```bash
BASE_MODEL=/absolute/path/to/Qwen2.5-VL-7B-Instruct \
CHECKPOINT=/absolute/path/to/adapter \
DATASET_ROOT=/absolute/path/to/MSMU \
OUTPUT=/absolute/path/to/run/predictions.jsonl \
RUN_SCORE=1 \
SCORE_OUTPUT_DIR=/absolute/path/to/run/local_judge_official_compat_v3_strict_quant_length \
  bash scripts/msmu/run_qwen_peft_pipeline.sh
```

评估 base model 时不要设置 `CHECKPOINT`：

```bash
unset CHECKPOINT
```

也可分别运行：

```bash
bash scripts/msmu/infer_qwen_peft.sh

PREDICTIONS="$OUTPUT" \
REPORT="$(dirname "$OUTPUT")/prediction_validation.json" \
  bash scripts/msmu/validate_predictions.sh

PREDICTIONS="$OUTPUT" \
OUTPUT_DIR="$SCORE_OUTPUT_DIR" \
  bash scripts/msmu/score_predictions.sh
```

`msmu-score` 在任何 judge 请求前会再次强制校验完整 test split。空 prediction 会作为明确 warning
保留并继续评分；prediction 必须使用精确六字段 schema，任何额外字段、缺失/重复 index、subset
或错误数据源都会中止评分。`official_type` 仅由 scorer 在校验通过后从 dataset-owned `raw_type`
派生，不接受 prediction 文件声明。

## 运行产物

一次正式运行至少保留：

```text
predictions.jsonl
predictions.jsonl.metadata.json
prediction_validation.json
infer.log
local_judge_official_compat_v3_strict_quant_length/
├── prediction_validation.json
├── judge_cache.jsonl
├── judge_failures.jsonl
├── scored_rows.jsonl
├── summary.json
└── score.log
```

正式报告至少读取：

- `publishable == true` 且 `publication_gate_failures == []`
- `num_samples == 987`
- `missing_official_types == []`
- `num_judge_failures == 0`
- `official_macro8_accuracy`
- `micro_accuracy`
- `quantitative_match_success_rate`
- `judge_model`、`judge_base_url` 与 `protocol`

judge 的 HTTP、解析或响应 schema 失败不会写入成功 cache，并会在下次运行时自动重试。只要仍有
未解决失败，scorer 就以非零状态退出，并写出 `publishable == false`、不含正式指标的诊断
`summary.json` 以及 `judge_failures.jsonl`；该目录不得进入正式结果表。

## 测试

在满足依赖的环境中：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

测试覆盖多数值严格长度、空 prediction warning、字段污染阻断，以及 scorer 的强制校验门禁。

## 扩展新模型或 benchmark

- 新模型放入 `src/spatial_vlm_eval/models/<model_family>/`，只负责模型原生输入与 generation。
- 新 benchmark 放入 `src/spatial_vlm_eval/benchmarks/<benchmark>/`，拥有自己的 schema、validator 和 scorer。
- shell 入口放入 `scripts/<benchmark>/`，只做路径、进程和日志编排。
- 不得把 reference、类型标签或同图历史泄露给被测模型，除非 benchmark 明确要求。
- 新协议或语义变化必须更换 protocol id、补充测试，并在结果表中单列 protocol。

详细边界见 [架构说明](docs/architecture.md) 与
[MSMU 协议](docs/benchmarks/msmu/protocol.md)。协作者与自动化 agent 必须先阅读
[AGENTS.md](AGENTS.md)。服务器实验中已解决的运行问题统一精简记录在
[DEVLOG.md](DEVLOG.md)。
