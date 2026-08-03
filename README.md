# Spatial VLM Evaluation

用于可复现地评测通用与空间专用视觉语言模型的多 benchmark 工作区。仓库当前实现
MSMU-Bench official test 987 条的受限输入合同、统一可恢复推理、严格 prediction validator 和本地
judge v4 scorer。模型适配与 benchmark 评分分层，任何模型都不能收到 reference、类型标签、其他 QA
或同图历史。

## 当前能力

- Benchmark：MSMU-Bench official `test` split（987 条）。
- 推理：多模型 fair/native profile、输入审计、fsync journal、断点恢复和原子输出。
- 验收：严格六字段 prediction validator，debug subset 与正式 full split 强制分离。
- 评分：八类非加权 `official_macro8_accuracy`，目录驱动串行评分和 publication gates。
- 结果性质：official-compatible internal score，不是 GPT-4-Turbo strict official score。

profile inventory、锁定 revision 和注明日期的已验证状态只在[模型矩阵](docs/model-matrix.md)维护；服务器
当前状态以结果目录中的 `status.tsv`、validator、metadata 和 `summary.json` 为准，不从 README 推断。
完整文档分类与更新规则见[文档地图](docs/README.md)，语义变更见 [CHANGELOG](CHANGELOG.md)。

正式推理、评分和汇总产物以 `.env.server` 配置的 `MANUAL_TEST_OUTPUT_ROOT` 为准；该路径应位于
仓库外的 `OUTPUT_ROOT`。仓库根禁止创建 `output/` 或 `outputs/`；可再生成的人工抽查和临时导出
同样写入仓库外，不能作为 canonical 发布或恢复来源。

当前服务器项目与输出分别位于
`/media/datasets/lihaoran/latent_reasoning/spatial-vlm-eval` 和
`/media/datasets/lihaoran/latent_reasoning/msmu-outputs`。新下载的数据、模型、Conda 环境和各类缓存也
统一写入 `/media/datasets/lihaoran/`；既有 `tangzecong` 数据、模型与环境不迁移，由 `.env.server`
中的 legacy 路径继续显式引用。完整目录表见[推理手册的服务器存储约定](docs/msmu-inference.md#1-运行前边界)。

人工测试从[三阶段入口](docs/msmu-all-model-test-commands.md)开始；每个阶段均有统一入口脚本，会自动
加载 `.env.server`。阶段三默认范围与 Qwen3 补测顺序以
[`run_stage3_serial_inference.sh`](scripts/msmu/run_stage3_serial_inference.sh)的 `--list` /
`--qwen3 --list` 及
[阶段三 runbook](docs/msmu-stage3-full-eval.md)为准。

## 仓库结构

```text
src/spatial_vlm_eval/
├── benchmarks/msmu/          # 数据所有权、validator、smoke selector、v4 scorer
└── models/
    ├── common/               # 输入审计、journal、resume、原子 finalization
    ├── openai_compatible/    # OpenRouter/OpenAI/Google/vLLM
    ├── qwen_vl/              # Qwen2.5-VL/Qwen3-VL 共享 Transformers 核心
    ├── qwen25_vl/
    ├── qwen3_vl/
    ├── ssr/
    ├── spatialrgpt/
    ├── three_d_thinker/
    └── spatialbot/
scripts/msmu/                 # 环境、GPU preflight、服务与 pipeline 编排
tests/                        # 协议不变量和 bug 回归
docs/                         # 文档地图、canonical 协议、runbook、ADR 与 troubleshooting
CHANGELOG.md                  # 影响结果、行为或操作方式的语义变化
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

以 vLLM LLaVA 7B 为例，先做 processor 静态检查、服务启动和非 MSMU 组合视觉 canary（白底图左上
红圆、右下蓝方块）：

```bash
PROFILE=llava_next_mistral_7b MODEL_PATH="$LLAVA_MISTRAL_7B_MODEL" \
  bash scripts/msmu/preflight_vllm_processor.sh

PROFILE=llava_next_mistral_7b MODEL_PATH="$LLAVA_MISTRAL_7B_MODEL" \
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
先 `unset INDICES LIMIT`，完整生成 `0..986` 并通过 validator。全部阶段三结果完成后，使用独立
judge 和 `scripts/msmu/score_pending_results.sh` 串行评分；命令见
[阶段三串行评分指令](docs/msmu-stage3-scoring-commands.md)。已有评分可由
`scripts/msmu/build_results_report.sh` 跨 scorer protocol 发现，并在完整 publication gates 与
metadata 检查后，为单一 scorer protocol 生成一份只含标题、输入配置说明和中文精简表格的
Markdown 报告。专用模型不使用泛化的“公平版/原生版”展示后缀，而是在模型名称中直接标明
`RGB`、`RGB + 深度估计` 或 `RGB + Mental-3D 提示词`；SpatialRGPT 保持模型原名。

已通过 stage 1/2 的单个注册模型可以用一个命令完成正式 stage 3、仅该模型评分和全局汇总：

```bash
bash scripts/msmu/run_model_evaluation.sh MODEL
```

该入口从 `run_manual_stage3.sh` 的共享注册信息解析 backend、服务身份和精确输出路径，不另维护模型
名单；因此适用于所有已注册且获准 stage 3 的 API、vLLM、Qwen 和空间专用模型。它只把本次
`predictions.jsonl` 交给目录驱动评分器，评分完成后重建全局
`03_full987/msmu-result.md`。`--check`、`--status`、`--list` 和 `MANUAL_DRY_RUN=1` 用于只读检查；
未注册模型仍须先实现并登记合法 adapter/profile。

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
[架构说明](docs/architecture.md)。协作者从[文档地图](docs/README.md)选择任务相关材料；coding agent
修改前必须阅读 [AGENTS.md](AGENTS.md) 并按其中的触发路由执行。
