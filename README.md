# Spatial VLM Evaluation

用于可复现地评测通用与空间专用视觉语言模型的多 benchmark 工作区。仓库当前实现 MSMU-Bench
official test 987 条、CV-Bench locked test 2638 条、Q-Spatial Bench locked test 271 条和 SPBench-SI
single-image test 1009 条的受限输入
合同、可恢复推理、严格 validator、独立 scorer protocol 与发布门禁。模型适配与 benchmark 评分分层，
任何模型都不能收到答案、reference、类型标签、来源或其他协议禁止字段。

## 当前能力

- Benchmark：MSMU-Bench official `test`（987）、CV-Bench locked `test_2d+test_3d`（2638）与
  Q-Spatial ScanNet + Q-Spatial++（170+101）、SPBench-SI single-image `test`（1009）。
- 推理：benchmark-owned 输入、CV-Bench 23 轨、Q-Spatial/SPBench-SI 各 21 轨、单图/template 审计、fsync journal、
  断点恢复和原子输出。
- 验收：subset/full 强制隔离；CV-Bench、Q-Spatial 与 SPBench-SI test gate 各自绑定完整 provenance。
- 评分：MSMU macro-8、CV-Bench 2D/3D/Overall、Q-Spatial split-macro `δ≤2`、SPBench-SI 原始严格
  MRA 四题型宏平均使用彼此独立的 scorer
  protocol；评分与报告均按目录发现并强制 publication gates。
- 结果性质：MSMU 是 official-compatible internal score；CV-Bench、Q-Spatial 与 SPBench-SI 主分是
  official/original formula + robust parser internal score，均不冒充上游实现的逐字节复刻。

## 当前评测范围

项目目标覆盖 MSMU-Bench、CV-Bench、Q-Spatial Bench 和 SPBench-SI。MSMU 的既有 18 条目标 profile
已完成；CV-Bench 已实现 23 条目标轨的链路。截至 2026-08-06，除需四张 80GB GPU 的
InternVL3-78B 外，其余 22 条轨均已完成 full-2638、正式 validator、当前 scorer protocol 评分和
publication gates，全局报告状态为 22/23。Q-Spatial Bench 的 21 轨代码、协议、回归与运行入口已经
实现；截至 2026-08-07，除固定 TP=4 blocked 的 InternVL3-78B 外，其余 20 轨均已在服务器通过
red/blue canary + smoke8 当前 test gate、full-271、正式 validator、完整 provenance、当前 v2 scorer
评分与 publication gates；全局报告状态为 20/21。SPBench-SI 的 21 轨 contract、
两阶段 gate、双 scorer、调度与报告已于 2026-08-07 完成本地实现和回归；截至 2026-08-08，20 条
非 78B 轨中 18 条保留当前 full-1009；Gemini full 失败，SpatialLadder 旧 v1 full 因官方 left-padding
要求未落实而作废，等待 v2 test/full 重跑。尚未启动正式评分，也没有全局报告。精确范围、
数据准备边界与当前阶段见
[四 Benchmark 评测范围](docs/evaluation-scope.md)。

项目级目标范围现为 19 个模型身份：MSMU 阶段已有 15 个，加上 RoboBrain2.5-8B-NV、
RoboBrain2.5-8B-MT、HiSpatial-3B 和 SpatialLadder-3B。CV-Bench registry 将 fair/native、额外提示词
和不同 checkpoint 拆成 23 条独立轨；模型身份、输入公平性和已验证状态只在
[模型矩阵](docs/model-matrix.md)维护。
服务器当前结果状态以结果目录中的 `status.tsv`、validator、metadata 和 `summary.json` 为准，不从
README 推断。
完整文档分类与更新规则见[文档地图](docs/README.md)，语义变更见 [CHANGELOG](CHANGELOG.md)。

MSMU 正式产物以 `MANUAL_TEST_OUTPUT_ROOT` 为准；CV-Bench、Q-Spatial 与 SPBench-SI 分别以
`CVBENCH_OUTPUT_ROOT`、`QSPATIAL_OUTPUT_ROOT`、`SPBENCH_SI_OUTPUT_ROOT` 为准。它们都必须位于仓库外。仓库根禁止创建
`output/` 或 `outputs/`；可再生成的人工抽查和临时导出同样写入仓库外，不能作为 canonical 发布或
恢复来源。

当前服务器项目与输出分别位于
`/media/datasets/lihaoran/latent_reasoning/spatial-vlm-eval` 和
`/media/datasets/lihaoran/latent_reasoning/msmu-outputs`。新下载的数据、模型、Conda 环境和各类缓存也
统一写入 `/media/datasets/lihaoran/`；既有 `tangzecong` 数据、模型与环境不迁移，由 `.env.server`
中的 legacy 路径继续显式引用。完整目录表见[推理手册的服务器存储约定](docs/msmu-inference.md#1-运行前边界)。
CV-Bench 既有数据与旧模型继续从 `/media/datasets/tangzecong/huggingface/` 只读引用；新增下载写入
`/media/datasets/lihaoran/huggingface/`，详情见
[评测范围的数据与模型位置](docs/evaluation-scope.md#服务器数据与模型位置)。
服务器命令需要显式出站代理时，使用仓库外的本机 Mihomo 服务；首次配置、tmux 启停、按 shell 开关
和出口验证见[服务器网络代理手册](docs/server-network-proxy.md)。

人工测试从[三阶段入口](docs/msmu-all-model-test-commands.md)开始；每个阶段均有统一入口脚本，会自动
加载 `.env.server`。阶段三默认范围与 Qwen3 补测顺序以
[`run_stage3_serial_inference.sh`](scripts/msmu/run_stage3_serial_inference.sh)的 `--list` /
`--qwen3 --list` 及
[阶段三 runbook](docs/msmu-stage3-full-eval.md)为准。

## 仓库结构

```text
src/spatial_vlm_eval/
├── benchmarks/msmu/          # 987 条数据合同、validator、judge/scorer
├── benchmarks/cv_bench/      # 2638 条合同、23-profile registry、推理/评分/报告
├── benchmarks/q_spatial/     # 271 条合同、21-profile registry、numeric scorer/报告
├── benchmarks/spbench_si/    # 1009 条合同、21-profile registry、MRA + upstream audit
├── orchestration/            # 跨 benchmark 控制器；不承载 benchmark 评分逻辑
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
scripts/cv_bench/             # 两阶段推理、目录驱动评分和报告入口
scripts/q_spatial/            # Q-Spatial test/full、评分、报告与 vLLM 入口
scripts/spbench_si/            # SPBench-SI test/full、双卡调度、双 scorer 与报告入口
scripts/internvl3_78b/         # 三 benchmark 共用一次四卡 78B vLLM 的补测入口
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

### InternVL3-78B 三 Benchmark 一键测评

四卡服务器可让 Q-Spatial、SPBench-SI、CV-Bench 共用一次 78B 模型加载。入口默认自动加载仓库根
`.env.server`；首次使用时按[公共配置模板](configs/internvl3-78b-three-bench.env.example)补齐配置。
三个 benchmark 都会独立完成 78B 评分；只有该 benchmark 的既有报告源仅缺 78B 时才重建全局报告，
其他模型结果不完整只会跳过对应汇总，不阻塞 78B 测评。

```bash
cd /media/datasets/lihaoran/latent_reasoning/spatial-vlm-eval
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --dry-run
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --status
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --check
```

正式运行：

```bash
bash scripts/internvl3_78b/run_three_bench_evaluation.sh
```

完整指令见[三 Benchmark 一键测评](docs/internvl3-78b-three-bench-evaluation.md)。

### CV-Bench

将 [CV-Bench 配置模板](configs/cv-bench-server.env.example)合并到未跟踪 `.env.server`，再先列出并测试
单轨：

```bash
bash scripts/cv_bench/run_inference.sh --list
bash scripts/cv_bench/run_inference.sh --stage test --model qwen3_vl_8b
bash scripts/cv_bench/run_inference.sh --stage full --model qwen3_vl_8b
bash scripts/cv_bench/score_results.sh --predictions /absolute/path/to/predictions.jsonl
bash scripts/cv_bench/build_results_report.sh
```

full 必须使用当前绑定的 test gate；服务器 endpoint、专用 runner、GPU、付费 API 和分阶段验收见
[CV-Bench 简明运行指令](docs/cv-bench-commands.md)与
[两阶段 runbook](docs/cv-bench-two-stage-runbook.md)。

四卡 InternVL3-78B 可用一条命令完成 test gate、full-2638、独立校验、精确单轨评分并重建同一个
全局报告：

```bash
bash scripts/cv_bench/run_internvl3_78b_evaluation.sh
```

运行前检查、tmux 与恢复方式见
[CV-Bench InternVL3-78B 一键评测](docs/cv-bench-internvl3-78b-evaluation.md)。

### Q-Spatial Bench

将 [Q-Spatial 配置模板](configs/q-spatial-server.env.example)合并到未跟踪 `.env.server`。先 test，只有
当前绑定 gate 通过后才能 full：

```bash
bash scripts/q_spatial/run_inference.sh --list
bash scripts/q_spatial/run_inference.sh --stage test --model qwen3_vl_8b
bash scripts/q_spatial/run_inference.sh --stage full --model qwen3_vl_8b
bash scripts/q_spatial/run_scheduled_batch.sh --list
bash scripts/q_spatial/run_scheduled_batch.sh --check
bash scripts/q_spatial/run_scheduled_batch.sh --dry-run
bash scripts/q_spatial/run_scheduled_batch.sh --stage test --dry-run
bash scripts/q_spatial/score_results.sh --predictions /absolute/path/to/predictions.jsonl
bash scripts/q_spatial/build_results_report.sh
```

双卡正式批次另需显式传入 `--without-internvl78 --with-paid-api`；`--stage test` 只建立或复用 gate，
默认 `--stage full` 才继续 full/validator，控制器不自动评分。两根数据合同、纯色 canary、smoke8、
20 轨分阶段 lane 与付费 API 边界见
[Q-Spatial 简明指令](docs/q-spatial-commands.md)和
[两阶段 runbook](docs/q-spatial-two-stage-runbook.md)。API 实跑必须另行明确批准。

四卡补齐 InternVL3-78B 时，先同步现有 20/21 输出根，再用一条独立命令完成当前 test gate、
full-271、validator、精确单轨评分和原有全局报告 21/21 重建：

```bash
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --check
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --faq
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh
```

正式产物仍写入原 `QSPATIAL_OUTPUT_ROOT`，报告仍是原 `q-spatial-result.md`；迁移与恢复说明见
[Q-Spatial InternVL3-78B 四卡补测](docs/q-spatial-internvl3-78b-evaluation.md)。

### SPBench-SI

将 [SPBench-SI 配置模板](configs/spbench-si-server.env.example)合并到未跟踪 `.env.server`。只读检查、
单轨两阶段和目录评分入口如下：

```bash
bash scripts/spbench_si/run_inference.sh --list
bash scripts/spbench_si/run_inference.sh --stage test --model qwen3_vl_8b
bash scripts/spbench_si/run_inference.sh --stage full --model qwen3_vl_8b
bash scripts/spbench_si/run_scheduled_batch.sh --list
bash scripts/spbench_si/run_scheduled_batch.sh --check
bash scripts/spbench_si/run_scheduled_batch.sh --dry-run
bash scripts/spbench_si/score_results.sh --check
bash scripts/spbench_si/score_results.sh --predictions /absolute/path/to/predictions.jsonl
bash scripts/spbench_si/build_results_report.sh

# 四卡 InternVL3-78B：只读检查后执行 test/full/validator/精确评分/报告重建
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --check
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh
```

正式双卡批次必须显式传入 `--without-internvl78 --with-paid-api`；控制器不自动评分。GPU test/full、
付费 API 与正式评分分别需要后续明确授权。数据合同、20/21 暂行报告与 TP=4 边界见
[SPBench-SI 简明指令](docs/spbench-si-commands.md)和
[两阶段 runbook](docs/spbench-si-two-stage-runbook.md)。迁移到四卡服务器补齐 78B 时使用
[InternVL3-78B 四卡完整评测](docs/spbench-si-internvl3-78b-evaluation.md)；入口不会接管已有端口或
GPU 进程。

### MSMU

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

MSMU 每行 `predictions.jsonl` 只有：

```text
index, raw_type, task_family, question, reference, prediction
```

前五项由 benchmark-owned test row 重新附着，模型只产生 `prediction`。逐样本 journal 保存 index、
清洗后题干、图片尺寸与像素 SHA-256、profile/protocol 和 generation metadata，不保存 API key 或图片
base64。只有所有目标 index 成功后才原子生成排序 JSONL；网络错误不能伪装成空答案，模型真实返回
空文本则保留并告警。

CV-Bench、Q-Spatial 与 SPBench-SI prediction 每行都严格只有 `index, raw_prediction`，但属于独立
validator 和 scorer protocol：CV-Bench 的 answer/task/source、Q-Spatial 的 answer/unit/split/type、
SPBench-SI 的 ground truth/question type 都只在评分时由各自数据合同重新关联。四套 benchmark schema
与 scorer 不互换。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

协议细节见 [MSMU canonical protocol](docs/benchmarks/msmu/protocol.md)、
[CV-Bench canonical protocol](docs/benchmarks/cv_bench/protocol.md)与
[Q-Spatial canonical protocol](docs/benchmarks/q_spatial/protocol.md)、
[SPBench-SI canonical protocol](docs/benchmarks/spbench_si/protocol.md)，分层边界见
[架构说明](docs/architecture.md)。协作者从[文档地图](docs/README.md)选择任务相关材料；coding agent
修改前必须阅读 [AGENTS.md](AGENTS.md) 并按其中的触发路由执行。
