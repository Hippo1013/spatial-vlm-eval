# Spatial VLM Evaluation

用于可复现地评测通用与空间专用视觉语言模型的多 benchmark 工作区。仓库已经覆盖
MSMU-Bench、CV-Bench、Q-Spatial Bench 与 SPBench-SI，并为每个 benchmark 独立维护输入合同、
prediction validator、scorer protocol、结果报告和 publication gates。

## 新 agent 快速接手

按以下顺序阅读，通常可以在五分钟内建立项目全貌：

1. [AGENTS.md](AGENTS.md)：安全边界、协议红线和按任务读取路由；
2. [文档地图](docs/README.md)：每类知识的唯一事实源；
3. [评测范围与进度](docs/evaluation-scope.md)：四个 benchmark 的日期化进度快照与下一步；
4. [模型矩阵](docs/model-matrix.md)：目标模型、profile、revision、输入轨和 decoding；
5. 当前任务命中的 canonical protocol、runbook、测试和服务器产物。

进度结论不能只引用本文或记忆。汇报前必须现场检查对应输出根的 `status.tsv`、test gate、
`prediction_validation.json`、metadata、`summary.json`、publication gates 和结果报告。

## 项目内容

| Benchmark | 正式范围 | 主结果 | 目标轨 |
|---|---:|---|---:|
| MSMU-Bench | official `test` 987 条 | official-compatible local-judge macro-8 | 18 条当前目标 profile |
| CV-Bench | locked 2D 1438 + 3D 1200 | 2D / 3D / Overall | 23 |
| Q-Spatial Bench | ScanNet 170 + Q-Spatial++ 101 | split-macro inclusive `delta <= 2` | 21 |
| SPBench-SI | single-image `test` 1009 条 | strict original MRA 四题型宏平均 | 21 |

共同能力包括 benchmark-owned 防泄漏输入、单图/processor/template 审计、fsync journal、断点恢复、
原子输出、subset/full 隔离、目录驱动评分和可追溯报告。结果性质与上游实现的偏差以各 benchmark 的
[canonical protocol](docs/README.md#规则架构与协议)为准，不能从表格标题推断。

项目级范围为 19 个模型身份；同一模型的 RGB、公平轨、派生 depth/XYZ 或额外提示词轨必须使用独立
profile、inference protocol 和输出目录。闭源 API 轨只作补充参照，不是阶段收尾门槛；未完成轨不会
自动触发付费调用。

## 核心边界

- adapter 只能看到对应图片和 benchmark 允许的 prompt，不能收到 answer/reference、类型、来源或同图历史。
- 四套 benchmark 的 prompt、validator、scorer、聚合和 cache identity 彼此独立。
- 不同 model revision、input track、decoding、inference/scorer protocol 的结果不得混用。
- 正式输出全部位于仓库外：MSMU 使用 `MANUAL_TEST_OUTPUT_ROOT`，其余分别使用
  `CVBENCH_OUTPUT_ROOT`、`QSPATIAL_OUTPUT_ROOT`、`SPBENCH_SI_OUTPUT_ROOT`。
- 仓库根不得创建 `output/` 或 `outputs/`；`tmp/` 是用户私有草稿区，agent 不读取或同步。
- 新下载与正式产物写入 `/media/datasets/lihaoran/`；`/media/datasets/tangzecong/` 既有资产只读。
- GPU 推理、正式评分和付费 API 调用分别需要明确授权；只读检查不等于运行许可。
- InternVL3-78B 固定 BF16、TP=4、四张 80GB GPU，不得用 TP=2 或量化替代。

## 仓库结构

```text
src/spatial_vlm_eval/
├── benchmarks/          # 每个 benchmark 的数据合同、validator、scorer、报告
├── models/              # model-family processor/template、图像输入与 generation
└── orchestration/       # 跨 benchmark 编排；不承载评分语义
scripts/
├── msmu/
├── cv_bench/
├── q_spatial/
├── spbench_si/
└── internvl3_78b/
tests/                   # 协议不变量与 bug 回归
docs/                    # 文档地图、协议、runbook、ADR、troubleshooting
```

模型、dataset、checkpoint、API key、prediction、judge cache、论文 PDF 和服务器环境 manifest 均不进入 Git。

## 环境准备

安装任何依赖前先复用现有 Conda 环境：

```bash
conda env list
```

本地非 CUDA 开发环境：

```bash
conda create -n spatial-vlm-eval-dev python=3.10 -y
conda activate spatial-vlm-eval-dev
python -m pip install -r requirements/local-dev.txt
python -m pip install -e . --no-deps
```

GPU 环境必须根据服务器 CUDA/驱动选择 PyTorch，并优先复用 `.env.server` 中的 `LATENT_PYTHON` 和
family-specific interpreter；不要向系统 Python 补包。完整存储与环境约定见
[MSMU 推理手册](docs/msmu-inference.md#1-运行前边界)。

## 公共操作入口

以下只展示路由与安全检查；完整参数、授权边界和恢复方式以对应 runbook 为准。

### 三 benchmark 共用 InternVL3-78B

```bash
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --dry-run
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --status
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --check
```

正式入口为同一脚本的无参数调用，按 Q-Spatial -> SPBench-SI -> CV-Bench 复用一次四卡 vLLM；见
[三 Benchmark 一键测评](docs/internvl3-78b-three-bench-evaluation.md)。

### CV-Bench

```bash
bash scripts/cv_bench/run_inference.sh --list
bash scripts/cv_bench/run_inference.sh --stage test --model PROFILE
bash scripts/cv_bench/run_inference.sh --stage full --model PROFILE
bash scripts/cv_bench/run_internvl3_78b_evaluation.sh --status
```

配置与产物根为 `CVBENCH_OUTPUT_ROOT`；评分和报告见
[简明指令](docs/cv-bench-commands.md)与[两阶段 runbook](docs/cv-bench-two-stage-runbook.md)。

### Q-Spatial

```bash
bash scripts/q_spatial/run_inference.sh --list
bash scripts/q_spatial/run_scheduled_batch.sh --check
bash scripts/q_spatial/run_scheduled_batch.sh --dry-run
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --status
```

配置和产物根为 `QSPATIAL_OUTPUT_ROOT`；21 轨、双卡 lane、评分与报告见
[简明指令](docs/q-spatial-commands.md)与[两阶段 runbook](docs/q-spatial-two-stage-runbook.md)。

### SPBench-SI

```bash
bash scripts/spbench_si/run_inference.sh --list
bash scripts/spbench_si/run_scheduled_batch.sh --check
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --status
bash scripts/spbench_si/build_results_report.sh --check
```

配置和产物根为 `SPBENCH_SI_OUTPUT_ROOT`；部分汇总、双 scorer 与 Gemini 可选续接边界见
[简明指令](docs/spbench-si-commands.md)与[两阶段 runbook](docs/spbench-si-two-stage-runbook.md)。

### MSMU

人工测试从[三阶段统一入口](docs/msmu-all-model-test-commands.md)开始。已经通过 stage 1/2 的注册模型可用：

```bash
bash scripts/msmu/run_model_evaluation.sh MODEL --check
bash scripts/msmu/run_model_evaluation.sh MODEL --status
```

获准后，无参数 `bash scripts/msmu/run_model_evaluation.sh MODEL` 完成 full-987、精确单轨评分和全局报告
重建。正式评分命令见[阶段三评分指令](docs/msmu-stage3-scoring-commands.md)。

## 验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

行为变化见 [CHANGELOG](CHANGELOG.md)，长期取舍见 [ADR](docs/decisions/README.md)，已解决的服务器问题见
[Troubleshooting](docs/troubleshooting/README.md)。
