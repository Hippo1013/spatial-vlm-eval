# 文档地图与维护规则

本文件是项目文档层的统一入口。`AGENTS.md` 保存 coding agent 必须遵守的规则和阅读路由；
本文件面向人类协作者与 agent，说明每类信息的唯一事实源、文档位置和更新时机。

## 单一事实源

| 信息 | 机器事实源 | 人类文档 |
|---|---|---|
| MSMU 输入、校验、judge、阈值与聚合 | `src/spatial_vlm_eval/benchmarks/msmu/` | [canonical protocol](benchmarks/msmu/protocol.md) |
| CV-Bench 数据、23 条轨、校验、评分与报告 | `src/spatial_vlm_eval/benchmarks/cv_bench/` | [CV-Bench protocol](benchmarks/cv_bench/protocol.md) |
| Q-Spatial 数据、21 条轨、校验、numeric scorer 与报告 | `src/spatial_vlm_eval/benchmarks/q_spatial/` | [Q-Spatial protocol](benchmarks/q_spatial/protocol.md) |
| SPBench-SI 数据、21 条轨、校验、双 scorer 与报告 | `src/spatial_vlm_eval/benchmarks/spbench_si/` | [SPBench-SI protocol](benchmarks/spbench_si/protocol.md) |
| 四 benchmark 范围、日期化进度快照与数据边界 | 服务器 validator/metadata/summary/publication gates；代码由各 registry 固化 | [评测范围与进度](evaluation-scope.md) |
| 项目级模型身份、profile、revision、输入轨与 decoding | 各 benchmark registry | [模型矩阵](model-matrix.md) |
| 已注册 MSMU profile、revision 与 inference protocol | `src/spatial_vlm_eval/models/profiles.py` 的 `PROFILES` / `CURRENT_TARGET_PROFILE_KEYS` | [模型矩阵的 MSMU profile](model-matrix.md#msmu-当前-18-条已完成目标-inference-profile) |
| CV-Bench 23 条目标轨及顺序 | `benchmarks.cv_bench.profiles.PROFILE_SEQUENCE` / `PROFILES` | [模型矩阵的 CV-Bench profile](model-matrix.md#cv-bench-当前-23-条目标-inference-profile) |
| Q-Spatial 21 条目标轨及顺序 | `benchmarks.q_spatial.profiles.PROFILE_SEQUENCE` / `PROFILES` | [模型矩阵的 Q-Spatial profile](model-matrix.md#q-spatial-当前-21-条目标-inference-profile) |
| SPBench-SI 21 条目标轨及顺序 | `benchmarks.spbench_si.profiles.PROFILE_SEQUENCE` / `PROFILES` | [模型矩阵的 SPBench-SI profile](model-matrix.md#spbench-si-当前-21-条目标-inference-profile) |
| InternVL3-78B 三 benchmark 共享服务编排 | `orchestration.internvl3_78b_three_bench` / `scripts/internvl3_78b/` | [三 Benchmark 一键测评](internvl3-78b-three-bench-evaluation.md) |
| 阶段三默认/Qwen3 补测轨与顺序 | `run_stage3_serial_inference.sh --list` / `--qwen3 --list` | [阶段三 runbook](msmu-stage3-full-eval.md) |
| 当前运行与评分状态 | 服务器 `status.tsv`、validator、metadata、`summary.json`、publication gates 和报告 | 评测范围只保存注明日期的已验证快照 |
| CLI、环境变量与输出布局 | 脚本 `--help`、`configs/*server.env.example` | 对应 runbook |
| 服务器显式出站代理 | 仓库外 `/media/datasets/lihaoran/tools/mihomo/` | [网络代理手册](server-network-proxy.md) |
| 精确代码历史 | Git commit/diff | 根目录 [CHANGELOG](../CHANGELOG.md) 只记录语义变化 |
| 长期设计取舍 | 实现与回归测试 | [ADR](decisions/README.md) |
| 原始故障证据与已解决问题 | 未跟踪运行日志 | [Troubleshooting](troubleshooting/README.md) 只保留可复用结论 |

机器事实源与文档冲突时先停止执行，核对代码、测试和已验证产物，再在同一变更中修正文档；不得为了
让文档“看起来一致”而静默改变协议。

## 信息层级与按需读取

| 层级 | 只保存 | 不保存 | 读取方式 |
|---|---|---|---|
| `AGENTS.md` | 每次 coding 都必须看到的红线、任务路由和更新门禁 | 历史叙事、完整协议、运行状态 | 每次任务加载 |
| 根 `README.md` | 稳定能力、最短入口和文档导航 | 完整 profile 表、实时进度、故障流水 | 新人或首次进入仓库时读取 |
| `docs/` | 按职责拆分的 canonical 协议、架构、状态快照与 runbook | 原始日志全文、个人便条 | 只按 `AGENTS.md` 路由读取任务命中的文件 |
| Agent 记忆 | 用户偏好、跨项目原则、代码与文档中不易发现的复用提醒 | 可从仓库读取的协议、profile、SHA、分数和实时状态 | 先定位线索，再现场核验 |
| 未跟踪运行产物 | journal、日志、validator、metadata、status、summary 等原始证据 | 长期规则和人工总结 | 诊断、汇报或发布前现场读取 |

不要为“保险”一次加载全部文档。常规任务先读 `AGENTS.md`、根 `README.md` 和本索引，再只补读路由
命中的 protocol、runbook、ADR、troubleshooting 与测试。Agent 记忆只能帮助定位事实源，不能覆盖
仓库文档或服务器产物；易漂移事实必须重新验证。

## 文档清单

### 规则、架构与协议

- [仓库架构](architecture.md)：数据所有权、模块边界、恢复机制和输出布局。
- [MSMU canonical protocol](benchmarks/msmu/protocol.md)：MSMU 评分与输入协议的唯一规范文档。
- [CV-Bench canonical protocol](benchmarks/cv_bench/protocol.md)：锁定数据、模型输入、23 条轨、
  robust scorer 和 publication gates。
- [Q-Spatial canonical protocol](benchmarks/q_spatial/protocol.md)：两根数据合同、Standard Prompt、21 条
  轨、numeric scorer 与 publication gates。
- [SPBench-SI canonical protocol](benchmarks/spbench_si/protocol.md)：ZIP 直读单图合同、default/direct
  prompt、21 条轨、严格原始 MRA 主分与独立 upstream audit。
- [来源记录](source-provenance.json)：上游 commit、模型 revision 和文件哈希。

### 状态与参考

- [四 Benchmark 评测范围与项目进度](evaluation-scope.md)：新人入口；集中维护日期化进度、下一步、
  SOTA 对照来源与服务器资产边界。
- [模型矩阵](model-matrix.md)：19 个项目级目标模型身份、CV-Bench 23 条、Q-Spatial/SPBench-SI 各
  21 条目标轨，以及 MSMU 已落地 profile、锁定身份和已知偏差；不复制易漂移进度。
- [Judge 提示词中文参考](msmu-judge-prompts-zh-reference.md)：人工阅读译文；英文 scorer 源码仍是
  唯一运行真值。
- [MSMU 遗留小问题](benchmarks/msmu/known-minor-issues.md)：仅供以后人工复核的暂缓问题清单。
- [Q-Spatial 遗留小问题](benchmarks/q_spatial/known-minor-issues.md)：单次/multi-seed 与公开图片计数说明。
- [本地论文目录说明](../benchmark_paper/README.md)：不进入 Git 的论文文件约定。

### 运行手册

- [InternVL3-78B 三 Benchmark 一键测评](internvl3-78b-three-bench-evaluation.md)：只加载一次
  四卡 vLLM，固定 Q-Spatial → SPBench-SI → CV-Bench 顺序，独立完成各自 validator/scorer，并仅在
  对应历史报告源齐全时重建 report。
- [CV-Bench 简明运行指令](cv-bench-commands.md)：操作者直接复制的 test/full/评分/汇总命令。
- [CV-Bench 两阶段 runbook](cv-bench-two-stage-runbook.md)：test gate、full-2638、目录评分与报告命令。
- [CV-Bench InternVL3-78B 一键评测](cv-bench-internvl3-78b-evaluation.md)：四卡 TP=4 的 test/full、
  独立校验、精确单轨评分和原有全局报告重建。
- [Q-Spatial 简明运行指令](q-spatial-commands.md)：21 轨 test/full/评分/报告的可复制命令。
- [Q-Spatial 两阶段 runbook](q-spatial-two-stage-runbook.md)：两根数据、endpoint、gate、full-271 与评分。
- [Q-Spatial InternVL3-78B 四卡补测](q-spatial-internvl3-78b-evaluation.md)：沿用现有输出根完成
  TP=4 test/full、精确单轨评分和原报告 21/21 重建，并提供内置运行 FAQ。
- [SPBench-SI 简明运行指令](spbench-si-commands.md)：test/full、双卡 20 轨、评分与报告命令。
- [SPBench-SI 两阶段 runbook](spbench-si-two-stage-runbook.md)：只读 ZIP 数据、绑定 gate、full-1009、
  双卡 lane 与四卡 78B 边界。
- [SPBench-SI InternVL3-78B 四卡完整评测](spbench-si-internvl3-78b-evaluation.md)：一键 test/full、
  独立校验、精确双协议评分、报告重建与 FAQ。
- [MSMU 多模型推理与验收](msmu-inference.md)：环境、模型 family 和完整产物说明。
- [三阶段统一入口](msmu-all-model-test-commands.md)：人工测试总入口。
- [阶段一 canary](msmu-stage1-canary.md)：接口、processor 和视觉链路检查。
- [阶段二 smoke8](msmu-stage2-smoke8.md)：八类各一条的 debug 验收。
- [阶段三 full-987](msmu-stage3-full-eval.md)：获准轨、正式推理、抽查与评分流程。
- [阶段三评分命令](msmu-stage3-scoring-commands.md)：只保留操作者需要输入的评分指令。
- [GPU burn 启停](server-gpu-burn-runbook.md)：项目协作者管理的固定 burn pane 操作。
- [服务器网络代理](server-network-proxy.md)：Mihomo 首次配置、tmux 生命周期、按 shell 开关与验证。

### 历史、决策与故障知识

- [CHANGELOG](../CHANGELOG.md)：对结果、行为或操作方式有影响的语义变化。
- [ADR 索引与模板](decisions/README.md)：长期设计决策及其后果。
- [Inference/scorer protocol 分离决策](decisions/0001-separate-inference-and-scorer-protocols.md)。
- [CV-Bench 稳健解析与发布门禁决策](decisions/0002-cv-bench-robust-parser-and-publication-gates.md)。
- [Q-Spatial declared-final numeric parser v2 与发布门禁决策](decisions/0003-q-spatial-robust-numeric-parser-and-publication-gates.md)。
- [SPBench-SI 原始 MRA、真实输出 parser v2 与 upstream audit 分离决策](decisions/0004-spbench-si-original-mra-and-upstream-audit.md)。
- [Troubleshooting 规则](troubleshooting/README.md)与
  [服务器问题库](troubleshooting/server.md)。

## 日志生命周期

1. 推理、批次和评分先把原始证据写入未跟踪结果目录；日志不得含密钥、私有 endpoint 或图片
   base64，也不复制进 Git。
2. 未定位问题只保留在原始日志或 issue。`status.tsv`、validator、metadata 和 `summary.json` 是当前
   状态事实源，不能用日志最后一行或 Agent 记忆替代。
3. 根因确认且修复验证后，只把可复用的“症状、原因、处理、验证”提炼到 troubleshooting；不粘贴
   traceback 或完整运行过程。
4. 行为或操作语义变化进入 CHANGELOG，长期设计取舍进入 ADR。不要新建 `DEVLOG.md`，也不要在
   AGENTS、README、CHANGELOG、ADR 和 troubleshooting 之间复制同一段叙述。
5. 正式结果、人工抽查和派生导出都写入仓库外 `OUTPUT_ROOT`；仓库根不得创建 `output/` 或
   `outputs/`。`logs/` 不进入 Git；用户自用的 `tmp/` 不属于项目知识源，agent 不读取或同步。

## 更新触发条件与时机

文档更新是实现的一部分，必须在同一变更完成；不能先合入行为变化、以后再补文档。

| 触发事件 | 必须更新 | 时机 |
|---|---|---|
| benchmark 范围、推进顺序或项目级目标模型身份改变 | 评测范围、模型矩阵、README、CHANGELOG、文档一致性测试 | 决定生效时 |
| 输入、prompt、图像处理、judge、阈值、聚合或 cache identity 改变 | protocol、相关 ADR、CHANGELOG、回归测试；必要时更换 protocol/cache id | 代码完成前 |
| profile、模型 revision、decoding 或原生/公平轨改变 | `profiles.py`、模型矩阵、推理手册、CHANGELOG、相关测试 | 同一提交 |
| 环境变量、服务器路径模板、CLI、输出布局或编排改变 | env example、对应 runbook、README 最短入口、脚本测试 | 同一提交 |
| 获准阶段范围或串行顺序改变 | 调度脚本、阶段三 runbook、AGENTS 约束、相关测试 | 执行新批次前 |
| inference/评分状态变化 | 先验证完整产物，再更新评测范围的日期化进度；里程碑才进入 CHANGELOG | 验证完成后、报告前 |
| 可复用故障已定位并验证 | troubleshooting：症状、根因、处理、验证；fix 必须有回归测试 | 与修复同一提交 |
| 故障尚未定位 | 只保留未跟踪运行日志或 issue，不进入 troubleshooting | 根因确认前禁止沉淀 |
| 新增、重命名或删除文档 | 本索引、所有相对链接和文档一致性测试 | 同一提交 |

## 写作边界

- README 只保留项目入口和稳定能力；进度只路由到评测范围，不复制完整 profile 名单或阶段状态。
- AGENTS 只保留硬约束、阅读路由和更新触发规则，不写历史流水。
- CHANGELOG 记录“发生了什么”，ADR 记录“为什么这样决定”，troubleshooting 记录“问题如何复现并
  解决”，运行日志保留原始证据；四者不得互相复制全文。
- 状态只能来自已经验证的服务器产物；推理完成不能写成评分完成。
- 文档、日志和示例不得包含 token、密码、私有 endpoint 或图片 base64。
