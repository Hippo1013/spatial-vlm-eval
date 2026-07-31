# 文档地图与维护规则

本文件是项目文档层的统一入口。`AGENTS.md` 保存 coding agent 必须遵守的规则和阅读路由；
本文件面向人类协作者与 agent，说明每类信息的唯一事实源、文档位置和更新时机。

## 单一事实源

| 信息 | 机器事实源 | 人类文档 |
|---|---|---|
| MSMU 输入、校验、judge、阈值与聚合 | `src/spatial_vlm_eval/benchmarks/msmu/` | [canonical protocol](benchmarks/msmu/protocol.md) |
| 模型 profile、revision 与 inference protocol | `src/spatial_vlm_eval/models/profiles.py` | [模型矩阵](model-matrix.md) |
| 阶段三默认/Qwen3 补测轨与顺序 | `run_stage3_serial_inference.sh --list` / `--qwen3 --list` | [阶段三 runbook](msmu-stage3-full-eval.md) |
| 当前运行与评分状态 | 服务器 `status.tsv`、validator、metadata 和 `summary.json` | 模型矩阵只保存注明日期的已验证快照 |
| CLI、环境变量与输出布局 | 脚本 `--help`、`configs/msmu-server.env.example` | 对应 runbook |
| 精确代码历史 | Git commit/diff | 根目录 [CHANGELOG](../CHANGELOG.md) 只记录语义变化 |
| 长期设计取舍 | 实现与回归测试 | [ADR](decisions/README.md) |
| 原始故障证据与已解决问题 | 未跟踪运行日志 | [Troubleshooting](troubleshooting/README.md) 只保留可复用结论 |

机器事实源与文档冲突时先停止执行，核对代码、测试和已验证产物，再在同一变更中修正文档；不得为了
让文档“看起来一致”而静默改变协议。

## 文档清单

### 规则、架构与协议

- [仓库架构](architecture.md)：数据所有权、模块边界、恢复机制和输出布局。
- [MSMU canonical protocol](benchmarks/msmu/protocol.md)：MSMU 评分与输入协议的唯一规范文档。
- [来源记录](source-provenance.json)：上游 commit、模型 revision 和文件哈希。

### 状态与参考

- [模型矩阵](model-matrix.md)：profile inventory、锁定身份、已知偏差和注明日期的验证状态快照。
- [Judge 提示词中文参考](msmu-judge-prompts-zh-reference.md)：人工阅读译文；英文 scorer 源码仍是
  唯一运行真值。
- [MSMU 遗留小问题](benchmarks/msmu/known-minor-issues.md)：仅供以后人工复核的暂缓问题清单。
- [本地论文目录说明](../benchmark_paper/README.md)：不进入 Git 的论文文件约定。

### 运行手册

- [MSMU 多模型推理与验收](msmu-inference.md)：环境、模型 family 和完整产物说明。
- [三阶段统一入口](msmu-all-model-test-commands.md)：人工测试总入口。
- [阶段一 canary](msmu-stage1-canary.md)：接口、processor 和视觉链路检查。
- [阶段二 smoke8](msmu-stage2-smoke8.md)：八类各一条的 debug 验收。
- [阶段三 full-987](msmu-stage3-full-eval.md)：获准轨、正式推理、抽查与评分流程。
- [阶段三评分命令](msmu-stage3-scoring-commands.md)：只保留操作者需要输入的评分指令。
- [GPU burn 启停](server-gpu-burn-runbook.md)：项目协作者管理的固定 burn pane 操作。

### 历史、决策与故障知识

- [CHANGELOG](../CHANGELOG.md)：对结果、行为或操作方式有影响的语义变化。
- [ADR 索引与模板](decisions/README.md)：长期设计决策及其后果。
- [Inference/scorer protocol 分离决策](decisions/0001-separate-inference-and-scorer-protocols.md)。
- [Troubleshooting 规则](troubleshooting/README.md)与
  [服务器问题库](troubleshooting/server.md)。

## 更新触发条件与时机

文档更新是实现的一部分，必须在同一变更完成；不能先合入行为变化、以后再补文档。

| 触发事件 | 必须更新 | 时机 |
|---|---|---|
| 输入、prompt、图像处理、judge、阈值、聚合或 cache identity 改变 | protocol、相关 ADR、CHANGELOG、回归测试；必要时更换 protocol/cache id | 代码完成前 |
| profile、模型 revision、decoding 或原生/公平轨改变 | `profiles.py`、模型矩阵、推理手册、CHANGELOG、相关测试 | 同一提交 |
| 环境变量、服务器路径模板、CLI、输出布局或编排改变 | env example、对应 runbook、README 最短入口、脚本测试 | 同一提交 |
| 获准阶段范围或串行顺序改变 | 调度脚本、阶段三 runbook、AGENTS 约束、相关测试 | 执行新批次前 |
| inference/评分状态变化 | 先验证完整产物，再更新模型矩阵的日期与状态；里程碑才进入 CHANGELOG | 验证完成后、报告前 |
| 可复用故障已定位并验证 | troubleshooting：症状、根因、处理、验证；fix 必须有回归测试 | 与修复同一提交 |
| 故障尚未定位 | 只保留未跟踪运行日志或 issue，不进入 troubleshooting | 根因确认前禁止沉淀 |
| 新增、重命名或删除文档 | 本索引、所有相对链接和文档一致性测试 | 同一提交 |

## 写作边界

- README 只保留项目入口和稳定能力，不复制完整 profile 名单或阶段状态。
- AGENTS 只保留硬约束、阅读路由和更新触发规则，不写历史流水。
- CHANGELOG 记录“发生了什么”，ADR 记录“为什么这样决定”，troubleshooting 记录“问题如何复现并
  解决”，运行日志保留原始证据；四者不得互相复制全文。
- 状态只能来自已经验证的服务器产物；推理完成不能写成评分完成。
- 文档、日志和示例不得包含 token、密码、私有 endpoint 或图片 base64。
