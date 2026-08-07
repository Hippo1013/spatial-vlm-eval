# AGENTS.md

本文件约束本仓库中的人类协作者与自动化 coding agent。开始任务前先阅读根 README 和
`docs/README.md`，再按下方路由读取对应协议、runbook 与测试；不得只凭 README 的摘要执行。

## 项目目标

本仓库用于对多种通用/空间专用 VLM 进行可复现的多 benchmark 评测。实现必须优先保证：

1. 被测模型看不到协议禁止的信息；
2. 每个结果可以追溯到数据 split、模型、prompt、图像处理、decoding、judge 和 scorer；
3. 不同 scorer protocol 的结果绝不在无 protocol 列的表中混合；精简展示表只有在逐行校验 provenance、一次只选择一个 scorer protocol，并在模型名称中区分 input track 时才可省略列；
4. benchmark 逻辑与模型适配逻辑解耦。

## Python 环境政策

- 安装任何 Python 依赖前，先运行 `conda env list` 并检查已有环境。
- 优先复用已经满足任务的 Conda 环境。
- 只有没有合适环境时才创建任务专用环境。
- 默认/系统 Python 缺包不是阻塞理由，也不是向默认解释器安装包的理由。
- PyTorch 必须依据服务器 CUDA/驱动选择安装源；不要盲目执行通用 requirements 安装命令。
- 必要的测试、验证或开发环节需要服务器环境时，必须优先通过 SSH 在服务器上执行；若不知道
  当前连接方式，必须向用户询问，不得自行猜测。
- 服务器项目、正式输出及任何新下载的 dataset/model/environment/cache/upstream/checkpoint 均写入
  `/media/datasets/lihaoran/`；`/media/datasets/tangzecong/` 现有资产只通过配置中的 legacy 变量引用，
  不移动、不删除，也不得继续向其中下载新资产。

## 代码边界

- `benchmarks/<name>/`：数据合同、预测 schema、validation、judge/scoring、汇总。
- `models/<family>/`：官方 processor/chat template、图像输入、模型加载与 generation。
- `scripts/<benchmark>/`：环境变量、日志、进程编排；不得复制 Python 评分逻辑。
- `tests/`：协议不变量和 bug 回归。
- 不要在源码中硬编码单台服务器路径；路径由 CLI 或环境变量提供。
- 模型、dataset、checkpoint、prediction、judge cache 和论文 PDF 不提交到 Git。
- 仓库根不得创建 `output/` 或 `outputs/`；正式结果、人工抽查和派生导出均写入仓库外
  `OUTPUT_ROOT` / `MANUAL_TEST_OUTPUT_ROOT`。
- `tmp/` 是用户自用的本地草稿区；agent 不读取、不整理、不提交，也不向服务器同步。

## 文档读取路由

以下材料必须在设计或执行对应动作前阅读；修改中若任务范围变化，立即补读新命中的文档。

| 任务触发条件 | 必须阅读 | 时机 |
|---|---|---|
| 修改 benchmark 输入、schema、validator、judge、阈值、cache 或聚合 | 对应 `docs/benchmarks/<name>/` 协议、`docs/architecture.md`、相关 benchmark 测试 | 方案与编辑前 |
| 运行或修改 Q-Spatial contract、21 轨、推理、评分或报告 | `docs/benchmarks/q_spatial/protocol.md`、`docs/q-spatial-two-stage-runbook.md` 与 Q-Spatial 测试 | 设计或执行前 |
| 运行或修改 SPBench-SI contract、21 轨、推理、评分或报告 | `docs/benchmarks/spbench_si/protocol.md`、`docs/spbench-si-two-stage-runbook.md` 与 SPBench-SI 测试 | 设计或执行前 |
| 修改模型 profile、processor/template、图像输入、decoding 或 revision | `docs/model-matrix.md`、`docs/msmu-inference.md`、对应 benchmark 输入协议和 model 测试 | 设计 adapter 前 |
| 修改 shell、环境变量、输出路径、服务器部署或 GPU 编排 | `docs/msmu-inference.md`、相关阶段 runbook、`docs/troubleshooting/` 和脚本测试 | 执行服务器命令前 |
| 配置、操作或修改服务器显式出站代理 | `docs/server-network-proxy.md` | 输入订阅或执行代理命令前 |
| 运行三阶段人工测试 | `docs/msmu-all-model-test-commands.md` 与当前阶段文档 | 启动模型前 |
| 启动 judge 或正式评分 | MSMU protocol、`docs/msmu-stage3-scoring-commands.md`、`docs/architecture.md` | readiness 检查与评分前 |
| 查询当前进度、汇报或发布结果 | `docs/model-matrix.md`，并现场检查服务器 validator/metadata/status/summary | 写结论前 |
| 追溯行为变化、设计原因或已知故障 | `CHANGELOG.md`、相关 ADR、`docs/troubleshooting/` 与原始运行日志 | 下结论或修复前 |
| 新增 benchmark | 本文件“新增 benchmark”、`docs/evaluation-scope.md`、`docs/model-matrix.md`、`docs/README.md` 和已有 benchmark 的同类文件 | 创建目录或协议前 |

## MSMU 不变量

当前 canonical 文档是 `docs/benchmarks/msmu/protocol.md`。以下约束不可静默修改：

- 只使用 official `test` split，共 987 条。
- 每条模型输入只有对应图片和第一条 user question。
- 不向被测模型输入 reference、raw type、task family、其他 QA 或同图历史。
- 对 Qwen 删除字面 `<image>`，用 structured image content，并使用原生 chat template。
- 当前补测的 Qwen3-VL-Instruct 2B/4B/8B/32B profile 均为 greedy、`num_beams=1`、
  `max_new_tokens=192`、图像像素范围 `16384..147456`，不添加 system message；不同参数量必须使用
  独立 model revision、protocol 和输出目录。
- 保留的 Qwen2.5-VL 7B/32B/72B profile 仍锁定图像像素范围 `12544..112896` 和各自已有 protocol；
  不得用 Qwen3-VL 设置恢复旧 journal。
- 正式输出必须覆盖 index `0..986`；前五个元数据字段由 test row 确定，只有 prediction 来自模型。
- 空 prediction 是 warning，允许进入评分并得到零分或抽取失败；不得把它静默吞掉。
- scorer 必须在 judge 前强制运行完整校验。不得增加绕过正式校验的 scorer 参数。
- subset 仅可调试，不得形成正式 summary。
- 主指标是八类非加权 `official_macro8_accuracy`，不是 micro accuracy。
- 非 grounding 数值阈值为对称 ratio `< 1.25`。
- coordinate grounding 为 mean absolute coordinate error `<= 0.1`。
- qualitative 为 `your_mark > 0.5`。
- 多数值 prediction 短于 reference 时失败；更长时忽略多余尾值。
- object-at-coordinate 使用当前 grounding 专用本地 judge 语义 mark，而不是官方 MiniLM。

当前 scorer protocol id：

```text
sdvlm_official_compat_local_judge_v4_grounding_split_strict_quant_length_malformed_zero
```

改变 prompt、grounding 路由、阈值、列表长度语义、聚合方式或 judge 身份进入 cache key 的规则时，
必须更换 protocol/cache id、添加回归测试并更新文档。只做不改变 judge response 的确定性后处理修复时，
应审慎判断是否保留 cache protocol。

## Q-Spatial 不变量

canonical 文档是 `docs/benchmarks/q_spatial/protocol.md`。正式数据顺序固定为 ScanNet 170 条后接
Q-Spatial++ 101 条；Parquet 与 ScanNet RGB 使用两个显式根，后者的许可内容不复制、不打包、不提交。
adapter 只接收 `index`、一张 RGB、Standard system prompt 和当前 `Question: ...`，prediction 只含
`index, raw_prediction`。21 条 profile 中 18 条 RGB、3 条派生 depth/XYZ；不得加入 Mental-3D、thinking
或 ScanNet GT depth。full 必须通过绑定 red/blue canary、smoke8、processor/template、单图证据与
完整 provenance 的当前 test gate；subset 永不评分。当前 scorer protocol 是
`q_spatial_robust_numeric_v2_standard_prompt_declared_final_equivalent_tags_controlled_wrappers_paper_inclusive_ratio`，主指标为
两个 split 等权的 inclusive `δ≤2`；改变 parser、单位、边界或聚合必须换 protocol 并补测试/ADR。

## SPBench-SI 不变量

canonical 文档是 `docs/benchmarks/spbench_si/protocol.md`。本阶段只评单图 `test` 1,009 题，不含
SPBench-MV。loader 必须显式读取锁定 Parquet 与 ZIP，直接解码 524 张 JPEG；adapter 只能看到
`index, image, system_prompt, user_prompt`，prediction 只有 `index, raw_prediction`。21 轨中 18 条 RGB、
3 条同图派生 depth/XYZ，统一使用官方 `default/direct`，禁止 thinking/Mental-3D/GT depth。主 scorer
protocol 是 `spbench_si_original_mra10_strict_robust_direct_four_task_macro_v1`，使用十阈值严格 MRA 与
四题型宏平均；当前上游 direct-mode 只作为独立 audit，禁止混表。subset 不评分；暂行 20/21 报告只能
缺固定 TP=4 的 InternVL3-78B。

## 修改与验证流程

1. 先检查工作树，保留不相关的用户修改。
2. 修改最小必要模块，不把 benchmark 特例泄漏进其他 benchmark。
3. 更新或新增回归测试。
4. 运行：

   ```bash
   PYTHONPATH=src python -m unittest discover -s tests -v
   python -m compileall -q src tests
   ```

5. 修改 shell 脚本时运行 `bash -n scripts/**/*.sh`（在 Bash/Ubuntu 环境）。
6. 修改实际推理协议后，先做 `--limit` smoke test，再跑完整 987 条；subset 不得发布。
7. 正式评分前检查 `prediction_validation.json`，评分后检查 summary 的样本数和八类完整性。
8. 已完成的阶段三批次固定由 `run_stage3_serial_inference.sh` 表示 13 条历史获准本地轨；不得改写
   其默认名单或完成标记。Qwen3-VL 2B/4B/8B/32B 四条补测轨使用同一脚本的 `--qwen3` 计划，
   状态与历史批次隔离。InternVL3-78B 使用固定 TP=4 的独立四卡手工补测入口，不得并入历史 13 轨。
9. 阶段三正式评分使用目录驱动的 `score_pending_results.sh`；不得维护模型名单或绕过批次锁、
   judge readiness/model 检查和评分后 publication gates。

## 文档更新触发

- 协议、profile、revision、环境变量、CLI、输出布局或批次范围改变时，必须在同一变更中更新
  `docs/README.md` 指定的事实源、CHANGELOG、相关 ADR/runbook 和回归测试。
- 阶段状态只能在对应 validator、metadata、status 或 summary 现场验证后更新；推理完成不得写成
  评分完成。
- 可复用故障在根因确认、修复和验证后，与修复代码同一提交写入 `docs/troubleshooting/`；未定位问题
  只保留在未跟踪运行日志或 issue。
- 新增、重命名或删除文档时同步更新 `docs/README.md` 和全部相对链接。
- 任务完成前运行文档一致性测试；详细职责与时机以 `docs/README.md` 为准。

## 结果与命名

- 每个模型、checkpoint、decoding profile 和 scorer protocol 使用独立输出目录。
- 目录名应包含有辨识度的模型/run 名和 protocol；不要复用模糊的 `local_judge/`。
- 报告必须区分 `official-compatible internal score` 与 `strict official score`。
- 当前实现不是 strict official：judge、JSON 约束、grounding object 评分和 Qwen decoding 均与官方不同。
- `score.py`、旧 `batch_msmu_local_score.sh`、v1/v2 历史目录不属于本仓库的 canonical pipeline，
  不得重新引入或作为当前结果来源。

## 新增 benchmark

新增 benchmark 时至少提供：

- 数据/split 来源和不可泄漏字段说明；
- prediction schema 与 validator；
- scorer、protocol id、主指标与分项指标；
- 一个模型适配器或通用 prediction 导入入口；
- shell 编排和产物清单；
- 单元测试、smoke test 说明和 benchmark 文档；
- 对官方实现的 commit/version 记录及已知偏差。
