# AGENTS.md

本文件约束本仓库中的人类协作者与自动化 coding agent。开始修改前必须先阅读根 README、对应
benchmark 的协议文档和相关测试。

## 项目目标

本仓库用于对多种通用/空间专用 VLM 进行可复现的多 benchmark 评测。实现必须优先保证：

1. 被测模型看不到协议禁止的信息；
2. 每个结果可以追溯到数据 split、模型、prompt、图像处理、decoding、judge 和 scorer；
3. 不同协议的结果绝不在无 protocol 列的表中混合；
4. benchmark 逻辑与模型适配逻辑解耦。

## Python 环境政策

- 安装任何 Python 依赖前，先运行 `conda env list` 并检查已有环境。
- 优先复用已经满足任务的 Conda 环境。
- 只有没有合适环境时才创建任务专用环境。
- 默认/系统 Python 缺包不是阻塞理由，也不是向默认解释器安装包的理由。
- PyTorch 必须依据服务器 CUDA/驱动选择安装源；不要盲目执行通用 requirements 安装命令。
- 必要的测试、验证或开发环节需要服务器环境时，必须优先通过 SSH 在服务器上执行；若不知道
  当前连接方式，必须向用户询问，不得自行猜测。

## 代码边界

- `benchmarks/<name>/`：数据合同、预测 schema、validation、judge/scoring、汇总。
- `models/<family>/`：官方 processor/chat template、图像输入、模型加载与 generation。
- `scripts/<benchmark>/`：环境变量、日志、进程编排；不得复制 Python 评分逻辑。
- `tests/`：协议不变量和 bug 回归。
- 不要在源码中硬编码单台服务器路径；路径由 CLI 或环境变量提供。
- 模型、dataset、checkpoint、prediction、judge cache 和论文 PDF 不提交到 Git。

## MSMU 不变量

当前 canonical 文档是 `docs/benchmarks/msmu/protocol.md`。以下约束不可静默修改：

- 只使用 official `test` split，共 987 条。
- 每条模型输入只有对应图片和第一条 user question。
- 不向被测模型输入 reference、raw type、task family、其他 QA 或同图历史。
- 对 Qwen 删除字面 `<image>`，用 structured image content，并使用原生 chat template。
- 当前 Qwen2.5-VL 7B/32B/72B profile 均为 greedy、`num_beams=1`、`max_new_tokens=192`、
  图像像素范围 `12544..112896`；不同参数量必须使用独立 model revision、protocol 和输出目录。
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
sdvlm_official_compat_local_judge_v3_grounding_split_strict_quant_length
```

改变 prompt、grounding 路由、阈值、列表长度语义、聚合方式或 judge 身份进入 cache key 的规则时，
必须更换 protocol/cache id、添加回归测试并更新文档。只做不改变 judge response 的确定性后处理修复时，
应审慎判断是否保留 cache protocol。

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
8. 当前阶段三计划使用 `run_stage3_serial_inference.sh` 串行运行 13 条本地轨；不得加入两个 API、
   Qwen PEFT、Qwen2.5-VL-72B 或 InternVL3-78B，除非用户重新批准测试范围。

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
