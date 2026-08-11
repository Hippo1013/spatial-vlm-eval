# MSMU 阶段三：完整 987 条推理与评分

## 目标

运行 official `test` split 全部 987 条，通过完整 validator 后用独立 local judge 评分。只有本阶段通过
publication gates 的 summary 才能进入结果表。固定写入：

```text
/media/datasets/lihaoran/latent_reasoning/msmu-outputs/manual-three-stage-v1/03_full987/
```

只有阶段二通过的模型才能进入本阶段。脚本会自动清除 `LIMIT`、`INDICES` 和
`MSMU_SMOKE_INDICES`，不会把 subset 参数带入正式运行。

本轮阶段三固定运行 13 条本地推理轨：

```text
llava_next_mistral_7b, llava_next_yi_34b
internvl3_8b, internvl3_38b
qwen25_vl_base, qwen25_vl_32b
ssr, ssr_native, spatialrgpt
3dthinker, 3dthinker_native
spatialbot, spatialbot_native
```

已完成的历史 13 轨明确排除 GPT-5、Gemini（API）、Qwen2.5-VL-72B、InternVL3-78B（70B+）和
当时未测的 Qwen PEFT。Qwen2.5-VL-72B 的阶段三手工入口仍会拒绝；InternVL3-78B 后续改为独立
四卡手工补测，不追溯加入默认 13 轨或改写其完成标记。

以上 13 轨保留为串行脚本默认计划。Qwen3-VL 2B/4B/8B/32B 在 stage 1/2 通过后单独补测：

```bash
bash scripts/msmu/run_stage3_serial_inference.sh --qwen3 --list
MANUAL_DRY_RUN=1 bash scripts/msmu/run_stage3_serial_inference.sh --qwen3
bash scripts/msmu/run_stage3_serial_inference.sh --qwen3 --check
bash scripts/msmu/run_stage3_serial_inference.sh --qwen3
bash scripts/msmu/run_stage3_serial_inference.sh --qwen3 --status
```

`--qwen3` 只串行四条新轨，状态位于 `_serial_inference/qwen3/`，不修改默认 13 轨完成标记。

RoboBrain NV/MT、HiSpatial、SpatialLadder direct/thinking 不加入上述两个历史计划，使用新的冻结双
lane 入口：

```bash
bash scripts/msmu/run_sota_supplement.sh --list
MANUAL_DRY_RUN=1 bash scripts/msmu/run_sota_supplement.sh
bash scripts/msmu/run_sota_supplement.sh --check
bash scripts/msmu/run_sota_supplement.sh
bash scripts/msmu/run_sota_supplement.sh --status
```

该入口在两条 lane 都 full-987 完成后自行统一评分并原子重建 23 行报告；详见
[MSMU SOTA 双 Lane runbook](msmu-sota-supplement.md)。

## 第一步：串行完成 13 条完整推理

如需释放服务器上由本项目协作者管理的 GPU burn，先按
[GPU burn 启停手册](server-gpu-burn-runbook.md)停止固定 pane；不要终止其他 GPU 进程。

推荐在 `msmu` tmux session 新建一个窗口，只运行一条命令：

```bash
tmux new-window -t msmu -n 30-full-batch
tmux send-keys -t msmu:30-full-batch \
  "cd /media/datasets/lihaoran/latent_reasoning/spatial-vlm-eval && bash scripts/msmu/run_stage3_serial_inference.sh" C-m
```

先查看实际顺序或做不占 GPU 的 dry-run：

```bash
bash scripts/msmu/run_stage3_serial_inference.sh --list
MANUAL_DRY_RUN=1 bash scripts/msmu/run_stage3_serial_inference.sh
bash scripts/msmu/run_stage3_serial_inference.sh --check
```

脚本对每条轨执行“等待所需 GPU 空闲 → 部署/加载 → 987 条推理 → 完整 validator → 停止自有服务
→ 确认 GPU 释放”，然后才进入下一条。它固定 `RUN_SCORE=0`，不会启动 judge 或评分。
`--check` 会一次性检查全部路径/解释器、独占锁、端口和两张 GPU，不加载任何模型。

批处理状态写入：

```text
03_full987/_serial_inference/
├── plan.env
├── status.tsv
├── active_process.env        # 仅运行时存在
├── completed/MODEL.complete
└── logs/TIMESTAMP.log
```

查看已完成/待运行模型：

```bash
bash scripts/msmu/run_stage3_serial_inference.sh --status
```

默认每个模型最多尝试 2 次。逐样本 journal 每次成功后 `fsync`；模型失败、终端中断或 watchdog
终止后，用完全相同的批处理命令重跑，会跳过同一 commit 下已有完成标记，并从当前模型 journal
续跑。不要编辑 journal、`plan.env` 或完成标记。

为防止无人值守任务无限卡住，脚本提供以下有界门禁：

- vLLM readiness 最多等待 1800 秒；
- 推理 journal/log 连续 3600 秒无更新才判定 stalled；
- GPU 等待最多 1800 秒，模型退出后释放最多等待 600 秒；
- `TERM` 后给自有进程组 90 秒清理，再只对该进程组使用 `KILL`；
- 同一输出根使用 `flock` 防止两个批次同时运行；
- 已占用 GPU、已有 `18081` 服务或遗留活动进程只报告并退出，绝不终止非本脚本进程。

服务 readiness 通过 `.env.server` 中的 `LATENT_PYTHON` 和 Python 标准库直接查询本地精确
`/v1/models` ID；运行镜像不需要另行安装 `curl`，也不会让本地探活请求经过代理。

这些值可以用脚本 `--help` 中列出的 `BATCH_*` 环境变量调整。默认失败两次后停止整个批次，比静默
跳过模型更安全；修复原因后重跑即可恢复。

### 单模型一键正式评测

已经通过 stage 1/2、但需要单独完成正式结果的任一注册模型，统一运行：

```bash
bash scripts/msmu/run_model_evaluation.sh MODEL
```

例如四卡 InternVL3-78B：

```bash
bash scripts/msmu/run_model_evaluation.sh internvl3_78b
```

入口自动完成“必要时启动被测服务 → full-987 + validator → 停止被测服务并释放 GPU → 启动 judge →
只评分本次模型的精确 `predictions.jsonl` → 停止 judge → 重建全局报告”。它从
`run_manual_stage3.sh` 的共享注册信息解析 backend、served model、GPU 和输出路径，不维护另一份
模型名单；API、vLLM、Qwen 和空间专用模型使用同一条命令。明确未获准 stage 3 的 profile 仍会
fail closed。

```bash
# 查看可用 MODEL 名称
bash scripts/msmu/run_model_evaluation.sh --list

# 只检查注册、路径、锁和端口
bash scripts/msmu/run_model_evaluation.sh MODEL --check

# 查看该模型精确结果的状态
bash scripts/msmu/run_model_evaluation.sh MODEL --status

# 只预览完整编排
MANUAL_DRY_RUN=1 bash scripts/msmu/run_model_evaluation.sh MODEL
```

中断后重跑同一条默认命令即可复用 journal/judge cache。若完整 inference 产物已经重新校验通过，入口
不会重新加载被测模型；若 summary 已通过全部 publication gates，则直接跳到全局报告。已有端口、
GPU 进程和批次锁均保持原状，脚本只终止自己创建的进程组。控制器日志写入
`03_full987/_single_model_evaluation/logs/UTC_TIMESTAMP-MODEL.log`。
被测服务和 judge 的 readiness 同样由 `LATENT_PYTHON` 标准库探测，不依赖系统 `curl`。

### 单模型手工备用入口

需要单独诊断时，LLaVA/InternVL 在两个终端分别运行：

```bash
bash scripts/msmu/run_manual_stage3.sh MODEL serve
bash scripts/msmu/run_manual_stage3.sh MODEL infer
```

Qwen 和空间专用模型直接运行：

```bash
bash scripts/msmu/run_manual_stage3.sh MODEL infer
```

`infer` 同样固定 `RUN_SCORE=0` 并在结束时运行完整 validator。

InternVL3-78B 的独立四卡补测日常使用上方一键入口；只有诊断服务或推理时才拆成以下手工命令。
默认 GPU `0,1,2,3`、固定 TP=4：

```bash
# 终端 A
bash scripts/msmu/run_manual_stage3.sh internvl3_78b serve

# 终端 B：服务 ready 后运行 full-987 + 正式 validator
bash scripts/msmu/run_manual_stage3.sh internvl3_78b infer
```

serve 会在加载前确认至少四张物理/选中 GPU 且逐卡空闲。中断后重新执行同一条 `infer` 命令会从
fsync journal 恢复；完整 validator 通过后，在终端 A 停止自有 78B 服务，再进入独立 judge 与目录
驱动评分流程。

## 可选：正式评分前抽查答案

仅在 13 条完整 validator 均通过、尚未生成评分目录时运行：

```bash
# 固定抽取同一组 30 条，生成本地 Markdown 与图片
source scripts/msmu/prepare_manual_test.sh
python scripts/msmu/build_stage3_answer_audit.py \
  --stage3-root "$OUTPUT_ROOT/03_full987" \
  --dataset-root "$DATASET_ROOT" \
  --output "$OUTPUT_ROOT/_answer_audit/msmu-stage3-answer-audit.md"
```

仓库根不得创建 `output/` 或 `outputs/`；抽查文档及图片资产保存在仓库外，脚本会拒绝覆盖已有文件。

## 第二步：启动独立 judge

确认完整推理和 validator 已通过，然后停止不再需要的被测模型服务。在一个独立终端启动 judge：

```bash
bash scripts/msmu/run_manual_stage3.sh judge serve
```

它从 `.env.server` 读取 `JUDGE_MODEL`，固定提供
`http://127.0.0.1:18080/v1`，并在启动前运行 GPU preflight。指定其他已协调 GPU：

```bash
MANUAL_JUDGE_CUDA_VISIBLE_DEVICES=1 \
  bash scripts/msmu/run_manual_stage3.sh judge serve
```

被测 vLLM 固定使用 `18081`；judge 与被测服务不能使用同一个 endpoint。

## 第三步：串行评分已有结果

judge ready 后，在另一个终端执行：

```bash
bash scripts/msmu/score_pending_results.sh --list
bash scripts/msmu/score_pending_results.sh --check
bash scripts/msmu/score_pending_results.sh
bash scripts/msmu/score_pending_results.sh --status
```

完整命令见[阶段三串行评分指令](msmu-stage3-scoring-commands.md)。

## 第四步：生成结果表

先只读列出全部 scorer protocol 下已有评分：

```bash
bash scripts/msmu/build_results_report.sh --list
```

未指定筛选时收录当前 canonical scorer protocol 下全部通过完整 publication gates 的结果；需要
指定模型时使用 metadata profile，参数顺序也是输出顺序：

```bash
bash scripts/msmu/build_results_report.sh \
  --profile PROFILE_A \
  --profile PROFILE_B
```

必要时可传入一个 `--scorer-protocol` 精确选择历史或当前评分；精简表不允许混合多个 scorer
protocol。默认输出为 `03_full987/msmu-result.md`，文件内容固定只有标题
`MSMU-Bench评测结果` 和一张中文表。专用模型直接在模型名称中标实际配置：SSR 使用 `RGB` /
`RGB + 深度估计`，SpatialRGPT 不加注释，3DThinker 使用 `RGB` / `RGB + Mental-3D 提示词`，
SpatialBot 使用 `RGB` / `RGB + 深度估计`。标题与表格之间固定包含一行注释，说明括号内是实际输入
或提示配置，并明确深度估计来自当前 RGB 而不是 GT 深度。
不完整、不可发布或 provenance 不一致的诊断 summary 会在终端告警并跳过；筛选后没有任何合法
评分时 fail closed。

SOTA supplement 写报告前必须先运行只读 `build_results_report.sh --check`；默认冻结检查 18 条 baseline
+ 4 条 main + 1 条 thinking，缺失、重复或不完整 summary 都拒绝覆盖现有报告。

## 推荐 tmux 名称

继续使用三个阶段共用的 session `msmu`，窗口职责建议：

```text
30-full-batch  # 13 条串行完整推理
MODEL-srv     # 仅 vLLM 被测模型需要
MODEL-full    # 987 条推理
judge-srv     # 本地 judge
serial-score  # 目录驱动的串行评分
```

## 正式通过标准

- `predictions.jsonl` 恰好 987 行，index 精确覆盖 `0..986`；
- metadata 中 `num_predictions: 987`、dataset `num_targets: 987`、
  `publishable_inference: true`；
- `prediction_validation.json` 中 `passed: true`、`allow_subset: false`、
  `num_prediction_rows: 987`、`num_unique_indices: 987`；
- `scores/<scorer-protocol>/summary.json` 中 `num_samples: 987`、八类齐全、
  `num_judge_failures: 0`、`publishable: true`。

查看正式产物：

```bash
find /media/datasets/lihaoran/latent_reasoning/msmu-outputs/manual-three-stage-v1/03_full987 \
  -type f \
  \( -name 'prediction_validation.json' -o -name 'predictions.jsonl.metadata.json' \
     -o -name 'summary.json' \) \
  -print | sort
```

需要检查结果时只需提供 SSH 连接方式，不要发送 API key。
