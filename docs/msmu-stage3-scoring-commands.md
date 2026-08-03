# MSMU 阶段三：串行评分指令

结果性质为 **official-compatible internal score**，不是 strict official score。

## 登录服务器

```bash
ssh msmu-a800
cd /media/datasets/tangzecong/latent_reasoning/spatial-vlm-eval
```

## 终端 A：启动 judge

```bash
bash scripts/msmu/run_manual_stage3.sh judge serve
```

保持该终端运行。

## 终端 B：检查并评分

```bash
# 只读列出全部结果及其当前状态；不检查 judge，不评分。
bash scripts/msmu/score_pending_results.sh --list

# 检查目录、dataset、批次锁和 judge readiness；不评分。
bash scripts/msmu/score_pending_results.sh --check

# 可选：预览本次实际评分顺序和输出目录；不调用 judge/scorer。
MANUAL_DRY_RUN=1 bash scripts/msmu/score_pending_results.sh

# 正式串行评分；这是唯一会调用 judge/scorer 的命令。
bash scripts/msmu/score_pending_results.sh

# 只读汇总评分后的各状态数量；确认 pending 为 0。
bash scripts/msmu/score_pending_results.sh --status
```

`--status` 中 `pending` 为 `0` 即没有待评分结果。中断或失败后，保持 judge 正常运行并重新执行同一条
评分命令：

```bash
bash scripts/msmu/score_pending_results.sh
```

评分完成后，在终端 A 按 `Ctrl-C` 停止本任务启动的 judge。

单模型一键入口内部会把刚解析出的绝对 prediction 路径传给同一个目录驱动评分器：

```bash
bash scripts/msmu/score_pending_results.sh \
  --results-root /阶段三/绝对结果根 \
  --predictions /该模型/协议目录/predictions.jsonl
```

`--predictions` 不按模型名称筛选，也不绕过批次锁、judge identity、完整 validator 或 publication
gates；它只把冻结候选从结果根中的全部 prediction 收窄为这个精确文件。日常批量评分仍省略该参数，
继续发现结果根下全部待评分轨。

## 汇总评分结果

```bash
# 只读列出全部 scorer protocol 下已有 summary 及其可收录状态。
bash scripts/msmu/build_results_report.sh --list

# 收录当前 canonical scorer protocol 下全部通过 publication gates 的评分结果。
bash scripts/msmu/build_results_report.sh

# 按模型 profile 精确收录；profile 参数可重复，scorer protocol 只能指定一个。
bash scripts/msmu/build_results_report.sh \
  --profile PROFILE \
  --scorer-protocol SCORER_PROTOCOL
```

默认写入：

```text
/media/datasets/tangzecong/latent_reasoning/msmu-outputs/manual-three-stage-v1/03_full987/msmu-result.md
```

## 使用其他结果根目录

只接受绝对路径：

```bash
bash scripts/msmu/score_pending_results.sh \
  --results-root /其他/绝对路径

bash scripts/msmu/build_results_report.sh \
  --results-root /其他/绝对路径 \
  --output /其他/绝对路径/msmu-result.md
```
