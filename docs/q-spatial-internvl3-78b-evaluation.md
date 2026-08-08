# Q-Spatial InternVL3-78B 四卡补测

本页用于在四张 80GB GPU 上补齐 `internvl3_78b`。入口沿用现有 Q-Spatial 正式输出根：自动完成
test gate、full-271、validator、精确单轨评分，并把原有 `q-spatial-result.md` 从 20/21 原地重建为
21/21；不会建立另一套 78B 正式结果根。

入口不硬编码旧 protocol id：track 由当前 profile/revision/inference protocol 解析，test gate 必须匹配
当前完整 binding，评分目录与报告直接读取当前 `SCORER_PROTOCOL`。已有 prediction 只有在 inference、
scorer 声明、dataset、revision、binding、gate 和 artifact hash 全部仍为当前值时才可复用。

该 profile 的 served name 已统一为 `internvl3-78b-three-bench`。若还要补齐 SPBench-SI 与 CV-Bench，
优先按[三 Benchmark 一键测评](internvl3-78b-three-bench-evaluation.md)共享一次模型加载；本页的
单 benchmark 入口仍可独立使用并写入同一 Q-Spatial canonical 路径。

## 1. 迁移准备

先把当前仓库和已有 20 条正式结果完整同步到四卡服务器。`QSPATIAL_OUTPUT_ROOT` 必须指向这份现有
20/21 输出根，而不是新空目录。在仓库根设置未跟踪环境文件：

```bash
export QSPATIAL_ENV_FILE="$PWD/.env.server"
```

至少确认：

```bash
QSPATIAL_PARQUET_ROOT=/path/to/locked/Q-Spatial-Bench
QSPATIAL_SCANNET_RGB_ROOT=/path/to/authorized/QSpatial_scannet/images
QSPATIAL_OUTPUT_ROOT=/path/to/existing/q-spatial-outputs
INTERNVL3_78B_MODEL=/path/to/locked/InternVL3-78B-hf/snapshot
QSPATIAL_PYTHON=/path/to/benchmark-python
QSPATIAL_VLLM=/path/to/vllm
QSPATIAL_INTERNVL3_78B_GPU_IDS=0,1,2,3
QSPATIAL_INTERNVL3_78B_PORT=18101
```

四张 GPU 必须各有至少 79,000 MiB 总显存、76,000 MiB 空闲显存且无 compute process。入口不会
停止 burn、未知服务或其他任务；运行前按 [GPU burn 手册](server-gpu-burn-runbook.md)人工停止目标
四卡的 burn，整个补测结束后恢复。

## 2. 检查与运行

```bash
# 只读检查现有 20/21 结果、数据、模型、四卡、端口、锁、processor/template 和 binding
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --check

# 完整补测
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh
```

固定顺序：

```text
test gate -> full-271 -> validator -> 释放自有 vLLM
-> 只评分 internvl3_78b -> 原地重建 q-spatial-result.md -> 验证 21/21
```

建议在 tmux 中运行：

```bash
tmux new-session -d -s qspatial-internvl3-78b \
  "cd '$PWD' && export QSPATIAL_ENV_FILE='$QSPATIAL_ENV_FILE' && bash scripts/q_spatial/run_internvl3_78b_evaluation.sh"
tmux attach -t qspatial-internvl3-78b
```

## 3. 状态、预演与恢复

```bash
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --status
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --dry-run
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --faq
```

中断后重新执行无参数命令即可。同一 binding 的 test gate、fsync journal、完整 prediction 和 complete
score 会复用；已有完整 prediction 重新通过 validator 时不会加载模型。`--status` 的完成目标是
`full_validator=passed`、score `complete`、`report-completeness=21/21` 且 `missing` 为空。

## 4. 输出与验收

正式产物继续写入原有 canonical 位置：

```text
$QSPATIAL_OUTPUT_ROOT/
├── runs/internvl3_78b/LOCKED_REVISION/INFERENCE_PROTOCOL/
│   ├── test_gate.json
│   ├── predictions.jsonl
│   ├── predictions.jsonl.metadata.json
│   ├── prediction_validation.json
│   └── scores/CURRENT_SCORER_PROTOCOL/
│       ├── scored_rows.jsonl
│       ├── summary.json
│       └── publication_gates.json
├── _single_model_evaluation/logs/   # 控制日志，不参与正式汇总
└── q-spatial-result.md              # 原有报告原地增加 InternVL3-78B
```

最终复核：

```bash
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --status
bash scripts/q_spatial/build_results_report.sh --check
```

报告应显示全轨 `21/21`、缺失 profile 为无。脚本只终止自身记录的进程组；端口或 GPU 忙时会在加载
模型前 fail closed，不会接管或清理外部进程。
