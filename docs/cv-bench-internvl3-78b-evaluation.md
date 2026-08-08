# CV-Bench InternVL3-78B 一键完整评测

本页用于用四张 80GB GPU 补齐 `internvl3_78b`。入口自动完成 test gate、full-2638、独立校验、
精确单轨评分和原有全局报告重建。

该 profile 的 served name 已统一为 `internvl3-78b-three-bench`。若还要补齐 Q-Spatial 与 SPBench-SI，
优先按[三 Benchmark 一键测评](internvl3-78b-three-bench-evaluation.md)共享一次模型加载；本页的
单 benchmark 入口继续保留并写入同一 canonical CV-Bench 路径。

## 运行前准备

在服务器仓库根目录执行：

```bash
export CVBENCH_ENV_FILE="$PWD/.env.cvbench.server"
```

环境文件至少包含：

```bash
CVBENCH_DATASET_ROOT=/path/to/locked/CV-Bench
CVBENCH_OUTPUT_ROOT=/path/to/existing/cv-bench-outputs
INTERNVL3_78B_MODEL=/path/to/locked/InternVL3-78B-hf/snapshot
CVBENCH_PYTHON=/path/to/vllm-0.19-python
CVBENCH_VLLM=/path/to/vllm
CVBENCH_INTERNVL3_78B_GPU_IDS=0,1,2,3
```

`CVBENCH_OUTPUT_ROOT` 必须指向已有 22 轨结果所在的标准输出根。四张 GPU 必须各有至少 79000 MiB
总显存、76000 MiB 空闲显存且无 compute process；脚本不会停止 burn 或其他任务。

## 一键运行

```bash
# 只读检查模型、数据、四卡、端口和路径
bash scripts/cv_bench/run_internvl3_78b_evaluation.sh --check

# 完整执行
bash scripts/cv_bench/run_internvl3_78b_evaluation.sh
```

执行顺序：

```text
test gate -> full-2638 -> validator -> 释放自有 vLLM
-> 只评分 internvl3_78b -> 重建 cv-bench-result.md
```

可选 tmux：

```bash
tmux new-session -d -s cvbench-internvl3-78b \
  "cd '$PWD' && export CVBENCH_ENV_FILE='$CVBENCH_ENV_FILE' && bash scripts/cv_bench/run_internvl3_78b_evaluation.sh"
tmux attach -t cvbench-internvl3-78b
```

## 状态、预演与恢复

```bash
bash scripts/cv_bench/run_internvl3_78b_evaluation.sh --status
bash scripts/cv_bench/run_internvl3_78b_evaluation.sh --dry-run
```

中断后重新执行无参数命令即可。当前 test gate、journal、完整 prediction 和完整 score 会复用；已有
完整 prediction 重新通过 validator 时不会加载模型。

## 标准产物

正式产物不另开目录：

```text
$CVBENCH_OUTPUT_ROOT/
├── runs/internvl3_78b/3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356/
│   └── cv_bench_internvl3_78b_rgb_official_decode_v1/
│       ├── test_gate.json
│       ├── predictions.jsonl
│       ├── predictions.jsonl.metadata.json
│       ├── prediction_validation.json
│       └── scores/SCORER_PROTOCOL/
│           ├── scored_rows.jsonl
│           ├── summary.json
│           └── publication_gates.json
└── cv-bench-result.md
```

报告器从原有 22 条 summary 和新结果重建同一个 `cv-bench-result.md`，新增 InternVL3-78B 一行。
控制器日志位于 `_single_model_evaluation/logs/`，不属于正式结果。

## 完成核验

```bash
bash scripts/cv_bench/run_internvl3_78b_evaluation.sh --status
bash scripts/cv_bench/build_results_report.sh --check
```

目标状态应为 full validator 与 score publication gate 均通过；报告 `missing` 为空并显示 `23/23`。

## 手工备用

仅在诊断时拆分执行：

```bash
export CVBENCH_INTERNVL3_78B_GPU_IDS=0,1,2,3
export CVBENCH_INTERNVL3_78B_BASE_URLS=http://127.0.0.1:18101/v1

# 终端 A
bash scripts/cv_bench/serve_vllm_profile.sh \
  --model internvl3_78b --gpu-ids 0,1,2,3 --port 18101

# 终端 B
bash scripts/cv_bench/run_inference.sh --stage test --model internvl3_78b
bash scripts/cv_bench/run_inference.sh --stage full --model internvl3_78b
```

服务停止后，再按 [CV-Bench 简明运行指令](cv-bench-commands.md#6-校验)执行校验、指定结果评分和汇总。
