# SPBench-SI InternVL3-78B 四卡完整评测

本页用于把两卡服务器上固定缺失的 `internvl3_78b` 迁移到四张 80GB GPU 补齐。独立入口自动完成
test gate、full-1009、独立 validator、主协议与 upstream compatibility audit 精确评分，并把既有
`spbench-si-result.md` 从暂行 20/21 原地重建为完整 21/21。

该 profile 的 served name 已统一为 `internvl3-78b-three-bench`。若还要补齐 Q-Spatial 与 CV-Bench，
优先按[三 Benchmark 一键测评](internvl3-78b-three-bench-evaluation.md)共享一次模型加载；本页的
单 benchmark 入口继续保留并写入同一 SPBench-SI canonical 路径。

## 运行前准备

在四卡服务器仓库根目录设置未跟踪环境文件：

```bash
export SPBENCH_SI_ENV_FILE="$PWD/.env.spbench-si.server"
```

至少确认以下变量：

```bash
SPBENCH_SI_PARQUET=/path/to/locked/SPBench-SI.parquet
SPBENCH_SI_IMAGES_ARCHIVE=/path/to/locked/SPBench-SI-images.zip
SPBENCH_SI_OUTPUT_ROOT=/path/to/existing/spbench-si-outputs
INTERNVL3_78B_MODEL=/path/to/locked/InternVL3-78B-hf/snapshot
SPBENCH_SI_PYTHON=/path/to/benchmark-python
SPBENCH_SI_VLLM=/path/to/vllm-0.19
SPBENCH_SI_VLLM_RUNTIME_VERSION=0.19.0
SPBENCH_SI_INTERNVL3_78B_GPU_IDS=0,1,2,3
SPBENCH_SI_INTERNVL3_78B_PORT=18102
```

`SPBENCH_SI_OUTPUT_ROOT` 必须指向既有 20 轨结果所在的同一个标准输出根，不能改成 78B 专属目录。
四张 GPU 必须各有至少 79,000 MiB 总显存、76,000 MiB 空闲显存且无 compute process。脚本不会停止
burn、未知服务或其他任务。最终报告门禁会要求该根从合法 20/21 补齐为 21/21；若 78B 曾留下可恢复的
partial score，入口仍允许原地继续修复，而不会被一次前置报告检查阻塞。

## 一键运行

```bash
# 只读检查数据、模型、四卡、端口、锁、processor/template 与路径
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --check

# 完整执行
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh
```

执行顺序：

```text
test gate -> full-1009 -> validator -> 释放自有 vLLM
-> 只评分 internvl3_78b 的主协议 + compatibility audit
-> 重建 spbench-si-result.md
```

可选 tmux：

```bash
tmux new-session -d -s spbench-internvl3-78b \
  "cd '$PWD' && export SPBENCH_SI_ENV_FILE='$SPBENCH_SI_ENV_FILE' && bash scripts/spbench_si/run_internvl3_78b_evaluation.sh"
tmux attach -t spbench-internvl3-78b
```

## 状态、预演与恢复

```bash
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --status
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --dry-run
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --faq
```

中断后重新执行无参数命令即可。同一 binding 的 test gate、fsync journal、完整 prediction 和完整双协议
score 会复用；已有完整 prediction 重新通过 validator 时不会加载模型。

## 常见问题

**问：迁移到四卡服务器后第一步是什么？**

答：先运行 `--check`。它只读检查锁定数据、模型路径、四卡空闲、端口、运行锁和当前 processor/binding；
任一条件不满足都会在加载模型前停止。

**问：脚本会为了腾卡停止 burn 或未知进程吗？**

答：不会。端口被占用或任一 GPU 有 compute process 时 fail closed；信号处理只终止本控制器记录的
vLLM/步骤进程组。

**问：中断或 SSH 断开后是否从头付出全部计算？**

答：不会。推荐在 tmux 中运行；即使控制器中断，重跑同一命令也会从合法 journal 恢复，已经完成并
重新通过 validator 的 prediction 会直接进入评分。

**问：是否会另建一套 78B 专属正式结果根？**

答：不会。prediction、validator、两套 score 和报告都使用原有 canonical 路径；控制日志仅写入
`$SPBENCH_SI_OUTPUT_ROOT/_single_model_evaluation/logs/`。

**问：为什么先停 vLLM 再评分？**

答：SPBench-SI 评分是确定性本地 scorer，不需要被测模型；先释放四卡能明确隔离推理和评分资源，且
保证控制器不会在评分阶段继续占卡。

## 标准产物

```text
$SPBENCH_SI_OUTPUT_ROOT/
├── runs/internvl3_78b/LOCKED_REVISION/INFERENCE_PROTOCOL/
│   ├── test_gate.json
│   ├── predictions.jsonl
│   ├── predictions.jsonl.metadata.json
│   ├── prediction_validation.json
│   └── scores/
│       ├── spbench_si_original_mra10_strict_robust_direct_controlled_final_expected_unit_four_task_macro_v2/
│       │   ├── scored_rows.jsonl
│       │   ├── summary.json
│       │   └── publication_gates.json
│       └── SPBENCH_UPSTREAM_AUDIT_PROTOCOL/
│           ├── scored_rows.jsonl
│           └── summary.json
└── spbench-si-result.md
```

## 完成核验

```bash
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --status
bash scripts/spbench_si/build_results_report.sh --check
```

目标状态是 full validator 与双协议 publication provenance 全部通过；报告显示 `21/21`、`missing` 为空。

## 手工备用

仅在诊断时拆分执行：

```bash
export SPBENCH_SI_INTERNVL3_78B_GPU_IDS=0,1,2,3
export SPBENCH_SI_INTERNVL3_78B_BASE_URLS=http://127.0.0.1:18102/v1

# 终端 A
bash scripts/spbench_si/serve_vllm_profile.sh \
  --model internvl3_78b --gpu-ids 0,1,2,3 --port 18102

# 终端 B
bash scripts/spbench_si/run_inference.sh --stage test --model internvl3_78b
bash scripts/spbench_si/run_inference.sh --stage full --model internvl3_78b
```

服务停止后，再运行 validator、`score_results.sh --predictions` 与报告入口。手工模式不得绕过 test gate
或把 InternVL3-78B 降为 TP=2/量化替代。
