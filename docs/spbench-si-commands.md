# SPBench-SI 简明运行指令

完整边界见 [两阶段 runbook](spbench-si-two-stage-runbook.md)与
[canonical protocol](benchmarks/spbench_si/protocol.md)。以下命令不会替代 GPU/API/评分授权。

```bash
# 公共只读检查
bash scripts/spbench_si/run_inference.sh --list
bash scripts/spbench_si/run_scheduled_batch.sh --list
bash scripts/spbench_si/run_scheduled_batch.sh --check
bash scripts/spbench_si/run_scheduled_batch.sh --dry-run

# 单轨：test gate 后才能 full
bash scripts/spbench_si/run_inference.sh --stage test --model qwen3_vl_8b
bash scripts/spbench_si/run_inference.sh --stage full --model qwen3_vl_8b
bash scripts/spbench_si/validate_predictions.sh \
  --predictions /absolute/path/to/predictions.jsonl

# 闭源补充轨，当前搁置；仅在新的明确需求和付费授权后恢复以下续接流程
bash scripts/spbench_si/set_packyapi_key.sh
source /media/datasets/lihaoran/tools/mihomo/proxy-on.sh
bash scripts/spbench_si/run_gemini_packyapi_resume.sh

# 双卡 20 轨（InternVL3-78B 明确排除；API 必须另行获准）
bash scripts/spbench_si/run_scheduled_batch.sh --stage test --dry-run
bash scripts/spbench_si/run_scheduled_batch.sh \
  --stage test --without-internvl78 --with-paid-api
bash scripts/spbench_si/run_scheduled_batch.sh \
  --stage full --without-internvl78 --with-paid-api

# 只读 watcher
bash scripts/spbench_si/watch_scheduled_health.sh \
  --control-root "$SPBENCH_SI_OUTPUT_ROOT/_scheduled_batch" \
  --run-id RUN_ID --lane gpu1

# 四卡 InternVL3-78B 独立全链路与内置 FAQ
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --check
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --dry-run
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --status
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --faq

# 正式评分必须单独授权
bash scripts/spbench_si/score_results.sh --check
bash scripts/spbench_si/score_results.sh --dry-run
bash scripts/spbench_si/score_results.sh --predictions /absolute/path/to/predictions.jsonl
bash scripts/spbench_si/build_results_report.sh
bash scripts/spbench_si/build_results_report.sh \
  --exclude-profile internvl3_78b \
  --exclude-profile gemini31pro_openrouter_non_zdr
```

双卡调度器不自动评分。正式产物只写入仓库外 `SPBENCH_SI_OUTPUT_ROOT`；subset 不评分。报告可以是
任意非空的 publication-gated 结果子集，并显式记录 `--exclude-profile` 排除项与其余缺失。

四卡入口仍在同一 `SPBENCH_SI_OUTPUT_ROOT` 只评分目标 prediction 的主协议与 compatibility audit，
再调用同一报告发现器；若全部轨已完成则为 21/21，否则保留明确的部分汇总状态。详细说明见
[InternVL3-78B 四卡完整评测](spbench-si-internvl3-78b-evaluation.md)。
