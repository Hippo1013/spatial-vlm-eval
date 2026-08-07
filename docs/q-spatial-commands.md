# Q-Spatial 简明运行指令

先把 [`configs/q-spatial-server.env.example`](../configs/q-spatial-server.env.example) 中实际可用的值合并到
未跟踪 `.env.server`。正式输出必须位于仓库外。

```bash
bash scripts/q_spatial/run_inference.sh --list
bash scripts/q_spatial/run_inference.sh --check --model qwen3_vl_8b

bash scripts/q_spatial/run_inference.sh --stage test --model qwen3_vl_8b
bash scripts/q_spatial/run_inference.sh --stage test --models qwen3_vl_2b,qwen3_vl_4b
bash scripts/q_spatial/run_inference.sh --stage test --all --skip-resource-blocked

bash scripts/q_spatial/run_inference.sh --stage full --model qwen3_vl_8b
bash scripts/q_spatial/run_inference.sh --stage full --models qwen3_vl_2b,qwen3_vl_4b
bash scripts/q_spatial/run_inference.sh --stage full --all --skip-resource-blocked
bash scripts/q_spatial/run_inference.sh --status
```

`--all` 始终是 21 轨；只有显式 `--skip-resource-blocked` 才允许把四卡 78B 写成
`BLOCKED_RESOURCE` 后继续。API test/full 会产生真实费用，必须在执行前得到用户明确批准并通过环境变量
注入 key。

四卡服务器补齐 `internvl3_78b` 时使用独立全链路入口；它沿用现有输出根和报告，不加入双卡 20 轨
调度：

```bash
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --check
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --dry-run
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --faq
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh
```

入口从当前 registry/scorer 解析最新版协议，精确评分该轨后原地重建既有 `q-spatial-result.md`。迁移、
burn、恢复与 21/21 验收见[InternVL3-78B 四卡补测](q-spatial-internvl3-78b-evaluation.md)。

双卡服务器的 20 轨冻结计划优先使用分阶段调度器；前三条命令都不启动模型或 API：

```bash
bash scripts/q_spatial/run_scheduled_batch.sh --list
bash scripts/q_spatial/run_scheduled_batch.sh --check
bash scripts/q_spatial/run_scheduled_batch.sh --dry-run
bash scripts/q_spatial/run_scheduled_batch.sh --stage test --dry-run

# 只完成或复用 20 轨 test gate；不会进入 full 或评分。
bash scripts/q_spatial/run_scheduled_batch.sh \
  --stage test --without-internvl78 --with-paid-api

# 完整 test/full 执行；不会评分。
bash scripts/q_spatial/run_scheduled_batch.sh \
  --stage full --without-internvl78 --with-paid-api --skip-completed
```

阶段 A 双卡 lane 与串行 API lane 并行；双卡成功后立即启动阶段 B 的 GPU 0/1 两 lane，不等待 API。
状态与冻结计划在 `$QSPATIAL_OUTPUT_ROOT/_scheduled_batch/`。独立 watcher 只报告高层状态/异常：

```bash
bash scripts/q_spatial/watch_scheduled_health.sh --lane dual
bash scripts/q_spatial/watch_scheduled_health.sh --lane api
bash scripts/q_spatial/watch_scheduled_health.sh --lane gpu0
bash scripts/q_spatial/watch_scheduled_health.sh --lane gpu1
```

```bash
bash scripts/q_spatial/validate_predictions.sh \
  --predictions /absolute/path/predictions.jsonl

bash scripts/q_spatial/score_results.sh --predictions /absolute/path/predictions.jsonl
bash scripts/q_spatial/score_results.sh --list
bash scripts/q_spatial/score_results.sh
bash scripts/q_spatial/score_results.sh --status

bash scripts/q_spatial/build_results_report.sh --check
bash scripts/q_spatial/build_results_report.sh
```

完整 endpoint、gate、恢复和产物说明见[两阶段 runbook](q-spatial-two-stage-runbook.md)；评分语义只以
[canonical protocol](benchmarks/q_spatial/protocol.md)为准。
