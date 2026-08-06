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
