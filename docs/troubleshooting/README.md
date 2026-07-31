# Troubleshooting Knowledge Base

这里保存已经定位、修复并验证的可复用问题，不是逐日开发日志，也不替代原始运行日志。

## 问题路由

- Ubuntu 服务器环境、部署和评测运行：[server.md](server.md)。
- 未定位问题：保留在未跟踪输出目录的 infer/batch/score 日志或 issue 中；确认根因前不要写入本目录。
- 协议或架构取舍：写入 [ADR](../decisions/README.md)，不要混入故障条目。
- 用户可见行为变化：同时更新根目录 [CHANGELOG](../../CHANGELOG.md)。

## 写入规则

- 新条目置顶；同类问题合并更新，不重复记录。
- 必须包含场景、可识别症状、根因、实际处理和验证结果。
- 只保留可复用结论，不粘贴完整 traceback；完整日志只记录未跟踪路径。
- 修复代码必须有回归测试；条目与修复在同一提交完成。
- 不记录 token、密码、私有 endpoint 或其他敏感信息。

运行时原始证据通常位于：

```text
predictions.infer.log
predictions.jsonl.journal.jsonl
03_full987/_serial_inference/logs/TIMESTAMP.log
scores/SCORER_PROTOCOL/score.log
scores/SCORER_PROTOCOL/judge_failures.jsonl
03_full987/_serial_scoring/SCORER_PROTOCOL/runs/TIMESTAMP.log
```
