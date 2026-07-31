# MSMU 遗留小问题

本文件只记录当前评测流程中已知、暂不修复的小问题，供以后人工复核。

## Malformed judge response 统一记零

适用 protocol：

```text
sdvlm_official_compat_local_judge_v4_grounding_split_strict_quant_length_malformed_zero
```

本地 judge 偶尔会返回语义上可能可恢复、但不满足当前 JSON/schema 约束的文本，例如带注释的
JSON、值位置上的算术表达式、未定义变量、多个对象后跟解释文字，或被后续 LaTeX 大括号干扰的对象。
当前 scorer 在完成重试后不继续扩展宽松解析，而是把所有 judge 路径中的这类样本统一记为 0 分并
继续整批评分。

这个选择便于批量测试，不代表这些被测模型回答必然错误。部分回答经过更宽松、任务特定或人工解析
后可能得到分数，因此当前结果可能存在小幅保守低估。

人工复核时可检查：

- `summary.json` 中的 `num_malformed_judge_zero_fallbacks` 和
  `malformed_judge_zero_fallback_indices`；
- `scored_rows.jsonl` 中 `judge_fallback == "malformed_judge_response_zero"` 的行；
- 对应 cache 行保留的 `__raw_content__` 与 `__parse_error__`。

完全没有收到 judge 文本的连接失败、超时或未处理 worker 异常不属于本问题，仍会进入
`judge_failures.jsonl` 并阻断 publication。

如果以后改为结构化输出、平衡括号提取或安全表达式求值，必须重新审计所有 judge 路径，并更换
scorer/cache protocol id 后重新评分；不得回写或覆盖当前 protocol 的历史结果。
