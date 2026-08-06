# ADR-0003: Q-Spatial 使用 tag-first robust numeric parser 与独立发布门禁

- Status: Accepted
- Date: 2026-08-04

## Context

Q-Spatial 论文指标需要把自由文本答案换算为统一长度单位。官方 notebook 取最后一组标签、对 scalar
中的多个数字求平均、把未知单位当厘米，并使用严格 `<`；这些容错会把冲突或 malformed 输出计为有效，
而论文公式写作 `δ≤2`。模型输出还可能没有标签，但在最终答案行给出唯一数字与单位。

## Decision

主 scorer 使用独立 protocol id，采用 tag-first 解析：任何标签痕迹都禁止 fallback；完整标签必须唯一、
正有限十进制且单位已知。完全无标签时，只在优先级锁定的最终区域接受唯一相邻 numeric-unit pair。
Decimal 换算后按论文 inclusive `δ≤2` / `δ≤1.25` 评分；Overall 对 ScanNet 和 Q-Spatial++ 等权。

同次运行生成不影响主分的旧 notebook 审计列。评分前强制 full validator，评分和报告再校验 dataset、
profile、revision、prompt、decoding、input track、prediction/scored-row hash 与 publication gates。同一
profile 有多个候选时拒绝选择。

## Consequences

- 主分不再受多数字、未知单位或 partial tag 的宽松解释影响；边界 ratio 恰为 1.25/2 时与旧审计可能
  不同，差异会逐行保存。
- 结果是 official formula + robust numeric parser internal score，不宣称逐字节复刻官方 notebook。
- 解析、单位、阈值、split macro 或发布身份变化必须更换 scorer protocol 并补回归测试。

## References

- [Q-Spatial canonical protocol](../benchmarks/q_spatial/protocol.md)
- [Q-Spatial 两阶段 runbook](../q-spatial-two-stage-runbook.md)
