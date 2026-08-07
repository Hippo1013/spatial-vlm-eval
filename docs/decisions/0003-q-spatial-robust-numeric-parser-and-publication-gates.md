# ADR-0003: Q-Spatial 使用 declared-final robust numeric parser 与独立发布门禁

- Status: Accepted
- Date: 2026-08-04
- Last amended: 2026-08-07 (parser v2)

## Context

Q-Spatial 论文指标需要把自由文本答案换算为统一长度单位。官方 notebook 取最后一组标签、对 scalar
中的多个数字求平均、把未知单位当厘米，并使用严格 `<`；这些容错会把冲突或 malformed 输出计为有效，
而论文公式写作 `δ≤2`。

初始 v1 采用任何标签痕迹都禁止 fallback 的严格 tag-first 方案。20 条 full-271 的 5,420 个真实输出
表明该方案会系统性漏掉等价重复标签、unit-only 标签、紧凑单位和 LaTeX wrapper，还会把 `PS4`、
`Region [0]` 中的数字误当竞争答案；这些是假阴性，不是模型答案歧义。

## Decision

主 scorer 使用独立 v2 protocol。解析器优先使用最后明确声明区域；等价重复标签折叠，计算过程后的唯一
final 标签可覆盖前文中间量。受控支持相邻、反斜杠、brace、comma、boxed、distance、diameter/unit
wrapper，以及有限十进制和简单分数。对象/区域标识数字不参与答案竞争，raw prediction 永不改写。

冲突标签、范围、复合表达式、多候选、缺单位、拒答继续 fail closed。零、负数和未知单位保留模型声明
用于审计，但不进入正值物理距离主分。Decimal 换算后按论文 inclusive `δ≤2` / `δ≤1.25` 评分；Overall
对 ScanNet 和 Q-Spatial++ 等权。

推理与评分协议保持分离：v2 scorer 仅消费声明 v1 或 v2 scorer ID 的完整 inference metadata，并在
summary 记录原声明；既有 prediction 不修改、不重跑，未知声明仍拒绝。同次运行继续生成不影响主分的
旧 notebook 审计产物并保留完整 publication gates，但面向使用者的 Markdown 汇总只展示当前 v2
scorer 的主指标和 `δ≤1.25` 严格阈值，不再展示旧解析方式。精简主表沿用 MSMU 命名方式，把实际
派生输入配置写入模型名括号，不再单列 input track/comparison group；内部 provenance 与分组加粗规则
保持不变。

## Consequences

- 当前 5,420 条输出中，4,895 条成为合法正物理量；另外 198 条明确的零或未知单位答案保留声明值但
  继续计零。其余输出是范围、多候选、缺单位、拒答、截断或不相关格式。
- v1 与 v2 分数不得混用；v2 使用独立 score 目录、summary、publication gates 和报告身份。
- 结果仍是 official formula + robust numeric parser internal score，不宣称逐字节复刻官方 notebook。
- 解析、单位、阈值、split macro 或发布身份再次变化时，必须更换 scorer protocol 并补回归测试。

## References

- [Q-Spatial canonical protocol](../benchmarks/q_spatial/protocol.md)
- [Q-Spatial 两阶段 runbook](../q-spatial-two-stage-runbook.md)
- [`scorer.py`](../../src/spatial_vlm_eval/benchmarks/q_spatial/scorer.py)
