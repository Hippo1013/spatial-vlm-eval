# ADR-0004: SPBench-SI 主结果使用原始 MRA、稳健 parser v2 与独立上游审计

- Status: Accepted
- Date: 2026-08-07
- Amended: 2026-08-08

## Context

SPBench-SI 的原始 MRA 定义在十个阈值上使用严格不等式，而当前 SpatialLadder direct-mode 代码使用
inclusive `<=`，并采用宽松的 first-match 数字词/数字和选择题提取。直接把当前代码输出作为唯一主分
会把上游实现偏差与 benchmark 定义混为一谈；只保留稳健主分又不利于定位与上游表格的差异。

首批 19 条完整轨与 Gemini 部分输出的只读审计进一步暴露了 v1 主 parser 的实际歧义：自由文本或选项
标签中的 `a/A` 会被提取成 1；`<answer>A. 20</answer>`、明确的最终 distance，以及先列尺寸再声明
longest dimension 的回答可能反而失败；同一回答显式提供 meter/centimeter 时，纯数值冲突也无法忠实
选择题干要求的单位。改用上游 first-match 不能解决这些问题，因为 audit 的职责是复刻锁定 commit。

## Decision

当前主 protocol 是：

```text
spbench_si_original_mra10_strict_robust_direct_controlled_final_expected_unit_four_task_macro_v2
```

它继续使用 tag/final-area 优先、冲突 fail-closed 的选择题/数值解析、Decimal 相对误差，以及
`θ=0.50..0.95` 十阈值严格 `<` MRA；四题型分别平均，NQ/MCQ 各自两类等权，Overall 为四题型宏平均。
v2 parser 进一步锁定：

- 主数字词只接受明确的 `zero..ninety` 单词，不把自由文本 `a/an` 解释为 1。
- 只有唯一 `<answer>` 或显式 final-answer 强区域可剥离开头的标准 `A-D.` 标签；普通全文仍 fail closed。
- 没有强区域时，只识别受控的最后 distance 等式/声明、longest-dimension 声明和
  `provide ... as the longest dimension`，不采用任意“最后一个数字”策略。
- absolute-distance 的期望单位为 meter，size-estimation 为 centimeter。回答若显式写出期望单位数值，
  只用这些数值做唯一性判断；否则保留唯一数值语义。单位仅用于选择，不做换算。
- range、上下界、负数、非有限值、多值冲突继续失败，并保留逐行 parse evidence。

每次评分另在独立目录生成
`spbench_si_upstream_7a0d2ee_default_direct_compat_v1`，精确保留锁定 commit 当前 direct-mode 的提取、
inclusive 边界和聚合。两套逐行结果、summary 和 protocol identity 不混表；报告第二表必须明确标作
upstream compatibility audit。

评分前强制 full validator，报告再检查 dataset/profile/revision/prompt/decoding/input track、prediction
与 scored-row hash 和 publication gates。同 profile 多候选拒绝自动选择。

v1/v2 scorer 不共享正式 score 目录。v1 inference metadata 被列为兼容输入，因为 parser-only 升级不改变
prediction、prompt、图像、processor 或 decoding；v2 summary 和 publication gate 必须记录并验证原声明。

## Consequences

- 恰落在 MRA 边界或包含前置推理数字的输出可能在主分与 audit 间不同，差异会被逐行保存。
- 主结果标记为 original MRA definition + robust direct parser internal score，不宣称逐字节复刻上游。
- 现有完整 raw prediction 可由 v2 重新评分，不因 parser-only 升级而要求重跑模型；v1 主分不再是当前
  发布结果。
- SpatialLadder 的 prompt 片段/占位符输出属于独立推理故障，不能由 parser 放宽来补救。
- parser、阈值、聚合或兼容实现身份变化必须更换对应 protocol id，并更新真实输出回归、本文和
  canonical protocol。
- 暂行报告只允许 20/21 且唯一缺少固定四卡的 InternVL3-78B；其他部分状态不能发布。

## References

- [SPBench-SI canonical protocol](../benchmarks/spbench_si/protocol.md)
- [SPBench-SI 两阶段 runbook](../spbench-si-two-stage-runbook.md)
- [`scorer.py`](../../src/spatial_vlm_eval/benchmarks/spbench_si/scorer.py)
- [`test_scorer.py`](../../tests/benchmarks/spbench_si/test_scorer.py)
