# ADR-0004: SPBench-SI 主结果使用原始 MRA 严格边界并分离上游兼容审计

- Status: Accepted
- Date: 2026-08-07

## Context

SPBench-SI 的原始 MRA 定义在十个阈值上使用严格不等式，而当前 SpatialLadder direct-mode 代码使用
inclusive `<=`，并采用宽松的 first-match 数字词/数字和选择题提取。直接把当前代码输出作为唯一主分
会把上游实现偏差与 benchmark 定义混为一谈；只保留稳健主分又不利于定位与上游表格的差异。

## Decision

主 protocol `spbench_si_original_mra10_strict_robust_direct_four_task_macro_v1` 使用 tag/final-area 优先、
冲突 fail-closed 的选择题/数值解析，Decimal 相对误差，以及 `θ=0.50..0.95` 十阈值严格 `<` MRA。
四题型分别平均，NQ/MCQ 各自两类等权，Overall 为四题型宏平均。

每次评分另在独立目录生成
`spbench_si_upstream_7a0d2ee_default_direct_compat_v1`，精确保留锁定 commit 当前 direct-mode 的提取、
inclusive 边界和聚合。两套逐行结果、summary 和 protocol identity 不混表；报告第二表必须明确标作
upstream compatibility audit。

评分前强制 full validator，报告再检查 dataset/profile/revision/prompt/decoding/input track、prediction
与 scored-row hash 和 publication gates。同 profile 多候选拒绝自动选择。

## Consequences

- 恰落在 MRA 边界或包含前置推理数字的输出可能在主分与 audit 间不同，差异会被逐行保存。
- 主结果标记为 original MRA definition + robust direct parser internal score，不宣称逐字节复刻上游。
- parser、阈值、聚合或兼容实现身份变化必须更换对应 protocol id，并更新测试、本文和 canonical
  protocol。
- 暂行报告只允许 20/21 且唯一缺少固定四卡的 InternVL3-78B；其他部分状态不能发布。

## References

- [SPBench-SI canonical protocol](../benchmarks/spbench_si/protocol.md)
- [SPBench-SI 两阶段 runbook](../spbench-si-two-stage-runbook.md)
