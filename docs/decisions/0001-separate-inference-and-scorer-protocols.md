# ADR-0001: 分离 inference protocol 与 scorer/cache protocol

- Status: Accepted
- Date: 2026-07-26
- Supersedes: none

## Context

MSMU 横评包含不同模型、prompt、图像派生组件和 decoding。它们可以使用同一套 judge、阈值与
macro-8 聚合，但模型实际看到的输入和生成方式并不相同。若只使用一个 protocol 字段，这些轨会被
错误混表；若每次增加模型都更换 scorer/cache protocol，又会无意义地废弃合法 judge cache。

## Decision

- 每个输入、prompt、图像处理或 decoding 组合使用独立 `inference_protocol`。
- judge prompt、grounding 路由、阈值、列表长度、聚合与 cache identity 由独立
  `scorer/cache protocol` 标识。
- 正式结果目录和机器 provenance 同时携带两个 protocol。精简人类展示表可以省略 protocol 列，但
  必须先逐行校验 inference/scorer provenance、一次只选择一个 scorer protocol，并在模型名称中
  区分不同 input track；机器产物仍是审计真值。
- 改变 judge response 或评分语义时更换相应 scorer/cache id。仅修复不改变合法 judge response 的
  确定性后处理时，可以在回归测试和 protocol 文档说明后保留 cache id。

具体 protocol 值、阈值与 cache key 字段仍以
[MSMU canonical protocol](../benchmarks/msmu/protocol.md)和 scorer 源码为准，本 ADR 不复制这些
易变化细节。

## Consequences

- 模型输入差异和评分语义可以独立审计，公平轨与原生轨不会静默混表。
- 新增 adapter 通常只新增 inference protocol，不必更换 scorer/cache protocol。
- 输出目录层级和结果表必须显式保存两类 protocol，操作与汇总逻辑更严格。
- profile registry、validator/scorer、输出路径和回归测试必须共同维护这一边界。

## References

- [仓库架构](../architecture.md)
- [MSMU canonical protocol](../benchmarks/msmu/protocol.md)
- `src/spatial_vlm_eval/models/profiles.py`
- `src/spatial_vlm_eval/benchmarks/msmu/scorer.py`
- `tests/models/test_specialized_profiles.py`
- `tests/benchmarks/msmu/test_scorer.py`
