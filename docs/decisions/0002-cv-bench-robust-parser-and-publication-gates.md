# ADR-0002: CV-Bench 使用稳健选择题解析与独立发布门禁

- Status: Accepted
- Date: 2026-08-03
- Supersedes: none

## Context

Cambrian 锁定评测脚本主要读取答案首字符，难以区分解释性输出、冲突字母、多答案和完整选项文本。
直接复用 MSMU scorer 又会混淆完全不同的 benchmark 协议、指标与结果身份。CV-Bench 还需要阻止
subset、错误 input track 或不同 decoding/revision 的结果进入同一张精简表。

## Decision

CV-Bench 建立独立 scorer protocol，只接受唯一合法显式字母或唯一完整选项文本；带官方 thinking
prompt 的轨允许唯一完整 `<answer>...</answer>` 界定最终答案，多个 answer tag、冲突和歧义统一记零。
指标严格使用官方 ADE/COCO 等权 2D、Omni3D 3D 和二者等权 Overall。正式评分前强制完整 validator，
评分后生成 benchmark-owned publication gates；报告只发现当前 protocol 下 provenance 完整的结果。

推理 protocol、scorer protocol 和 input track 保持分离。23 条目标轨的 registry 只决定目标矩阵与
provenance，不成为评分器的硬编码发现名单。

## Consequences

- 分数可以按官方公式比较，但解析行为不是旧首字符脚本的逐字节复刻，必须标为 robust-parser
  internal score。
- 当前 answer-tag-aware scorer 使用 v2 identity；parser 或聚合语义再改变时必须继续更换 scorer
  protocol、更新 protocol/ADR 并增加回归测试。
- model revision、input track、decoding、dataset/adapter binding 改变时旧 test gate 自动失效。
- subset 和缺 provenance 的历史 prediction 不能发布；必要时只能作为诊断产物保留。

## References

- [CV-Bench protocol](../benchmarks/cv_bench/protocol.md)
- `src/spatial_vlm_eval/benchmarks/cv_bench/scorer.py`
- `src/spatial_vlm_eval/benchmarks/cv_bench/report.py`
- `tests/benchmarks/cv_bench/`
