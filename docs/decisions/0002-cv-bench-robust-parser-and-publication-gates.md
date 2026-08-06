# ADR-0002: CV-Bench 使用稳健选择题解析与独立发布门禁

- Status: Accepted
- Date: 2026-08-03
- Amended: 2026-08-06
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

数据合同只验证并返回数据集题目与有序选项，不拥有最终回答格式。普通 profile 在 profile 层追加官方
direct-letter 后缀；3DThinker Mental-3D 与 SpatialLadder thinking profile 只使用各自锁定的
`<think>/<answer>` 模板，不同时追加 direct-answer 指令。prompt 策略进入 profile/inference protocol
与 test gate binding。

推理 protocol、scorer protocol 和 input track 保持分离。23 条目标轨的 registry 只决定目标矩阵与
provenance，不成为评分器的硬编码发现名单。

2026-08-06 的 v3 修订把“显式字母”收敛为可审计的 declared-answer 语义：解析器只剥离末尾已知
generation terminator，接受整段、首行、末行或紧凑字母/选项文本中的唯一答案，并把证据写入逐行
结果；原始 prediction 保持不变。字母和完整选项文本必须一致，第二个竞争字母即使跟在终止 token
之前也会令该条无效。修订同时为 `answer|option|choice` 和 `A or B` 正则增加完整单词边界，避免把
`options and` 的 `S` 当作候选。这里不引入本地模型或语义 fallback，因为它会把模型未明确声明的
答案猜入正式分数。

v3 scorer 允许消费 inference metadata 中明确记录的 v2 或 v3 scorer ID，并把原声明复制进 summary
供报告复核。这是评分协议兼容，不是 inference gate 迁移：prediction、metadata 和模型输出均不改写，
其他未知 scorer ID 继续 fail closed。

## Consequences

- 分数可以按官方公式比较，但解析行为不是旧首字符脚本的逐字节复刻，必须标为 robust-parser
  internal score。
- 当前 answer-tag-aware scorer 使用 v3 identity；parser 或聚合语义再改变时必须继续更换 scorer
  protocol、更新 protocol/ADR 并增加回归测试。
- model revision、input track、decoding、dataset/adapter binding 改变时旧 test gate 自动失效。
- reasoning prompt 从冲突的双重指令改为单一 answer-tag 指令时必须升级对应 inference protocol；scorer
  的 answer-tag 解析语义未变，因此 scorer protocol 不随之升级。
- subset 和缺 provenance 的历史 prediction 不能发布；必要时只能作为诊断产物保留。

## References

- [CV-Bench protocol](../benchmarks/cv_bench/protocol.md)
- `src/spatial_vlm_eval/benchmarks/cv_bench/scorer.py`
- `src/spatial_vlm_eval/benchmarks/cv_bench/report.py`
- `tests/benchmarks/cv_bench/`
