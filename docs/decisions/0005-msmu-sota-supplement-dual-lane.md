# ADR-0005: MSMU SOTA 双 Lane、统一评分与范围晋级

- Status: Accepted
- Date: 2026-08-11
- Supersedes: none

## Context

MSMU 已有 18 条完整目标结果，需要补测 RoboBrain2.5 NV/MT、HiSpatial 和 SpatialLadder。三种模型家族
的官方输入架构、decoding 和运行环境不同；SpatialLadder 还需要 direct 与 generic thinking 两条不可
混用的轨。两张 A800 80GB 允许两个并行 lane，但本地 judge 与推理模型不能同时占用同一资源。

其他 benchmark 已验证部分 processor、MoGe 与 left-padding 技术，但 MSMU 的首问输入、prediction
schema、validator、scorer、阈值和 macro-8 必须保持 benchmark-owned，不能复用其 prompt 或评分语义。
同时，新增正式范围不能在运行前写成“已完成”。

## Decision

1. 在 benchmark-neutral model-family package 中实现 RoboBrain2.5、HiSpatial 与 SpatialLadder adapter；
   运行时只接收 `index/image/question`，五条轨分别绑定独立 inference protocol。
2. 冻结两条 lane：GPU0 为 NV → HiSpatial → direct，GPU1 为 MT → thinking；lane 内串行、lane 间并发。
3. 每条 lane 独占一个 pipe-driven watcher。控制器只清理自有进程组；已有 finalized 产物不合法时原地
   fail closed，合法 journal 和正式产物可恢复。
4. 两条 lane full-987 全部完成前不启动 judge。之后只启动一个 judge，按五条冻结顺序对精确 prediction
   路径串行评分；恢复时只补缺失评分。
5. 报告 `--check` 要求既有 18 条与新增 5 条在同一 scorer protocol 下各有且只有一个完整 summary；
   通过后才原子生成 23 行 `msmu-result.md`。
6. 开发阶段四条 main profile 不加入 `CURRENT_TARGET_PROFILE_KEYS`。只有 predictions、validator、summary、
   judge failures、publication gates 和报告现场验收后才由 18 晋级为 22；thinking 永久只作补充轨。
7. MSMU scorer protocol 不变。任何 prompt、派生输入或 decoding 差异只改变对应 inference protocol。

## Consequences

- 服务器需要三个可复用或专用 interpreter、锁定模型/upstream/MoGe 资产和两张空闲 80GB GPU。
- 控制器增加仓库外 `_sota_supplement/frozen-plan.json`、`status.tsv` 和 lane/judge/score/report 日志。
- 一条 lane 的故障会停止另一条尚在运行的控制器自有进程，但不会删除已经完成的正式产物。
- 23 行报告把 SpatialLadder thinking 明确标为补充输入轨，主矩阵完成数仍为 22。
- adapter digest 绑定选中 profile 的完整配置与 family 源码，不绑定可后置变更的全局目标集合，保证完成
  后的 18→22 范围晋级不会使已经验证的 inference provenance 自失效。

## References

- [MSMU canonical protocol](../benchmarks/msmu/protocol.md)
- [MSMU SOTA 双 Lane runbook](../msmu-sota-supplement.md)
- `src/spatial_vlm_eval/models/sota_spatial/`
- `scripts/msmu/run_sota_supplement.sh`
- `tests/models/test_sota_supplement_profiles.py`
- `tests/models/test_sota_supplement_orchestration.py`
