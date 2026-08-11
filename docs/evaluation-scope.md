# 四 Benchmark 评测范围与项目进度

本文件是项目进度的唯一日期化文档快照。模型身份与 benchmark-specific profile 见
[目标测试模型矩阵](model-matrix.md)；split、输入合同、scorer 和发布门禁仍以对应 canonical protocol
与实现为准。实时状态必须重新读取服务器产物，不能把本页当作持续更新的监控面板。

## 已验证进度快照

2026-08-11 在 `msmu-a800` 的 canonical 输出根只读复核：

| Benchmark | 正式范围 | 已验证进度 | 下一步 |
|---|---|---|---|
| MSMU-Bench | official `test` 987 条 | 当前 18 条目标 profile 均有完整 validator、v4 summary 与 publication gates；全局报告存在 | 本阶段完成；新增模型属于新范围 |
| CV-Bench | locked 2D 1438 + 3D 1200，共 2638 条 | 23/23 条轨可发布；`cv-bench-result.md` 完整且 missing 为空 | 本阶段完成 |
| Q-Spatial Bench | ScanNet 170 + Q-Spatial++ 101，共 271 条 | 21/21 条轨可发布；RGB 18/18、全轨 21/21，报告 missing 为空 | 本阶段完成 |
| SPBench-SI | official single-image `test` 1009 条；不含 SPBench-MV | 20/21 条轨可发布，仅 Gemini 无 summary；当前报告按操作者选择纳入 19/21，并明确排除已完成的 InternVL3-78B 与未完成的 Gemini | 核心开源比较已完成；Gemini 仅在新需求和付费授权后恢复 |

复核使用各 benchmark 的公开 `--status` / `build_results_report.sh --check` 入口，并检查正式 report；
没有启动推理、GPU、评分或 API。SPBench-SI 的“20 条可发布候选”和“报告 19 条”不是冲突：前者描述
canonical 产物，后者描述一次显式的报告集合选择。

截至该快照，四个 benchmark 的当前代码/回归完成；后续工作属于新范围或可选闭源补充，不是修补
现有开源主链路。

## 比较重点与完成边界

项目核心比较对象是开源通用模型与空间专用模型。GPT-5、Gemini 等闭源 API 轨只提供补充参照，不是
阶段收尾必须补齐的形式门槛。未完成的闭源轨可以明确搁置；不得为了矩阵完整度自动恢复付费调用。

该优先级不降低结果门禁：任何准备入表的闭源结果仍须通过完整 validator、provenance、scorer
protocol 与 publication gates。SPBench-SI Gemini 自 2026-08-08 起搁置，只有新的明确比较需求和独立
付费授权才能恢复。

## 目标模型覆盖

项目级范围为 MSMU 阶段已有 15 个模型身份，加上 4 个开源 SOTA 参考，共 19 个模型身份。新增模型是
RoboBrain2.5-8B-NV、RoboBrain2.5-8B-MT、HiSpatial-3B 和 SpatialLadder-3B；权重身份、合法输入轨与
decoding 见[模型矩阵](model-matrix.md#新增的-4-个开源-sota-模型)。

“19 个模型身份”不等于“19 条 inference profile”。fair RGB-only、派生 depth/XYZ、额外提示词和不同
checkpoint 必须拆成独立 profile、protocol 和结果目录。MSMU 的 18 条
`CURRENT_TARGET_PROFILE_KEYS` 是其已完成目标集合；CV-Bench 的 23 条目标轨由独立 registry 维护，
Q-Spatial 与 SPBench-SI 也各自维护 21 条，不能跨 benchmark 复制语义。

## 开源 SOTA 参考

下列数字来自 2026-08-03 收集的论文或项目公开报告，只用于选择对照模型，**不是本项目复现结果**。

| Benchmark | 公开报告参考 | 纳入模型 | 输入说明 |
|---|---|---|---|
| MSMU-Bench | 64.17 | RoboBrain2.5-8B-NV | RGB-only |
| CV-Bench | 94.58 | RoboBrain2.5-8B-NV | RGB-only |
| Q-Spatial Bench | 85.16 / 78.31 | HiSpatial-3B / RoboBrain2.5-8B-MT | 前者使用 RGB + MoGe-2 XYZ，后者 RGB-only |
| SPBench-SI | 79.10 / 70.20 | SpatialLadder-3B | 79.10 的 GAMSI-S1+S2 权重未公开；矩阵采用开源 RGB-only 权重 |

“公平”只表示模型输入限于 benchmark 原始 RGB，不额外输入深度、XYZ 或真实点云。每个 benchmark
仍须独立定义 prompt、图像处理和 scorer。

## 服务器数据与模型位置

- CV-Bench、SPBench-SI 和 Q-Spatial Parquet 的既有下载位于
  `/media/datasets/tangzecong/huggingface/`，只读引用，不移动、不删除、不继续下载。
- Q-Spatial ScanNet RGB 从
  `/media/datasets/lihaoran/huggingface/datasets/Q-Spatial-Bench/QSpatial_scannet/images` 显式读取；
  不从 Parquet 根推断，也不复制、打包或提交许可内容。
- 新增模型、dataset、environment、cache、upstream 和 checkpoint 一律写入
  `/media/datasets/lihaoran/`。运行前仍须现场核对 snapshot、license、processor/template 与 runner
  实现 SHA。

## 各 benchmark 的锁定实现

### MSMU-Bench

official `test` 固定 987 条，主指标为八类非加权 macro-8。当前实现是 official-compatible local-judge
internal score，不是 strict official score。输入与评分细节见
[MSMU canonical protocol](benchmarks/msmu/protocol.md)。

### CV-Bench

锁定 2638 条数据、23-profile registry、两阶段 gate、robust MCQ scorer 和 publication-gated 报告。
HiSpatial 合法轨运行完整 2638 条并保存相对上游子集脚本的 deviation metadata。详见
[CV-Bench protocol](benchmarks/cv_bench/protocol.md)。

### Q-Spatial

锁定官方代码 commit `ebe8137eae9781aaf7e29691ce8bc68b2a498a83` 与数据 revision
`17b92e470d58fa46859ebd48ff35a1669828c9be`。顺序固定为 ScanNet `0..169` 后接 Q-Spatial++
`170..270`；21 轨中 18 条 RGB、3 条同图派生 depth/XYZ。主分使用 robust numeric parser 与两个 split
等权的 inclusive `delta <= 2`。详见[Q-Spatial protocol](benchmarks/q_spatial/protocol.md)。

### SPBench-SI

锁定 SpatialLadder commit `7a0d2ee85c28728835300310a349a53a15967f2e` 与数据 revision
`03611025a4e6032c558117c0e86b76c8b084c305`。loader 直接读取 24,423-byte Parquet 和
49,171,512-byte ZIP，覆盖 1,009 题、524 张 JPEG。21 条轨统一使用 `default/direct`；主分为严格原始
MRA 四题型宏平均，上游 current direct 行为只作独立 audit。详见
[SPBench-SI protocol](benchmarks/spbench_si/protocol.md)。

## 上游入口

| Benchmark | 官方代码 | 数据集 |
|---|---|---|
| Q-Spatial Bench | [andrewliao11/Q-Spatial-Bench-code](https://github.com/andrewliao11/Q-Spatial-Bench-code) | [andrewliao11/Q-Spatial-Bench](https://huggingface.co/datasets/andrewliao11/Q-Spatial-Bench) |
| CV-Bench | [cambrian-mllm/cambrian](https://github.com/cambrian-mllm/cambrian) | [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) |
| SPBench-SI | [ZJU-REAL/SpatialLadder](https://github.com/ZJU-REAL/SpatialLadder) | [hongxingli/SPBench](https://huggingface.co/datasets/hongxingli/SPBench) |
