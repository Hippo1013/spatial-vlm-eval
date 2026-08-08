# 四 Benchmark 评测范围与推进顺序

本文件维护项目级 benchmark 范围、推进顺序和数据准备边界。模型身份与已经落地的 inference profile
见[目标测试模型矩阵](model-matrix.md)；每个 benchmark 的 split、输入合同、scorer、protocol id 和
发布门禁仍须在对应 canonical protocol 与实现中单独锁定。

## 当前范围

| Benchmark | 目标范围 | 仓库实现状态 | 当前阶段 |
|---|---|---|---|
| MSMU-Bench | official `test`，987 条 | 已实现 input contract、validator、inference 与 scorer | 18 条既有目标 profile 已完成；本阶段告一段落 |
| CV-Bench | locked 2D 1438 + 3D 1200，共 2638 条 | contract、23-profile registry、两阶段推理、scorer 与报告已实现 | 22 条轨已通过 full-2638 validator、评分和 publication gates；仅四卡 InternVL3-78B 缺失，报告 22/23（2026-08-06） |
| Q-Spatial Bench | Q-Spatial-ScanNet 170 + Q-Spatial++ 101，共 271 条 | contract、21-profile registry、两阶段推理、numeric scorer 与报告已实现 | 除 TP=4 blocked 的 InternVL3-78B 外，20 轨 test/full-271、正式 validator、provenance、当前 v2 scorer 与 publication gates 已通过；报告 20/21（2026-08-07） |
| SPBench-SI | official 单图 `test`，1009 条；不包含 SPBench-MV | contract、21-profile registry、两阶段推理、双 scorer 与报告已实现 | 包含 InternVL3-78B 在内的 20 轨通过当前 full-1009、双协议评分与 publication gates；Gemini 无可发布 summary，已按闭源补充轨定位搁置。当前文档明确排除 Gemini 与 78B，汇总其余 19 轨（2026-08-08） |

“尚未实现”表示仓库中还没有可发布的 benchmark contract、validator、scorer protocol、运行入口或
结果目录，不能因为数据已经下载就宣称可以正式评测。CV-Bench 的“已实现”只指代码、协议和本地
验证链路，不表示 23 条服务器结果已经产生；状态必须以 test gate、validator、metadata、summary 和
publication gates 为准。Q-Spatial 的“已实现”同样不表示已有模型结果；服务器状态必须读取其
`test_gate.json`、validator、metadata、summary 与 publication gates。SPBench-SI 现场报告发现器已逐轨重算验证
20 条合法候选，包含 InternVL3-78B；Gemini 仍未出现当前主协议 summary。全局汇总不再以测满为前置，
但入表轨仍须通过完整单轨门禁；本次明确排除 Gemini 与 InternVL3-78B，表中保留 19 条。

## 比较重点与完成边界

本项目的核心比较对象是开源通用模型与空间专用模型。GPT-5、Gemini 等闭源 API 轨在所有 benchmark
中只提供补充参照，不是研究工作的比较重点，也不构成阶段收尾必须补齐的完成条件。未完成的闭源轨
可以在状态表和报告中明确标为搁置；不得为了追求形式上的全矩阵完整度自动恢复付费调用。

该优先级不降低结果门禁：任何已经运行并准备入表的闭源轨，仍必须通过对应 benchmark 的完整
validator、provenance、scorer protocol 与 publication gates。2026-08-08 起，SPBench-SI Gemini 轨按
此共识搁置；只有出现新的明确比较需求并重新取得付费 API 授权时才恢复。

## 目标模型覆盖

项目级待测范围为：MSMU 阶段已有 15 个模型身份，加上 2026-08-03 新纳入的
4 个开源 SOTA 参考模型，共 19 个模型身份。新增模型是 RoboBrain2.5-8B-NV、
RoboBrain2.5-8B-MT、HiSpatial-3B 和 SpatialLadder-3B；权重身份、输入轨和当前准备状态见
[模型矩阵的新增 SOTA 模型](model-matrix.md#新增的-4-个开源-sota-模型)。

这里的“19 个模型身份”不等于“19 条 inference profile”。同一模型在 fair RGB-only 与官方原生
输入下必须拆成不同 profile、protocol 和结果目录。现有 18 条 `CURRENT_TARGET_PROFILE_KEYS` 只描述
已经落地并完成的 MSMU profile；CV-Bench 的 23 条目标轨由独立 registry 维护，其中已包含新增 4 个
模型的合法输入轨，但不追溯计入 MSMU 的已完成集合。是否以后补测它们的 MSMU 结果属于独立范围决策。

## 开源 SOTA 参考

下表数值来自 2026-08-03 收集的论文/项目公开报告，只用于选择对照模型，**不是本项目复现结果**。
正式报告必须把项目实测值与外部报告值分开，并保留 benchmark 版本、输入轨与 scorer provenance。

| Benchmark | 公开报告参考 | 纳入矩阵的模型 | 公平性说明 |
|---|---|---|---|
| MSMU-Bench | 64.17 | RoboBrain2.5-8B-NV | 开源、RGB-only 公平报告值 |
| CV-Bench | 94.58 | RoboBrain2.5-8B-NV | 开源、RGB-only 公平报告值 |
| Q-Spatial Bench | 85.16 / 78.31 | HiSpatial-3B / RoboBrain2.5-8B-MT | 85.16 使用 RGB + MoGe-2 估计 XYZ point map；78.31 是开源 RGB-only 公平报告值 |
| SPBench-SI | 79.10 / 70.20 | SpatialLadder-3B | 79.10 的 GAMSI-S1+S2 权重未公开，不纳入待测矩阵；70.20 是 SpatialLadder-3B 的开源 RGB-only 报告值 |

“公平”在本阶段只表示模型输入限于 benchmark 原始提供的 RGB 图像，不额外输入深度图、XYZ point
map 或真实点云。最终 fair/native 合同仍须按各 benchmark 的官方数据与评测代码逐项设计；不得直接
把 MSMU 的 prompt、图像处理或 scorer 复制到其他 benchmark。

## 服务器数据与模型位置

- CV-Bench、SPBench-SI 和 Q-Spatial Parquet 的既有下载位于
  `/media/datasets/tangzecong/huggingface/`。它们是 legacy 资产：只读引用，不移动、不删除，也不向
  该 namespace 继续下载。Q-Spatial ScanNet RGB 使用显式新 namespace 入口
  `/media/datasets/lihaoran/huggingface/datasets/Q-Spatial-Bench/QSpatial_scannet/images`；代码不从
  Parquet 根推断它，也不复制、打包或提交许可内容。
- 2026-08-03 新增四个模型的目标 Hugging Face revision 和上游 commit 已锁入 CV-Bench registry；
  服务器运行前仍须现场核对 snapshot 完整性、license、processor/template 与 runner 实现 SHA。新增
  下载只写入 `/media/datasets/lihaoran/huggingface/`。
- 既有目标模型继续从 `/media/datasets/tangzecong/huggingface/` 的精确 legacy 路径读取，不做批量
  迁移。今后新增的模型、dataset、environment、cache、upstream 和 checkpoint 一律写入
  `/media/datasets/lihaoran/`。

## 上游入口

以下入口是上游身份来源；是否兼容其默认脚本仍以各 benchmark protocol 和测试为准：

| Benchmark | 官方代码 | 数据集 |
|---|---|---|
| Q-Spatial Bench | [andrewliao11/Q-Spatial-Bench-code](https://github.com/andrewliao11/Q-Spatial-Bench-code) | [andrewliao11/Q-Spatial-Bench](https://huggingface.co/datasets/andrewliao11/Q-Spatial-Bench) |
| CV-Bench | [cambrian-mllm/cambrian](https://github.com/cambrian-mllm/cambrian) | [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) |
| SPBench-SI | [ZJU-REAL/SpatialLadder](https://github.com/ZJU-REAL/SpatialLadder) | [hongxingli/SPBench](https://huggingface.co/datasets/hongxingli/SPBench) |

## SPBench-SI 已锁定实现

SPBench-SI 使用 SpatialLadder commit `7a0d2ee85c28728835300310a349a53a15967f2e` 与数据 revision
`03611025a4e6032c558117c0e86b76c8b084c305`。单图 `test` 固定 1,009 题、524 张 ZIP 内 JPEG；loader
直接读取 ZIP，并验证 24,423-byte Parquet 和 49,171,512-byte archive 的锁定 SHA、引用全集和全部
图片可解码。SPBench-MV 不在本阶段范围内。

21 条轨中 18 条 RGB、3 条同源派生 depth/XYZ。所有轨使用官方 `default/direct` prompt，不加入
thinking 或 Mental-3D。主 scorer 使用原始十阈值严格 MRA 与四题型宏平均；当前 SpatialLadder 代码的
提取、inclusive 边界和聚合另存为独立 compatibility audit。双卡计划只包含 20 条，固定 TP=4 的
InternVL3-78B 保留四卡入口。报告可汇总任意非空的 publication-gated 子集，并分开列出明确排除和
其余未完成轨；这不改变单轨 full/validator/双协议/publication provenance 要求。

截至 2026-08-08，代码/回归完成；包含 InternVL3-78B 在内的 20 条轨已通过 full-1009 validator、
主协议与 upstream audit 评分及 publication gates。Gemini 续跑仍未形成可发布 summary，并已按闭源
补充轨定位搁置；当前 `spbench-si-result.md` 按操作者选择排除 Gemini 与 InternVL3-78B，汇总其余
19 轨。详细协议见
[SPBench-SI canonical protocol](benchmarks/spbench_si/protocol.md)，执行
边界见 [SPBench-SI 两阶段 runbook](spbench-si-two-stage-runbook.md)。

## Q-Spatial 已锁定实现

Q-Spatial 使用官方代码 commit `ebe8137eae9781aaf7e29691ce8bc68b2a498a83` 与数据 revision
`17b92e470d58fa46859ebd48ff35a1669828c9be`。全局顺序固定为 ScanNet `0..169` 后接
Q-Spatial++ `170..270`；21 轨中 18 条是 RGB，3 条是 RGB 派生 depth/XYZ。Standard Prompt、两阶段
LLaVA 格式修复、red/blue canary、smoke8、逐请求 seed、四卡 78B 门禁、robust numeric scorer、
split-macro Overall 与报告 provenance 都已进入 registry/validator/tests。

截至 2026-08-07，除固定 TP=4 blocked 的 InternVL3-78B 外，其余 20 轨均已完成服务器 test gate、
full-271、正式 validator、完整 provenance、当前 v2 scorer 评分与 publication gates；全局报告为
20/21，唯一缺失 InternVL3-78B。逐轨实时状态仍须读取服务器 validator、metadata、summary、
publication gates 与批次 `status.tsv`。详细协议见
[Q-Spatial canonical protocol](benchmarks/q_spatial/protocol.md)，执行顺序见
[Q-Spatial 两阶段 runbook](q-spatial-two-stage-runbook.md)。

## CV-Bench 已锁定实现

以下边界已固化在 [CV-Bench protocol](benchmarks/cv_bench/protocol.md)、registry 和回归测试：

1. 锁定官方数据集 revision、split、样本数、媒体字段、答案字段与许可，验证本地数据完整性；
2. 锁定官方代码/评测 commit，确认 prompt、选项解析、计分公式、主指标及异常样本语义；
3. 定义 benchmark-owned model input、prediction schema、validator、subset/full 边界和输出布局；
4. 19 个目标模型身份展开为 23 条独立轨；fair/native、checkpoint 与提示词差异分别登记；
5. processor/template、视觉 canary、smoke8 与输入审计形成绑定 gate；只有完整 2638 条和 publication
   gates 通过后才能发布结果。

截至 2026-08-06，排除四卡 InternVL3-78B 后的 22 条目标轨均已通过当前 test gate、full-2638
validator、scorer v3 评分和全部 publication gates；全局 `cv-bench-result.md` 有 22 行并明确只缺
`internvl3_78b`。InternVL3-78B 仍因当前服务器只有两张 A800、协议要求四张 80GB GPU 而阻塞。
逐轨实时状态仍须读取服务器 validator、metadata、summary 和 publication gates，不能只从本段静态
快照推断。
