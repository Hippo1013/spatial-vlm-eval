# 四 Benchmark 评测范围与推进顺序

本文件维护项目级 benchmark 范围、推进顺序和数据准备边界。模型身份与已经落地的 inference profile
见[目标测试模型矩阵](model-matrix.md)；每个 benchmark 的 split、输入合同、scorer、protocol id 和
发布门禁仍须在对应 canonical protocol 与实现中单独锁定。

## 当前范围

| Benchmark | 目标范围 | 仓库实现状态 | 当前阶段 |
|---|---|---|---|
| MSMU-Bench | official `test`，987 条 | 已实现 input contract、validator、inference 与 scorer | 18 条既有目标 profile 已完成；本阶段告一段落 |
| CV-Bench | locked 2D 1438 + 3D 1200，共 2638 条 | contract、23-profile registry、两阶段推理、scorer 与报告已实现 | 本地链路验证中；服务器 test gate/full-2638 尚未运行 |
| Q-Spatial Bench | Q-Spatial++ 与 Q-Spatial-ScanNet | 尚未实现 | **下一项待定**；ScanNet 原始图像的授权与完整性须另行验收 |
| SPBench-SI | SPBench 单图版本；不包含 SPBench-MV | 尚未实现 | 与 Q-Spatial Bench 的先后待定 |

“尚未实现”表示仓库中还没有可发布的 benchmark contract、validator、scorer protocol、运行入口或
结果目录，不能因为数据已经下载就宣称可以正式评测。CV-Bench 的“已实现”只指代码、协议和本地
验证链路，不表示 23 条服务器结果已经产生；状态必须以 test gate、validator、metadata、summary 和
publication gates 为准。Q-Spatial Bench 与 SPBench-SI 的后续先后尚未确定。

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

- Q-Spatial Bench、CV-Bench、SPBench-SI 的既有下载位于
  `/media/datasets/tangzecong/huggingface/`。它们是 legacy 资产：只读引用，不移动、不删除，也不向
  该 namespace 继续下载。路径存在不等于 split、图片、license 或 fingerprint 已验收。
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

## CV-Bench 已锁定实现

以下边界已固化在 [CV-Bench protocol](benchmarks/cv_bench/protocol.md)、registry 和回归测试：

1. 锁定官方数据集 revision、split、样本数、媒体字段、答案字段与许可，验证本地数据完整性；
2. 锁定官方代码/评测 commit，确认 prompt、选项解析、计分公式、主指标及异常样本语义；
3. 定义 benchmark-owned model input、prediction schema、validator、subset/full 边界和输出布局；
4. 19 个目标模型身份展开为 23 条独立轨；fair/native、checkpoint 与提示词差异分别登记；
5. processor/template、视觉 canary、smoke8 与输入审计形成绑定 gate；只有完整 2638 条和 publication
   gates 通过后才能发布结果。

本轮不自动启动服务器 full-2638。截至 2026-08-03，22/23 条轨曾通过服务器 v1 test gate；prompt
冲突修复后 `3dthinker_mental3d` 与 `spatialladder3b_thinking` 已于 2026-08-04 通过 v2 test gate，
其余轨可在最终 prompt 不变且仅 adapter digest 变化时审计迁移；
InternVL3-78B 因当前服务器只有两张 A800、协议要求四张 80GB GPU 而阻塞。下一执行动作是取得四卡
资源补齐该 gate；其余轨可在用户确认后进入正式 full 阶段。
