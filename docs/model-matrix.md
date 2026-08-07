# 项目目标测试模型矩阵

本矩阵同时区分“项目级模型身份”和“已经落地的 benchmark-specific inference profile”。三个后续
benchmark 的目标范围是 MSMU 阶段已有 15 个模型身份加 4 个新开源 SOTA 模型，共 19 个模型身份；
benchmark 范围与推进顺序见[四 Benchmark 评测范围](evaluation-scope.md)。

MSMU 注册 profile 保存在 `src/spatial_vlm_eval/models/profiles.py`；同文件的
`CURRENT_TARGET_PROFILE_KEYS` 唯一确定已经实现并完成的 MSMU 目标集合。CV-Bench 使用独立的
`src/spatial_vlm_eval/benchmarks/cv_bench/profiles.py`，其 `PROFILE_SEQUENCE` 唯一确定 23 条目标轨和
串行顺序。Q-Spatial 与 SPBench-SI 也分别由各自 benchmark package 的 `PROFILE_SEQUENCE` 独立锁定
21 条轨，禁止跨 benchmark 复用 prompt、decoding 或 scorer 语义。每条正式结果仍须记录 model revision、
inference protocol、prompt/template、图像处理、decoding 和 scorer protocol。精简展示表只有在逐行
校验这些 provenance、一次只选择一个 scorer protocol，并在模型名称中区分不同 input track 时才可
省略 protocol 列。

三个 benchmark 的 `internvl3_78b` profile 共享 served name `internvl3-78b-three-bench`，model revision
仍固定为 `3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356`，均为 BF16 TP=4、四张 80GB GPU。它们可由
`scripts/internvl3_78b/run_three_bench_evaluation.sh` 只加载一次 vLLM 后依次补齐；各自的 prompt、
decoding、validator、scorer protocol 和输出目录仍完全独立。

## 新增的 4 个开源 SOTA 模型

以下四个模型于 2026-08-03 纳入项目级目标模型范围。其 CV-Bench snapshot revision、上游 commit、
input track 和 inference protocol 已锁入 CV-Bench registry；服务器 snapshot、license、专用 runner 与
测试 gate 已于 2026-08-03 现场验收。它们不加入 MSMU 的 `CURRENT_TARGET_PROFILE_KEYS`。

| Model identity | Hugging Face weights | 纳入原因与报告输入 | 当前状态 |
|---|---|---|---|
| [RoboBrain2.5-8B-NV](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV) | `BAAI/RoboBrain2.5-8B-NV@3d77a19a3ddd8616b3979e03de56096edfb12ff6` | MSMU-Bench 64.17、CV-Bench 94.58 的开源 RGB-only 报告模型 | CV-Bench full + v3 publication gates passed（2026-08-06） |
| [RoboBrain2.5-8B-MT](https://huggingface.co/BAAI/RoboBrain2.5-8B-MT) | `BAAI/RoboBrain2.5-8B-MT@01145b89a0fe49f78f5d677d25af7351088d7c7d` | Q-Spatial Bench 78.31 的开源 RGB-only 公平报告模型；论文表格也写作 MTT | CV-Bench full + v3 publication gates passed（2026-08-06） |
| [HiSpatial-3B](https://huggingface.co/lhzzzzzy/HiSpatial-3B) | `lhzzzzzy/HiSpatial-3B@75a5e3d65351d7602c492aa91533f62b8a252604` | Q-Spatial Bench 85.16 的开源报告模型；该值使用 RGB + MoGe-2 估计 XYZ point map，属于原生/非公平输入 | CV-Bench full + v3 publication gates passed（2026-08-06）；无伪 RGB-only 轨 |
| [SpatialLadder-3B](https://huggingface.co/hongxingli/SpatialLadder-3B) | `hongxingli/SpatialLadder-3B@0819c3adf8827a2ea6c0348d49a23503ecb1f428` | SPBench-SI 70.20 的最高开源 RGB-only 报告模型 | CV-Bench direct/thinking full + v3 publication gates passed（2026-08-06） |

这些分数是外部公开报告参考，不是本项目结果。RoboBrain 两个权重使用相同架构与训练数据，但分别是
NVIDIA 与 Moore Threads 训练版本，作为两个独立模型身份保留。HiSpatial 的架构强制需要 XYZ point
map，因此不建立虚假的 RGB-only 轨。实现 adapter 时分别
从 [FlagOpen/RoboBrain2.5](https://github.com/FlagOpen/RoboBrain2.5)、
[microsoft/HiSpatial](https://github.com/microsoft/HiSpatial) 和
[ZJU-REAL/SpatialLadder](https://github.com/ZJU-REAL/SpatialLadder) 锁定上游 commit，不以模型页的
浮动默认分支代替 revision。

## CV-Bench 当前 23 条目标 inference profile

以下 23 条轨已在 registry、CLI、validator、scorer 和报告发现中注册。prompt 修复后的两条 reasoning
gate 使用 v2，其他轨可在仅 adapter digest 变化时审计迁移。截至 2026-08-06，除四卡
InternVL3-78B 外的 22 条轨均已通过 full-2638 validator、当前 scorer v3 评分和 publication gates；
全局报告为 22/23。下表保留逐轨已验证的静态状态；实时完成情况必须读取服务器 validator、metadata、
summary 和 publication gates。通用轨先审计官方 Transformers processor/template，再使用 vLLM 0.19；
不一致时只能显式回退到锁定 runner。

| Profile | Model / locked revision | Input track | Backend / decoding | 当前状态 |
|---|---|---|---|---|
| `llava_next_mistral_7b` | `llava-hf/llava-v1.6-mistral-7b-hf@2424fdd47412fccc66d91719126b420e9fbd7065` | RGB | vLLM；greedy/512/seed 42 | full + v3 publication gates passed（2026-08-06） |
| `llava_next_yi_34b` | `llava-hf/llava-v1.6-34b-hf@84e4488fffae48f9da316ec31288b7c03f102ec7` | RGB | vLLM TP=2；greedy/512/seed 42 | full + v3 publication gates passed（2026-08-06） |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B-hf@259a3b64a14623c0ec91a045cb43f7c5af5fa6af` | RGB | vLLM；greedy/512/seed 42 | full + v3 publication gates passed（2026-08-06） |
| `internvl3_38b` | `OpenGVLab/InternVL3-38B-hf@b2a05c0c325235f7530d8274c313a1d01082e069` | RGB | vLLM TP=2；greedy/512/seed 42 | full + v3 publication gates passed（2026-08-06） |
| `internvl3_78b` | `OpenGVLab/InternVL3-78B-hf@3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356` | RGB | vLLM TP=4，四张 80GB；greedy/512/seed 42 | blocked：当前服务器仅 2×A800（2026-08-06） |
| `qwen3_vl_2b` | `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203` | RGB | vLLM；greedy/512/seed 42 | full + v3 publication gates passed（2026-08-06） |
| `qwen3_vl_4b` | `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` | RGB | vLLM；greedy/512/seed 42 | full + v3 publication gates passed（2026-08-06） |
| `qwen3_vl_8b` | `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | RGB | vLLM；greedy/512/seed 42 | full + v3 publication gates passed（2026-08-06） |
| `qwen3_vl_32b` | `Qwen/Qwen3-VL-32B-Instruct@0cfaf48183f594c314753d30a4c4974bc75f3ccb` | RGB | vLLM TP=2；greedy/512/seed 42 | full + v3 publication gates passed（2026-08-06） |
| `gpt5_openrouter_non_zdr` | `openai/gpt-5-2025-08-07` | RGB | OpenRouter first-party non-ZDR；medium/16384 | full + v3 publication gates passed（2026-08-06） |
| `gemini31pro_openrouter_non_zdr` | `google/gemini-3.1-pro-preview-20260219` | RGB | OpenRouter first-party non-ZDR；temp 0/medium/16384 | full + v3 publication gates passed（2026-08-06） |
| `ssr_rgb` | `SSR-VLM-7B@7bcb4636f1396325f27f7fbb2f2df121128931bf` | RGB | locked upstream runner；generation manifest 必需 | full + v3 publication gates passed（2026-08-06） |
| `ssr_native` | 上述 VLM + `SSR-MIDI-7B@8ed878fa16e3e440741ed8c1fedfcfe40710258d` | RGB + DepthPro + MIDI + TOR10 | locked upstream runner；generation manifest 必需 | full + v3 publication gates passed（2026-08-06） |
| `spatialrgpt_rgb` | `SpatialRGPT-VILA1.5-8B@64df7902f82b5053f5a53455095805e6de3a1f87` | RGB，无 region/mask/depth | official VILA；greedy/128 | full + v3 publication gates passed（2026-08-06） |
| `3dthinker_rgb` | `3DThinker-Mindcube@69a70411605f86ec69bada0a625bb96ddee995d9` | RGB | locked upstream runner；generation manifest 必需 | full + v3 publication gates passed（2026-08-06） |
| `3dthinker_mental3d` | 同上 | RGB + Mental-3D 提示词 | sampling 0.7/top-p 0.9/2048/seed 42；prompt protocol v2 | full + v3 publication gates passed（2026-08-06） |
| `spatialbot_rgb` | `SpatialBot-3B@41d3b52c642058dfb087885bec0b8e37e0e67f8d` | RGB | official Bunny；greedy/128 | full + v3 publication gates passed（2026-08-06） |
| `spatialbot_zoedepth` | 同上 | RGB + ZoeDepth | official Bunny RGB-D；greedy/128 | full + v3 publication gates passed（2026-08-06） |
| `robobrain25_8b_nv_rgb` | `RoboBrain2.5-8B-NV@3d77a19a3ddd8616b3979e03de56096edfb12ff6` | RGB | official processor；sampling 0.7/top-p 0.8/768/seed 42 | full + v3 publication gates passed（2026-08-06） |
| `robobrain25_8b_mt_rgb` | `RoboBrain2.5-8B-MT@01145b89a0fe49f78f5d677d25af7351088d7c7d` | RGB | official processor；sampling 0.7/top-p 0.8/768/seed 42 | full + v3 publication gates passed（2026-08-06） |
| `hispatial3b_moge2_xyz` | `HiSpatial-3B@75a5e3d65351d7602c492aa91533f62b8a252604` | RGB + MoGe-2 XYZ | official predictor；greedy/100 | full + v3 publication gates passed（2026-08-06） |
| `spatialladder3b_rgb` | `SpatialLadder-3B@0819c3adf8827a2ea6c0348d49a23503ecb1f428` | RGB | official Qwen2.5-VL；SDPA；128；generation manifest 必需 | full + v3 publication gates passed（2026-08-06） |
| `spatialladder3b_thinking` | 同上 | RGB + 官方思考提示词 | SDPA；temp 0.01/1024/seed 42；prompt protocol v2；generation manifest 必需 | full + v3 publication gates passed（2026-08-06） |

通用开源轨的其余统一参数为 temperature 0、`top_p=None`、beam 1。sampling 专用轨的 seed、batch 和
sharding 进入 inference protocol/gate。HiSpatial 上游 CV 脚本只覆盖 2D Relation 和 3D；本项目统一
运行完整 2638 条，因此该结果必须保留 deviation metadata。完整 protocol id 以 registry 为准，不在
本文复制第二份易漂移清单。

## Q-Spatial 当前 21 条目标 inference profile

Q-Spatial 使用独立的 `src/spatial_vlm_eval/benchmarks/q_spatial/profiles.py`；`PROFILE_SEQUENCE` 唯一
确定下表的 21 条轨与顺序，其中 RGB 18 条、派生输入 3 条。代码、协议和回归已于 2026-08-06
完成。截至 2026-08-07，除固定 TP=4 blocked 的 `internvl3_78b` 外，其余 20 轨的服务器
red/blue canary + smoke8 当前 test gate、full-271、正式 validator、完整 provenance、当前 v2 scorer
评分与 publication gates 均已独立复核通过；全局报告为 20/21，唯一缺失 `internvl3_78b`。

| Profile | Model / locked revision | Input track / comparison group | Backend / decoding |
|---|---|---|---|
| `llava_next_mistral_7b` | `llava-v1.6-mistral-7b-hf@2424fdd47412fccc66d91719126b420e9fbd7065` | RGB / RGB | vLLM TP=1 单卡单 endpoint；官方两阶段格式修复，512+64，seed 42 |
| `llava_next_yi_34b` | `llava-v1.6-34b-hf@84e4488fffae48f9da316ec31288b7c03f102ec7` | RGB / RGB | vLLM TP=2；官方两阶段格式修复，512+64，seed 42 |
| `internvl3_8b` | `InternVL3-8B-hf@259a3b64a14623c0ec91a045cb43f7c5af5fa6af` | RGB / RGB | vLLM TP=1 单卡单 endpoint；greedy/512/seed 42 |
| `internvl3_38b` | `InternVL3-38B-hf@b2a05c0c325235f7530d8274c313a1d01082e069` | RGB / RGB | vLLM TP=2；greedy/512/seed 42 |
| `internvl3_78b` | `InternVL3-78B-hf@3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356` | RGB / RGB | vLLM BF16 TP=4；四张 80GB；greedy/512/seed 42 |
| `qwen3_vl_2b` | `Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203` | RGB / RGB | vLLM TP=1 单卡单 endpoint；0.7/top-p 0.8/top-k 20/presence 1.5/1024/seed 3407 |
| `qwen3_vl_4b` | `Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` | RGB / RGB | vLLM TP=1 单卡单 endpoint；同 Qwen sampling |
| `qwen3_vl_8b` | `Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | RGB / RGB | vLLM TP=1 单卡单 endpoint；同 Qwen sampling |
| `qwen3_vl_32b` | `Qwen3-VL-32B-Instruct@0cfaf48183f594c314753d30a4c4974bc75f3ccb` | RGB / RGB | vLLM TP=2；同 Qwen sampling |
| `gpt5_openrouter_non_zdr` | `openai/gpt-5-2025-08-07` | RGB / RGB | OpenRouter first-party non-ZDR；medium/16384；无 temperature |
| `gemini31pro_openrouter_non_zdr` | `google/gemini-3.1-pro-preview-20260219` | RGB / RGB | OpenRouter first-party non-ZDR；temp 0/medium/16384 |
| `ssr_rgb` | `SSR-VLM-7B@7bcb4636f1396325f27f7fbb2f2df121128931bf` | RGB / RGB | official runner；0.1/top-p .001/top-k 1/repetition 1.05/128 |
| `ssr_native` | 上述 VLM + `SSR-MIDI-7B@8ed878fa16e3e440741ed8c1fedfcfe40710258d` | RGB + DepthPro + MIDI + TOR10 / RGB + 派生深度 | official runner；同 SSR decoding，10 TOR |
| `spatialrgpt_rgb` | `SpatialRGPT-VILA1.5-8B@64df7902f82b5053f5a53455095805e6de3a1f87` | RGB，无 region/mask/depth / RGB | official VILA；greedy/128 |
| `3dthinker_rgb` | `3DThinker-Mindcube@69a70411605f86ec69bada0a625bb96ddee995d9` | RGB，无 Mental-3D / RGB | official runner；processor 12544..401408 pixels；0.7/top-p .9/2048 |
| `spatialbot_rgb` | `SpatialBot-3B@41d3b52c642058dfb087885bec0b8e37e0e67f8d` | RGB / RGB | official Bunny；greedy/128 |
| `spatialbot_zoedepth` | 同上 | RGB + ZoeDepth / RGB + 派生深度 | official Bunny；greedy/128 |
| `robobrain25_8b_nv_rgb` | `RoboBrain2.5-8B-NV@3d77a19a3ddd8616b3979e03de56096edfb12ff6` | RGB / RGB | official processor；0.7/top-p .8/768 |
| `robobrain25_8b_mt_rgb` | `RoboBrain2.5-8B-MT@01145b89a0fe49f78f5d677d25af7351088d7c7d` | RGB / RGB | official processor；0.7/top-p .8/768 |
| `hispatial3b_moge2_xyz` | `HiSpatial-3B@75a5e3d65351d7602c492aa91533f62b8a252604` | RGB + MoGe-2 XYZ / RGB + 派生 XYZ | official predictor；greedy/100；禁止 GT depth |
| `spatialladder3b_rgb` | `SpatialLadder-3B@0819c3adf8827a2ea6c0348d49a23503ecb1f428` | RGB，无 thinking prompt / RGB | official Qwen2.5-VL；0.01/top-p 1/repetition 1.05/128 |

完整 revision、protocol、processor、seed strategy 和 image-processing identity 以 registry 为准。采样
本地 runner 使用每请求固定 base seed；不支持重置 RNG 的 backend 只允许单 persistent runner；两条
API 轨明确标记 provider nondeterministic。执行和评分边界见
[Q-Spatial canonical protocol](benchmarks/q_spatial/protocol.md)。

双卡批次的机器计划由 `scheduled_batch.SCHEDULE` 唯一维护：20 条可运行轨分为阶段 A 双卡/API 与
阶段 B GPU 0/GPU 1；`internvl3_78b` 仍固定 TP=4 blocked，不进入 20 轨计划。该计划只改变资源调度，
不改变本表的模型身份、input track、decoding 或 inference protocol。迁移到四卡服务器后，该轨由
`run_internvl3_78b_evaluation.sh` 独立补齐，正式产物追加到原输出根并原地重建同一份 21 轨报告。

## SPBench-SI 当前 21 条目标 inference profile

SPBench-SI 使用独立的 `src/spatial_vlm_eval/benchmarks/spbench_si/profiles.py`；`PROFILE_SEQUENCE` 唯一
确定下表顺序，其中 RGB 18 条、同一源 RGB 派生输入 3 条。代码、协议和本地回归于 2026-08-07 完成；
除固定 TP=4 blocked 的 `internvl3_78b` 外，其余 20 轨已通过服务器当前 test gate。full 批次尚未形成
终态证据，未启动正式评分或发布；下表不把 test gate 写成 full 结果。

| Profile | Model / locked revision | Input track | Backend / locked decoding |
|---|---|---|---|
| `llava_next_mistral_7b` | `llava-v1.6-mistral-7b-hf@2424fdd47412fccc66d91719126b420e9fbd7065` | RGB | vLLM TP=1；native template；greedy/128/seed 42 |
| `llava_next_yi_34b` | `llava-v1.6-34b-hf@84e4488fffae48f9da316ec31288b7c03f102ec7` | RGB | vLLM TP=2；native template；greedy/128/seed 42 |
| `internvl3_8b` | `InternVL3-8B-hf@259a3b64a14623c0ec91a045cb43f7c5af5fa6af` | RGB | vLLM TP=1；greedy/128/seed 42 |
| `internvl3_38b` | `InternVL3-38B-hf@b2a05c0c325235f7530d8274c313a1d01082e069` | RGB | vLLM TP=2；greedy/128/seed 42 |
| `internvl3_78b` | `InternVL3-78B-hf@3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356` | RGB | vLLM BF16 TP=4；四张 80GB；greedy/128/seed 42 |
| `qwen3_vl_2b` | `Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203` | RGB | vLLM TP=1；0.7/top-p .8/top-k 20/presence 1.5/128/seed 3407 |
| `qwen3_vl_4b` | `Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` | RGB | vLLM TP=1；同 Qwen sampling |
| `qwen3_vl_8b` | `Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | RGB | vLLM TP=1；同 Qwen sampling |
| `qwen3_vl_32b` | `Qwen3-VL-32B-Instruct@0cfaf48183f594c314753d30a4c4974bc75f3ccb` | RGB | vLLM TP=2；同 Qwen sampling |
| `gpt5_openrouter_non_zdr` | `openai/gpt-5-2025-08-07` | RGB | OpenRouter first-party non-ZDR；medium/16384；无 temperature |
| `gemini31pro_openrouter_non_zdr` | `google/gemini-3.1-pro-preview-20260219` | RGB | OpenRouter first-party non-ZDR；temperature 0/medium/16384 |
| `ssr_rgb` | `SSR-VLM-7B@7bcb4636f1396325f27f7fbb2f2df121128931bf` | RGB | official runner；checkpoint generation config/128/seed 42 |
| `ssr_native` | 上述 VLM + `SSR-MIDI-7B@8ed878fa16e3e440741ed8c1fedfcfe40710258d` | RGB + DepthPro + MIDI + TOR10 | official persistent runner；128/seed 42 |
| `spatialrgpt_rgb` | `SpatialRGPT-VILA1.5-8B@64df7902f82b5053f5a53455095805e6de3a1f87` | RGB，无伪 region/depth | official VILA；greedy/128/seed 42 |
| `3dthinker_rgb` | `3DThinker-Mindcube@69a70411605f86ec69bada0a625bb96ddee995d9` | RGB，无 Mental-3D | official runner；0.7/top-p .9/2048/seed 42 |
| `spatialbot_rgb` | `SpatialBot-3B@41d3b52c642058dfb087885bec0b8e37e0e67f8d` | RGB | official Bunny；greedy/100/seed 42 |
| `spatialbot_zoedepth` | 同上 | RGB + 同图 ZoeDepth | official Bunny；greedy/100/seed 42 |
| `robobrain25_8b_nv_rgb` | `RoboBrain2.5-8B-NV@3d77a19a3ddd8616b3979e03de56096edfb12ff6` | RGB | official general VQA；0.7/768/seed 42 |
| `robobrain25_8b_mt_rgb` | `RoboBrain2.5-8B-MT@01145b89a0fe49f78f5d677d25af7351088d7c7d` | RGB | official general VQA；0.7/768/seed 42 |
| `hispatial3b_moge2_xyz` | `HiSpatial-3B@75a5e3d65351d7602c492aa91533f62b8a252604` | RGB + 同图 MoGe-2 XYZ | official predictor；greedy/100/seed 42 |
| `spatialladder3b_rgb` | `SpatialLadder-3B@0819c3adf8827a2ea6c0348d49a23503ecb1f428` | RGB，无 thinking prompt | official Qwen2.5-VL；BF16/FA2；0.01/top-p 1/repetition 1.05/128/seed 42 |

所有轨统一使用 SPBench-SI 官方 `default/direct` prompt。双卡 `scheduled_batch.SCHEDULE` 只覆盖 20 条，
明确排除固定 TP=4 的 `internvl3_78b`；该轨由 `run_internvl3_78b_evaluation.sh` 在四卡服务器独立补齐，
不允许量化或 TP=2 替代。20/21 报告只能暂行且必须只缺该轨。完整 prompt、processor、
image-processing、seed strategy、test gate 与 scorer 边界见
[SPBench-SI canonical protocol](benchmarks/spbench_si/protocol.md)。

## MSMU 当前 18 条已完成目标 inference profile

下表状态于 2026-08-03 在 `msmu-a800` 的 canonical 结果根现场复核。服务器实时状态仍须读取结果
目录中的 prediction、validator、metadata、scored rows、judge failures、`summary.json` 和 publication
gates，不能只引用本快照。

| Profile | Model / locked revision | Input track | Backend | Inference protocol | Status |
|---|---|---|---|---|---|
| `gpt5_openrouter_non_zdr` | `openai/gpt-5` alias，canonical `openai/gpt-5-2025-08-07` | question-only RGB | OpenRouter | `msmu_gpt5_question_only_openrouter_non_zdr_v3_medium_16384` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `gemini31pro_openrouter_non_zdr` | `google/gemini-3.1-pro-preview` alias，canonical `google/gemini-3.1-pro-preview-20260219` | question-only RGB | OpenRouter | `msmu_gemini31pro_question_only_openrouter_non_zdr_v3_medium_16384` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `llava_next_mistral_7b` | `llava-hf/llava-v1.6-mistral-7b-hf@2424fdd47412fccc66d91719126b420e9fbd7065` | question-only RGB | vLLM 0.19 | `msmu_llava_next_mistral_7b_question_only_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `llava_next_yi_34b` | `llava-hf/llava-v1.6-34b-hf@84e4488fffae48f9da316ec31288b7c03f102ec7` | question-only RGB | vLLM 0.19 TP=2 | `msmu_llava_next_yi_34b_question_only_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B-hf@259a3b64a14623c0ec91a045cb43f7c5af5fa6af` | question-only RGB | vLLM 0.19 | `msmu_internvl3_8b_question_only_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `internvl3_38b` | `OpenGVLab/InternVL3-38B-hf@b2a05c0c325235f7530d8274c313a1d01082e069` | question-only RGB | vLLM 0.19 TP=2 | `msmu_internvl3_38b_question_only_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `internvl3_78b` | `OpenGVLab/InternVL3-78B-hf@3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356` | question-only RGB | vLLM 0.19 TP=4，四张 80GB GPU | `msmu_internvl3_78b_question_only_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `qwen3_vl_2b` | `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203` | question-only RGB | Transformers，单卡 | `msmu_qwen3_vl_2b_question_only_deterministic_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `qwen3_vl_4b` | `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` | question-only RGB | Transformers，单卡 | `msmu_qwen3_vl_4b_question_only_deterministic_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `qwen3_vl_8b` | `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | question-only RGB | Transformers，单卡 | `msmu_qwen3_vl_8b_question_only_deterministic_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `qwen3_vl_32b` | `Qwen/Qwen3-VL-32B-Instruct@0cfaf48183f594c314753d30a4c4974bc75f3ccb` | question-only RGB | Transformers，单卡，batch size 1 | `msmu_qwen3_vl_32b_question_only_deterministic_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `ssr` | `SSR-VLM-7B@7bcb4636f1396325f27f7fbb2f2df121128931bf` | fair RGB-only，无 TOR/MIDI/depth | official Transformers | `msmu_ssr_rgb_only_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `ssr_native` | 上述 VLM + `SSR-MIDI-7B@8ed878fa16e3e440741ed8c1fedfcfe40710258d` | DepthPro + MIDI + 10 TOR | official Transformers | `msmu_ssr_native_depthpro_midi_tor10_native_v1` | full-987 validator + v4 publication gates passed（2026-08-03）；保留 checkpoint missing-key warning |
| `spatialrgpt` | `SpatialRGPT-VILA1.5-8B@64df7902f82b5053f5a53455095805e6de3a1f87` | RGB-only，无伪造 region/depth | official VILA | `msmu_spatialrgpt_rgb_only_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `3dthinker` | `3DThinker-Mindcube@69a70411605f86ec69bada0a625bb96ddee995d9` | fair question-only RGB | modified Transformers | `msmu_3dthinker_question_only_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `3dthinker_native` | 同上 | official mental-3D control prompt | modified Transformers | `msmu_3dthinker_native_mental3d_native_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `spatialbot` | `SpatialBot-3B@41d3b52c642058dfb087885bec0b8e37e0e67f8d` | fair RGB-only | official Bunny | `msmu_spatialbot_rgb_only_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |
| `spatialbot_native` | 同上 | same-RGB ZoeDepth RGB-D | official Bunny | `msmu_spatialbot_native_zoedepth_rgbd_native_v1` | full-987 validator + v4 publication gates passed（2026-08-03） |

## 完成状态

2026-08-03 现场审计发现且只发现上述 18 条目标结果。每条都满足：prediction index 精确覆盖
`0..986`、正式 validator 通过、metadata 与锁定 profile/revision/protocol 一致、scored rows 为 987
条、judge failures 为 0、八类齐全，且四项 publication gate 全部为真。评分调度状态为
`complete=18`、`pending=0`，因此当前 MSMU 目标 profile 集已全部完成正式测试。

本次只收敛“MSMU 已完成目标 profile 集”，不删除历史 adapter、注册 profile 或既有结果。直接
Qwen2.5-VL/PEFT 轨和其他历史注册 API 轨不进入本表；新增四个 SOTA 模型也不追溯计入这 18 条结果。
报告生成器仍按合法 summary 发现结果，不硬编码本表名单。

## 专用模型身份说明

- SSR-VLM 与 SSR-MIDI 是同一原生推理栈的互补权重，不作为两个模型重复计分。SSR 使用上游
  `yliu-cs/SSR@52a21a14a84a98f07575721dd3200f76c11930d8`；原生轨 DepthPro fork 锁定
  `edb23bbab37cfc4d3fe1048a2f126ca7c590ab64`。其内部 Qwen2.5 组件是专用模型实现依赖，不是独立
  目标模型。
- SpatialRGPT 使用 `AnjieCheng/SpatialRGPT@16715d4f1419997da18926c6ce574802d1eb3a37`。
  MSMU 没有 region/mask，题干也没有 region token，因此只建立 RGB-only 轨。
- 3DThinker 使用 `zhangquanchen/3DThinker@c9469e01b719310b0eaecc1133317e4ecfc74d8c`。
  公开权重必须标为 “MindCube-trained stage-1 checkpoint”，不能代表论文完整最终模型；其内部
  Qwen2.5 processor/model 是专用模型实现依赖，不是独立目标模型。CV-Bench 与其他全部轨一样，
  测试阶段仅要求通过纯红/纯蓝图颜色识别最低视觉门禁；smoke8、单图输入审计和其余 gate 不降低。
- SpatialBot 使用 merged instruction checkpoint，而非 pretrain 或同权重的 LoRA 部署形态；上游锁定
  `BAAI-DCAI/SpatialBot@775ad8cf2f9251261dcd70b2639133d506ff583f`。原生轨 ZoeDepth 锁定
  `d87f17b2f5fdcb174cf4fb115491f4a6c60de152`，只从当前 MSMU RGB 估计深度。两条轨的 vision tower
  均锁定 `google/siglip-so400m-patch14-384@9fdffc58afc957d1a03a25b10dba0329ab15c2a3`，
  通过已验证的本地快照路径加载；ZoeDepth 轨额外锁定
  `isl-org/MiDaS@454597711a62eabcbf7d1e89f3fb9f569051ac9b` 本地 checkout。二者都禁止隐式网络回退。

## 公平轨与原生轨

MSMU 公平轨只使用一张 MSMU RGB 和原题。原生轨可以启用模型官方设计中由该 RGB 派生的组件，但
不能使用传感器/GT depth、reference、类别、额外 QA 或开放词汇检测器伪造 region。其他 benchmark
必须重新定义各自的合法输入；通用原则仍是 fair RGB-only 与官方原生输入分别使用独立 protocol、
输出目录和结果列。

## 已知推理偏差

- Gemini 轨按本项目预注册设置使用 temperature 0；这不是 provider 默认 profile。
- Qwen3-VL 官方模型卡推荐 sampling；本项目为统一横评锁定 greedy/192 tokens 和更小的等视觉
  token 像素预算，因此不是 Qwen 官方推荐 generation/default-resolution profile。
- SSR、SpatialRGPT、3DThinker 和 SpatialBot 为统一横评锁定了计划中的 greedy/token limit，可能与上游
  demo 默认 sampling 或 token limit 不同。
- SpatialBot native 按本项目明确合同把 ZoeDepth 米值量化为 `round(m * 1000)` 的 uint16 毫米。
  锁定 ZoeDepth 源码的 `save_raw_16bit` helper 实际使用 `depth * 256`；因此本轨保留官方
  SpatialBot 三通道 packing，但不是该 helper 的逐字节复刻。此差异已进入 inference protocol metadata，
  不能与其他 depth quantization 混表。
