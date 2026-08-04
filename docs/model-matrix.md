# 项目目标测试模型矩阵

本矩阵同时区分“项目级模型身份”和“已经落地的 benchmark-specific inference profile”。剩余三个
benchmark 的目标范围是 MSMU 阶段已有 15 个模型身份加 4 个新开源 SOTA 模型，共 19 个模型身份；
benchmark 范围与推进顺序见[四 Benchmark 评测范围](evaluation-scope.md)。

MSMU 注册 profile 保存在 `src/spatial_vlm_eval/models/profiles.py`；同文件的
`CURRENT_TARGET_PROFILE_KEYS` 唯一确定已经实现并完成的 MSMU 目标集合。CV-Bench 使用独立的
`src/spatial_vlm_eval/benchmarks/cv_bench/profiles.py`，其 `PROFILE_SEQUENCE` 唯一确定 23 条目标轨和
串行顺序。每条正式结果仍须记录 model revision、
inference protocol、prompt/template、图像处理、decoding 和 scorer protocol。精简展示表只有在逐行
校验这些 provenance、一次只选择一个 scorer protocol，并在模型名称中区分不同 input track 时才可
省略 protocol 列。

## 新增的 4 个开源 SOTA 模型

以下四个模型于 2026-08-03 纳入项目级目标模型范围。其 CV-Bench snapshot revision、上游 commit、
input track 和 inference protocol 已锁入 CV-Bench registry；服务器 snapshot、license、专用 runner 与
测试 gate 已于 2026-08-03 现场验收。它们不加入 MSMU 的 `CURRENT_TARGET_PROFILE_KEYS`。

| Model identity | Hugging Face weights | 纳入原因与报告输入 | 当前状态 |
|---|---|---|---|
| [RoboBrain2.5-8B-NV](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV) | `BAAI/RoboBrain2.5-8B-NV@3d77a19a3ddd8616b3979e03de56096edfb12ff6` | MSMU-Bench 64.17、CV-Bench 94.58 的开源 RGB-only 报告模型 | CV-Bench RGB test gate passed（2026-08-03） |
| [RoboBrain2.5-8B-MT](https://huggingface.co/BAAI/RoboBrain2.5-8B-MT) | `BAAI/RoboBrain2.5-8B-MT@01145b89a0fe49f78f5d677d25af7351088d7c7d` | Q-Spatial Bench 78.31 的开源 RGB-only 公平报告模型；论文表格也写作 MTT | CV-Bench RGB test gate passed（2026-08-03） |
| [HiSpatial-3B](https://huggingface.co/lhzzzzzy/HiSpatial-3B) | `lhzzzzzy/HiSpatial-3B@75a5e3d65351d7602c492aa91533f62b8a252604` | Q-Spatial Bench 85.16 的开源报告模型；该值使用 RGB + MoGe-2 估计 XYZ point map，属于原生/非公平输入 | CV-Bench RGB + MoGe-2 XYZ test gate passed（2026-08-03）；无伪 RGB-only 轨 |
| [SpatialLadder-3B](https://huggingface.co/hongxingli/SpatialLadder-3B) | `hongxingli/SpatialLadder-3B@0819c3adf8827a2ea6c0348d49a23503ecb1f428` | SPBench-SI 70.20 的最高开源 RGB-only 报告模型 | CV-Bench direct/thinking test gate 均 passed（2026-08-03） |

这些分数是外部公开报告参考，不是本项目结果。RoboBrain 两个权重使用相同架构与训练数据，但分别是
NVIDIA 与 Moore Threads 训练版本，作为两个独立模型身份保留。HiSpatial 的架构强制需要 XYZ point
map，因此不建立虚假的 RGB-only 轨。实现 adapter 时分别
从 [FlagOpen/RoboBrain2.5](https://github.com/FlagOpen/RoboBrain2.5)、
[microsoft/HiSpatial](https://github.com/microsoft/HiSpatial) 和
[ZJU-REAL/SpatialLadder](https://github.com/ZJU-REAL/SpatialLadder) 锁定上游 commit，不以模型页的
浮动默认分支代替 revision。

## CV-Bench 当前 23 条目标 inference profile

以下 23 条轨已在 registry、CLI、validator、scorer 和报告发现中注册。截至 2026-08-03，服务器曾现场
审计 22 条 v1 test gate；prompt 修复后的两条 reasoning gate 已于 2026-08-04 通过 v2 test，其他轨的
最终 prompt 不变并可在仅 adapter digest 变化时审计迁移。InternVL3-78B 因当前只有两张 A800 而无法满足四卡门禁。
尚未启动 full-2638，不能把本表解释为已有分数。通用轨先审计
官方 Transformers processor/template，再使用 vLLM 0.19；不一致时只能显式回退到锁定 runner。

| Profile | Model / locked revision | Input track | Backend / decoding | 当前状态 |
|---|---|---|---|---|
| `llava_next_mistral_7b` | `llava-hf/llava-v1.6-mistral-7b-hf@2424fdd47412fccc66d91719126b420e9fbd7065` | RGB | vLLM；greedy/512/seed 42 | test gate passed（2026-08-03） |
| `llava_next_yi_34b` | `llava-hf/llava-v1.6-34b-hf@84e4488fffae48f9da316ec31288b7c03f102ec7` | RGB | vLLM TP=2；greedy/512/seed 42 | test gate passed（2026-08-03） |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B-hf@259a3b64a14623c0ec91a045cb43f7c5af5fa6af` | RGB | vLLM；greedy/512/seed 42 | test gate passed（2026-08-03） |
| `internvl3_38b` | `OpenGVLab/InternVL3-38B-hf@b2a05c0c325235f7530d8274c313a1d01082e069` | RGB | vLLM TP=2；greedy/512/seed 42 | test gate passed（2026-08-03） |
| `internvl3_78b` | `OpenGVLab/InternVL3-78B-hf@3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356` | RGB | vLLM TP=4，四张 80GB；greedy/512/seed 42 | test gate blocked：当前服务器仅 2×A800（2026-08-03） |
| `qwen3_vl_2b` | `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203` | RGB | vLLM；greedy/512/seed 42 | test gate passed（2026-08-03） |
| `qwen3_vl_4b` | `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` | RGB | vLLM；greedy/512/seed 42 | test gate passed（2026-08-03） |
| `qwen3_vl_8b` | `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | RGB | vLLM；greedy/512/seed 42 | test gate passed（2026-08-03） |
| `qwen3_vl_32b` | `Qwen/Qwen3-VL-32B-Instruct@0cfaf48183f594c314753d30a4c4974bc75f3ccb` | RGB | vLLM TP=2；greedy/512/seed 42 | test gate passed（2026-08-03） |
| `gpt5_openrouter_non_zdr` | `openai/gpt-5-2025-08-07` | RGB | OpenRouter first-party non-ZDR；medium/16384 | paid test gate passed（2026-08-03） |
| `gemini31pro_openrouter_non_zdr` | `google/gemini-3.1-pro-preview-20260219` | RGB | OpenRouter first-party non-ZDR；temp 0/medium/16384 | paid test gate passed（2026-08-03） |
| `ssr_rgb` | `SSR-VLM-7B@7bcb4636f1396325f27f7fbb2f2df121128931bf` | RGB | locked upstream runner；generation manifest 必需 | test gate passed（2026-08-03） |
| `ssr_native` | 上述 VLM + `SSR-MIDI-7B@8ed878fa16e3e440741ed8c1fedfcfe40710258d` | RGB + DepthPro + MIDI + TOR10 | locked upstream runner；generation manifest 必需 | test gate passed（2026-08-03） |
| `spatialrgpt_rgb` | `SpatialRGPT-VILA1.5-8B@64df7902f82b5053f5a53455095805e6de3a1f87` | RGB，无 region/mask/depth | official VILA；greedy/128 | test gate passed（2026-08-03） |
| `3dthinker_rgb` | `3DThinker-Mindcube@69a70411605f86ec69bada0a625bb96ddee995d9` | RGB | locked upstream runner；generation manifest 必需 | test gate passed（2026-08-03） |
| `3dthinker_mental3d` | 同上 | RGB + Mental-3D 提示词 | sampling 0.7/top-p 0.9/2048/seed 42；prompt protocol v2 | v2 test gate passed（2026-08-04） |
| `spatialbot_rgb` | `SpatialBot-3B@41d3b52c642058dfb087885bec0b8e37e0e67f8d` | RGB | official Bunny；greedy/128 | test gate passed（2026-08-03） |
| `spatialbot_zoedepth` | 同上 | RGB + ZoeDepth | official Bunny RGB-D；greedy/128 | test gate passed（2026-08-03） |
| `robobrain25_8b_nv_rgb` | `RoboBrain2.5-8B-NV@3d77a19a3ddd8616b3979e03de56096edfb12ff6` | RGB | official processor；sampling 0.7/top-p 0.8/768/seed 42 | test gate passed（2026-08-03） |
| `robobrain25_8b_mt_rgb` | `RoboBrain2.5-8B-MT@01145b89a0fe49f78f5d677d25af7351088d7c7d` | RGB | official processor；sampling 0.7/top-p 0.8/768/seed 42 | test gate passed（2026-08-03） |
| `hispatial3b_moge2_xyz` | `HiSpatial-3B@75a5e3d65351d7602c492aa91533f62b8a252604` | RGB + MoGe-2 XYZ | official predictor；greedy/100 | test gate passed（2026-08-03） |
| `spatialladder3b_rgb` | `SpatialLadder-3B@0819c3adf8827a2ea6c0348d49a23503ecb1f428` | RGB | official Qwen2.5-VL；SDPA；128；generation manifest 必需 | test gate passed（2026-08-03） |
| `spatialladder3b_thinking` | 同上 | RGB + 官方思考提示词 | SDPA；temp 0.01/1024/seed 42；prompt protocol v2；generation manifest 必需 | v2 test gate passed（2026-08-04） |

通用开源轨的其余统一参数为 temperature 0、`top_p=None`、beam 1。sampling 专用轨的 seed、batch 和
sharding 进入 inference protocol/gate。HiSpatial 上游 CV 脚本只覆盖 2D Relation 和 3D；本项目统一
运行完整 2638 条，因此该结果必须保留 deviation metadata。完整 protocol id 以 registry 为准，不在
本文复制第二份易漂移清单。

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
