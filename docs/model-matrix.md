# MSMU Evaluation Model Matrix

“Adapter available”只表示代码与本地 contract/mock 测试完成，不表示 GPU/API smoke 或 987 条评分已
完成。每个正式结果必须同时记录 model revision、inference protocol、prompt/template、图像处理、
decoding 和 scorer protocol。精简展示表只有在逐行校验这些 provenance、一次只选择一个 scorer
protocol，并在模型名称中区分不同 input track 时才可省略 protocol 列。

profile inventory 的机器事实源是 `src/spatial_vlm_eval/models/profiles.py`。下表运行状态最后核验于
2026-08-02；服务器实时状态必须读取结果目录中的 validator、metadata、`status.tsv` 和 `summary.json`，
不能只引用本快照。

## 当前 23 个 inference profile

| Key | Model / locked revision | Input track | Backend | Inference protocol | Status |
|---|---|---|---|---|---|
| `gpt5` | `openai/gpt-5`，provider-managed | question-only RGB | OpenRouter ZDR / OpenAI | `msmu_gpt5_question_only_v1` | Adapter + mock tests；OpenRouter live stage 1 被无可用 ZDR endpoint 阻断（2026-08-01） |
| `gpt5_openrouter_non_zdr` | `openai/gpt-5` alias，canonical `openai/gpt-5-2025-08-07` | question-only RGB；OpenAI only、无 fallback、`data_collection=deny`、不要求 ZDR | OpenRouter only | `msmu_gpt5_question_only_openrouter_non_zdr_v3_medium_16384` | Mac live stage 1/2、full-987 validator 与服务器 v4 publication gates passed；medium reasoning、16384 total completion tokens。既有 low/512 v2 journal 仅作历史诊断，不恢复到 v3（2026-08-02） |
| `gemini31pro` | `google/gemini-3.1-pro-preview`，provider-managed | question-only RGB | OpenRouter ZDR / Google | `msmu_gemini31pro_question_only_v1` | Adapter + mock tests；OpenRouter live stage 1 被无可用 ZDR endpoint 阻断（2026-08-01） |
| `gemini31pro_openrouter_non_zdr` | `google/gemini-3.1-pro-preview` alias，canonical `google/gemini-3.1-pro-preview-20260219` | question-only RGB；Google AI Studio only、无 fallback、`data_collection=deny`、不要求 ZDR | OpenRouter only | `msmu_gemini31pro_question_only_openrouter_non_zdr_v3_medium_16384` | Mac live stage 1/2、full-987 validator 与服务器 v4 publication gates passed；medium reasoning、16384 total completion tokens。既有 low/512 v2 结果仅作历史诊断，不恢复到 v3（2026-08-02） |
| `llava_next_mistral_7b` | `llava-hf/llava-v1.6-mistral-7b-hf@2424fdd47412fccc66d91719126b420e9fbd7065` | question-only RGB | vLLM 0.19 | `msmu_llava_next_mistral_7b_question_only_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `llava_next_yi_34b` | `llava-hf/llava-v1.6-34b-hf@84e4488fffae48f9da316ec31288b7c03f102ec7` | question-only RGB | vLLM 0.19 TP=2 | `msmu_llava_next_yi_34b_question_only_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B-hf@259a3b64a14623c0ec91a045cb43f7c5af5fa6af` | question-only RGB | vLLM 0.19 | `msmu_internvl3_8b_question_only_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `internvl3_38b` | `OpenGVLab/InternVL3-38B-hf@b2a05c0c325235f7530d8274c313a1d01082e069` | question-only RGB | vLLM 0.19 TP=2 | `msmu_internvl3_38b_question_only_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `internvl3_78b` | `OpenGVLab/InternVL3-78B-hf@3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356` | question-only RGB | vLLM 0.19 TP=4，四张 80GB GPU | `msmu_internvl3_78b_question_only_v1` | 四卡 live stage 1/2、full-987 validator 与 v4 publication gates passed（2026-08-01） |
| `qwen25_vl_7b` | `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5` | question-only RGB | Transformers，单卡 | `msmu_qwen25_vl_question_only_deterministic_v1` | full-987 validator + v4 publication gates passed（2026-07-31）；当前 18 行报告未收录 |
| `qwen25_vl_32b` | `Qwen/Qwen2.5-VL-32B-Instruct@7cfb30d71a1f4f49a57592323337a4a4727301da` | question-only RGB | Transformers，单卡 | `msmu_qwen25_vl_32b_question_only_deterministic_v1` | full-987 validator + v4 publication gates passed（2026-07-31）；当前 18 行报告未收录 |
| `qwen25_vl_72b` | `Qwen/Qwen2.5-VL-72B-Instruct@89c86200743eec961a297729e7990e8f2ddbc4c5` | question-only RGB | Transformers，双卡 balanced | `msmu_qwen25_vl_72b_question_only_deterministic_v1` | 权重/revision/two-GPU map verified；stage 1/2 passed；70B+，stage 3 excluded |
| `qwen3_vl_2b` | `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203` | question-only RGB | Transformers，单卡 | `msmu_qwen3_vl_2b_question_only_deterministic_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `qwen3_vl_4b` | `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` | question-only RGB | Transformers，单卡 | `msmu_qwen3_vl_4b_question_only_deterministic_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `qwen3_vl_8b` | `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | question-only RGB | Transformers，单卡 | `msmu_qwen3_vl_8b_question_only_deterministic_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `qwen3_vl_32b` | `Qwen/Qwen3-VL-32B-Instruct@0cfaf48183f594c314753d30a4c4974bc75f3ccb` | question-only RGB | Transformers，单卡，batch size 1 | `msmu_qwen3_vl_32b_question_only_deterministic_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `ssr` | `SSR-VLM-7B@7bcb4636f1396325f27f7fbb2f2df121128931bf` | fair RGB-only，无 TOR/MIDI/depth | official Transformers | `msmu_ssr_rgb_only_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `ssr_native` | 上述 VLM + `SSR-MIDI-7B@8ed878fa16e3e440741ed8c1fedfcfe40710258d` | DepthPro + MIDI + 10 TOR | official Transformers | `msmu_ssr_native_depthpro_midi_tor10_native_v1` | full-987 validator + v4 publication gates passed（2026-07-31）；保留 checkpoint missing-key warning |
| `spatialrgpt` | `SpatialRGPT-VILA1.5-8B@64df7902f82b5053f5a53455095805e6de3a1f87` | RGB-only，无伪造 region/depth | official VILA | `msmu_spatialrgpt_rgb_only_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `3dthinker` | `3DThinker-Mindcube@69a70411605f86ec69bada0a625bb96ddee995d9` | fair question-only RGB | modified Transformers | `msmu_3dthinker_question_only_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `3dthinker_native` | 同上 | official mental-3D control prompt | modified Transformers | `msmu_3dthinker_native_mental3d_native_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `spatialbot` | `SpatialBot-3B@41d3b52c642058dfb087885bec0b8e37e0e67f8d` | fair RGB-only | official Bunny | `msmu_spatialbot_rgb_only_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |
| `spatialbot_native` | 同上 | same-RGB ZoeDepth RGB-D | official Bunny | `msmu_spatialbot_native_zoedepth_rgbd_native_v1` | full-987 validator + v4 publication gates passed（2026-07-31） |

当前 18 行结果报告的 Qwen 横评选择以 Qwen3-VL-Instruct 2B/4B/8B/32B 替换 Qwen2.5-VL
7B/32B；Qwen2.5-VL-72B 未运行阶段三。四条新轨均使用原生 structured image content/chat template、无额外 system message、
greedy、192 tokens 和 pixel `16384..147456`；该范围按 Qwen3-VL 的 32-pixel spatial factor 保持
16..144 个 merged visual token 的预算。四个参数量的 model revision、inference protocol 和输出目录
互相独立，32B 固定 batch size 1。

Qwen2.5-VL adapter、PEFT 入口和已有结果继续保留以支持复现，但不属于当前四模型补测范围。结果
报告生成器本身不硬编码上述 18 行选择；无 profile 筛选时仍收录当前 scorer protocol 下全部合法
summary。旧三条
profile 仍锁定原生模板、greedy、192 tokens 和 pixel `12544..112896`；不得用 Qwen3-VL 的像素范围
恢复旧 journal。

本轮阶段三串行批次的 13 条本地轨均已生成完整 987 条 prediction，通过正式 validator，并完成当前
v4 scorer protocol 的 publication gates。两个 API 模型及其 provider-policy profile、Qwen PEFT 和两个
70B+ profile 未进入本批次；排除表示测试范围选择，不会删除已完成的 Qwen2.5-VL-72B stage 1/2 结果。

Qwen3-VL 四条轨作为后续补测单独依次执行，不追溯改写上述已完成的 13 轨批次或其完成标记。四个
参数量均已在服务器完成 stage 1/2、full-987 正式 validator 和当前 v4 scorer protocol 的
publication gates。运行中的状态仅在 checkout 与 `plan.env` 记录的 `repository_sha` 一致时使用串行脚本
`--qwen3 --status`；代码升级后保留旧 plan/complete marker 作为历史证据，并以各轨 validator、metadata
和 `summary.json` 现场核验，不在新 commit 上复用旧完成标记。本地 adapter/contract 验证不能写成
服务器 stage 1/2、完整推理或评分完成。

InternVL3-78B 以固定 TP=4 的独立四卡手工补测轨完成 stage 1/2、full-987 和 v4 评分；该结果不追溯
加入历史 13 轨或改写其完成标记。两个 non-ZDR API v3 轨同样在历史批次之外独立完成。2026-08-02
现场检查活动结果根为 `complete=18`、`pending=0`，默认报告含 18 行；标准 ZDR API 轨、
Qwen2.5-VL-72B 与 Qwen PEFT 仍不属于这 18 行正式结果。

## 专用模型身份说明

- SSR-VLM 与 SSR-MIDI 是同一原生推理栈的互补权重，不作为两个模型重复计分。SSR 使用上游
  `yliu-cs/SSR@52a21a14a84a98f07575721dd3200f76c11930d8`；原生轨 DepthPro fork 锁定
  `edb23bbab37cfc4d3fe1048a2f126ca7c590ab64`。MIDI checkpoint 的 `tor_proj` hidden size 为
  3584，因此内部 LLM 锁定 `Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796`，
  不能替换成 3B。
- SpatialRGPT 使用 `AnjieCheng/SpatialRGPT@16715d4f1419997da18926c6ce574802d1eb3a37`。
  MSMU 没有 region/mask，题干也没有 region token，因此只建立 RGB-only 轨。
- 3DThinker 使用 `zhangquanchen/3DThinker@c9469e01b719310b0eaecc1133317e4ecfc74d8c`。
  公开权重必须标为 “MindCube-trained stage-1 checkpoint”，不能代表论文完整最终模型。
- SpatialBot 使用 merged instruction checkpoint，而非 pretrain 或同权重的 LoRA 部署形态；上游锁定
  `BAAI-DCAI/SpatialBot@775ad8cf2f9251261dcd70b2639133d506ff583f`。原生轨 ZoeDepth 锁定
  `d87f17b2f5fdcb174cf4fb115491f4a6c60de152`，只从当前 MSMU RGB 估计深度。

## 公平轨与原生轨

公平轨只使用一张 MSMU RGB 和原题。原生轨可以启用模型官方设计中由该 RGB 派生的组件，但不能
使用传感器/GT depth、reference、类别、额外 QA 或开放词汇检测器伪造 region。公平轨与原生轨的
protocol id、输出目录和结果列必须保持独立。

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
