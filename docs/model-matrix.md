# MSMU Evaluation Model Matrix

“Adapter available”只表示代码与本地 contract/mock 测试完成，不表示 GPU/API smoke 或 987 条评分已
完成。每个正式结果必须同时记录 model revision、inference protocol、prompt/template、图像处理、
decoding 和 scorer protocol；不同 inference protocol 不得在缺少 protocol 列的表中混合。

## 当前 17 个 inference profile

| Key | Model / locked revision | Input track | Backend | Inference protocol | Status |
|---|---|---|---|---|---|
| `gpt5` | `openai/gpt-5`，provider-managed | question-only RGB | OpenRouter / OpenAI | `msmu_gpt5_question_only_v1` | Adapter + mock tests；live key pending |
| `gemini31pro` | `google/gemini-3.1-pro-preview`，provider-managed | question-only RGB | OpenRouter / Google | `msmu_gemini31pro_question_only_v1` | Adapter + mock tests；live key pending |
| `llava_next_mistral_7b` | `llava-hf/llava-v1.6-mistral-7b-hf@2424fdd47412fccc66d91719126b420e9fbd7065` | question-only RGB | vLLM 0.19 | `msmu_llava_next_mistral_7b_question_only_v1` | 用户手工 stage 1/2 passed |
| `llava_next_yi_34b` | `llava-hf/llava-v1.6-34b-hf@84e4488fffae48f9da316ec31288b7c03f102ec7` | question-only RGB | vLLM 0.19 TP=2 | `msmu_llava_next_yi_34b_question_only_v1` | stage 1/2 passed |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B-hf@259a3b64a14623c0ec91a045cb43f7c5af5fa6af` | question-only RGB | vLLM 0.19 | `msmu_internvl3_8b_question_only_v1` | stage 1/2 passed |
| `internvl3_38b` | `OpenGVLab/InternVL3-38B-hf@b2a05c0c325235f7530d8274c313a1d01082e069` | question-only RGB | vLLM 0.19 TP=2 | `msmu_internvl3_38b_question_only_v1` | stage 1/2 passed |
| `internvl3_78b` | `OpenGVLab/InternVL3-78B-hf@3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356` | question-only RGB | vLLM 0.19 TP=2 config | `msmu_internvl3_78b_question_only_v1` | stage 1 static passed；stage 2/3 blocked |
| `qwen25_vl_7b` | `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5` | question-only RGB | Transformers，单卡 | `msmu_qwen25_vl_question_only_deterministic_v1` | 手工入口名 `qwen25_vl_base`；stage 1/2 passed |
| `qwen25_vl_32b` | `Qwen/Qwen2.5-VL-32B-Instruct@7cfb30d71a1f4f49a57592323337a4a4727301da` | question-only RGB | Transformers，单卡 | `msmu_qwen25_vl_32b_question_only_deterministic_v1` | 权重/revision verified；stage 1/2 passed |
| `qwen25_vl_72b` | `Qwen/Qwen2.5-VL-72B-Instruct@89c86200743eec961a297729e7990e8f2ddbc4c5` | question-only RGB | Transformers，双卡 balanced | `msmu_qwen25_vl_72b_question_only_deterministic_v1` | 权重/revision/two-GPU map verified；stage 1/2 passed |
| `ssr` | `SSR-VLM-7B@7bcb4636f1396325f27f7fbb2f2df121128931bf` | fair RGB-only，无 TOR/MIDI/depth | official Transformers | `msmu_ssr_rgb_only_v1` | stage 1/2 passed |
| `ssr_native` | 上述 VLM + `SSR-MIDI-7B@8ed878fa16e3e440741ed8c1fedfcfe40710258d` | DepthPro + MIDI + 10 TOR | official Transformers | `msmu_ssr_native_depthpro_midi_tor10_native_v1` | stage 1/2 passed；保留 checkpoint missing-key warning |
| `spatialrgpt` | `SpatialRGPT-VILA1.5-8B@64df7902f82b5053f5a53455095805e6de3a1f87` | RGB-only，无伪造 region/depth | official VILA | `msmu_spatialrgpt_rgb_only_v1` | stage 1/2 passed |
| `3dthinker` | `3DThinker-Mindcube@69a70411605f86ec69bada0a625bb96ddee995d9` | fair question-only RGB | modified Transformers | `msmu_3dthinker_question_only_v1` | stage 1/2 passed |
| `3dthinker_native` | 同上 | official mental-3D control prompt | modified Transformers | `msmu_3dthinker_native_mental3d_native_v1` | stage 1/2 passed；smoke 回答质量需阶段三观察 |
| `spatialbot` | `SpatialBot-3B@41d3b52c642058dfb087885bec0b8e37e0e67f8d` | fair RGB-only | official Bunny | `msmu_spatialbot_rgb_only_v1` | stage 1/2 passed |
| `spatialbot_native` | 同上 | same-RGB ZoeDepth RGB-D | official Bunny | `msmu_spatialbot_native_zoedepth_rgbd_native_v1` | stage 1/2 passed |

Qwen 三个参数量均使用原生 structured image content/chat template、greedy、192 tokens 和
pixel `12544..112896`。`qwen25_vl_base` 明确指 7B；PEFT 入口只叠加到该 7B base，本轮不测试 PEFT。
三个参数量的 model revision、inference protocol 和输出目录互相独立。

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
- SSR、SpatialRGPT、3DThinker 和 SpatialBot 为统一横评锁定了计划中的 greedy/token limit，可能与上游
  demo 默认 sampling 或 token limit 不同。
- SpatialBot native 按本项目明确合同把 ZoeDepth 米值量化为 `round(m * 1000)` 的 uint16 毫米。
  锁定 ZoeDepth 源码的 `save_raw_16bit` helper 实际使用 `depth * 256`；因此本轨保留官方
  SpatialBot 三通道 packing，但不是该 helper 的逐字节复刻。此差异已进入 inference protocol metadata，
  不能与其他 depth quantization 混表。
