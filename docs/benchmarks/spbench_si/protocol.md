# SPBench-SI canonical protocol

本文是本仓库 SPBench-SI 单图 `test` split 的输入、推理、校验、评分和发布唯一规范。机器真值位于
`src/spatial_vlm_eval/benchmarks/spbench_si/`；代码、测试和本文冲突时必须停止运行并一并修复。
SPBench-MV 不在本阶段范围内。

## 1. 锁定来源与数据合同

- 官方评测代码：`ZJU-REAL/SpatialLadder` commit
  `7a0d2ee85c28728835300310a349a53a15967f2e`。
- 数据：`hongxingli/SPBench` revision
  `03611025a4e6032c558117c0e86b76c8b084c305`；该 revision 的 dataset card 声明 Apache-2.0，数据由
  ScanNet validation set 构建。运行者仍须遵守 ScanNet 与本地资产的实际许可边界。
- 论文：未跟踪的 `benchmark_paper/SPBench-SI.pdf`，只作为协议核对来源，不提交 Git。

正式加载必须显式给出两个只读文件，loader 直接从 ZIP 读取 JPEG，不解压、不复制 legacy 资产：

```text
SPBENCH_SI_PARQUET=/media/datasets/tangzecong/huggingface/datasets/SPBench/SPBench-SI.parquet
SPBENCH_SI_IMAGES_ARCHIVE=/media/datasets/tangzecong/huggingface/datasets/SPBench/SPBench-SI-images.zip
```

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `SPBench-SI.parquet` | 24,423 | `72aa46f998212a0d0a9c93ea24107eea086425ccc610083ede35c6218050c9a4` |
| `SPBench-SI-images.zip` | 49,171,512 | `bb53190a1eacf4268fb109b0d8e353c750908bdf33cad8a9221b187d81439461` |

Parquet 必须恰为 1,009 行，官方 `id=1..1009` 唯一，内部 `index=0..1008`。schema 固定为
`id, dataset, scene_name, question_type, question, ground_truth, options, images`，每行只引用一个 JPEG。
ZIP 必须包含 524 张 JPEG，引用集合与成员集合完全一致，且所有图片可解码为 RGB。四类计数固定为：

| question type | count | answer mode |
|---|---:|---|
| `object_abs_distance` | 149 | numerical |
| `object_size_estimation` | 463 | numerical |
| `object_rel_distance` | 91 | multiple choice |
| `object_rel_direction` | 306 | multiple choice |

## 2. 防泄漏输入与 prompt

adapter 只接收：

```text
SPBenchSIModelInput(index, image, system_prompt, user_prompt)
```

其中 `image` 是恰好一张 RGB。adapter 不得看到 `ground_truth`、`question_type`、scene、dataset、官方
id、原始 row 或同图其他问题。prediction JSONL 每行只允许：

```json
{"index": 0, "raw_prediction": "..."}
```

canonical system prompt 是 `You are a helpful assistant.`。user prompt 使用官方 `default/direct`：

```text
Question: {question}

Please answer the question using a numerical value (e.g., 42 or 3.1) directly.
```

选择题在 question 后原样附加 `Options:` 与官方选项，再空一行附加：

```text
Please answer with the option's letter from the given choices (e.g., A, B, etc.) directly.
```

支持 system role 的 processor/API 使用原生 system turn；checkpoint 自动注入同一 system 时不得重复；
不支持 system role 的官方 runner 以锁定分隔符折叠。传输方式、rendered prompt、processor/template、
视觉 token 和 digest 都写入 metadata/test gate。不使用 thinking、Mental-3D 或其他额外推理提示。

## 3. 21 条推理轨

唯一机器清单是 `profiles.PROFILE_SEQUENCE`：18 条 RGB 与 3 条同一 RGB 派生输入。派生轨仅为
`ssr_native`（DepthPro + MIDI + TOR10）、`spatialbot_zoedepth` 和
`hispatial3b_moge2_xyz`；不得使用传感器或 GT depth/XYZ。

- LLaVA-NeXT Mistral-7B/Yi-34B：vLLM 0.19，TP=1/2，原生单阶段 template，greedy、128、seed 42。
- InternVL3 8B/38B/78B：TP=1/2/4，greedy、128、seed 42；78B 固定四张 80GB GPU。
- Qwen3-VL 2B/4B/8B/32B：TP=1/1/1/2，temperature 0.7、top-p 0.8、top-k 20、presence
  penalty 1.5、128、逐请求 seed 3407。
- GPT-5/Gemini 3.1 Pro：模型轨身份锁定为现有 OpenRouter first-party non-ZDR profile，
  medium/16384；GPT-5 不发送 temperature，Gemini temperature 0。Gemini 额度不足时允许同一模型轨仅把
  缺失请求续接到 PackyAPI `Gemini-slb` 企业池；这不是新增 profile 或报告模型。续接仍发送
  `reasoning_effort=medium`、temperature 0 和同一 16,384 token 上限，且必须在 authenticated
  `/v1/models` 与实际 completion 中优先锁定显式 Gemini 3.1 Pro；若企业池只暴露 Packy 官方文档使用的
  `gemini-3-pro-preview`，允许把它作为该平台的兼容 route alias，但仍拒绝 Flash、2.5 或其他模型
  fallback，并在机器 provenance 保存实际 request/returned model id。
- specialized 轨使用 registry 锁定的官方 processor/runner/revision/decoding；SpatialRGPT 不伪造
  region/depth，3DThinker 不加 Mental-3D，SpatialLadder 使用 BF16/FlashAttention2 和
  `12544..401408` pixels。SpatialLadder 锁定上游在 `padding=True` 前显式设置
  `processor.tokenizer.padding_side = "left"`；本项目必须相同设置并 fail closed 证明，不能依赖 checkpoint
  默认的 right padding。对应 inference protocol 固定为
  `spbench_si_spatialladder3b_rgb_rgb_default_direct_folded_user_upstream_locked_v2`。

通用 vLLM 轨在正式运行前必须通过原生 Transformers processor/template 对照：rendered prompt、视觉
placeholder、单图 tensor 和最终 prompt 必须一致；不一致即阻塞，不静默换 backend。

## 4. test gate、full 与调度

`--stage test` 先做完整数据/hash/ZIP 解码检查，再运行两张独立 512×512 纯红/纯蓝 canary。颜色识别、
一张源 RGB 和合法 tensor/media 数量是硬门禁。派生轨还必须记录同一源 RGB 的 SHA 与 derived-input
provenance。固定 smoke8 是 `4,297,306,410,460,518,918,1008`，覆盖四类各两题、八个 scene 和两种
图片分辨率；答案得分只作诊断，不作为 adapter 正确性门禁。

`test_gate.json` 绑定数据与 prompt hash、model/revision、backend/runtime、processor/template、图像处理、
decoding、seed、GPU/TP、endpoint、精确 vLLM 0.19.x runtime version、capacity/batch、canary/smoke
证据。绑定变化时旧 test artifacts 无损
轮换为 `stale-*`，不能跨 signature 恢复。vLLM 容量按 `32→16→8→4→2→1`，API 按
`8→4→2→1`，SpatialLadder native batch 按 `16→8→4→2→1` 探测；每个大于 1 的候选必须同时处理两种
不同长度的 red/blue canary prompt，并在 generation、processor audit 与 gate 中证明 left padding。其他
upstream runner batch=1。

full 必须复用完全匹配的 gate，以 fsync journal 断点恢复，最终原子生成覆盖 `0..1008` 的
`predictions.jsonl`、metadata 和 full `prediction_validation.json`。subset 永不评分。
旧 SpatialLadder v1 gate/full 虽通过结构 validator，但因 right-padded native batch 已确认损坏，不能被
当前 v2 profile 发现或恢复；修复后必须重新运行 test 与 full。

唯一 API-source 续接例外是已经通过 test gate、但因 OpenRouter 额度耗尽而停在 fsync journal 的
`gemini31pro_openrouter_non_zdr`：经操作者明确要求，不重做 test，也不重发既有成功题。入口必须逐条
重新验证原 journal 的 dataset input audit、图片 SHA、prompt/template、模型 revision、Google AI Studio
provider 与单图 evidence；随后复制成功项到新的签名 journal，并仅从缺失 index 调 PackyAPI。旧 gate
与新 binding 除 backend、endpoint、served model、adapter digest 外的 dataset/prompt/profile/decoding/
image/capacity 字段必须逐项相等。首个缺失题必须串行完成并通过 OpenAI-compatible 请求/响应、返回模型
identity、单图和 generation metadata 检查后才允许恢复 gate 中的并发容量。最终 metadata 与独立
`api_source_continuation.json` 必须列出两种额度来源的题数和 index-set digest，明确
`test_stage_reused_without_retest=true`；不得把混合来源伪装为纯 OpenRouter，也不得改变报告中的
Gemini 3.1 Pro 模型身份。

双卡冻结调度覆盖除 InternVL3-78B 外的 20 条轨。Phase A 双卡 lane 为 InternVL3-38B → LLaVA-Yi-34B
→ Qwen3-VL-32B，同时 API lane 严格串行 GPT-5 → Gemini。双卡 lane 全部成功且自有 vLLM 退出、端口
释放后才启动 Phase B 的 GPU0/GPU1 两条 lane。lane 失败只停止本 lane；controller 只回收自有进程组，
不接管端口、不终止未知进程、不自动评分。每条 lane 有只读 watcher，只输出 PASS/FAIL/COMPLETE。

## 5. 主 scorer 与 upstream audit

主 scorer protocol：

```text
spbench_si_original_mra10_strict_robust_direct_controlled_final_expected_unit_four_task_macro_v2
```

解析优先唯一完整 `<answer>`，其次唯一显式 final-answer 区域。选择题只接受唯一独立 A-D，冲突即失败。
数值题只接受唯一有限非负数或 `zero..ninety` 范围内的官方简单英文数字词；自由文本中的 `a/an` 不是
数值。只有 `<answer>` / 显式 final-answer 强区域允许剥离开头的 `A-D.` 误带选项标签，普通全文中的
同形标签继续 fail closed。range、上下界、负数、非有限值和冲突值均为提取失败。

没有强答案区域时，parser 只额外识别受控的最后声明：`distance ... is ...`、含等式的最终 distance、
`longest dimension ... is ...` 与 `provide ... as the longest dimension`；它不会从任意推理句中选择最后一个
数字。`object_abs_distance` 的期望单位固定为 meter，`object_size_estimation` 固定为 centimeter；若回答
显式包含期望单位数值，冲突判断只使用这些数值，否则仍按唯一数值处理。该选择不进行单位换算，例如只
回答 `0.42 meters` 时不会变成 `42 centimeters`，但 `0.42 m (42 cm)` 在 centimeter 题中选择显式的
`42 cm`。

v1 inference metadata 可被 v2 scorer 读取，因为 parser-only 升级没有改变模型输入、图像或生成；summary
必须记录原 metadata 声明并通过兼容门禁。v1 score 目录不是当前主结果，需从原始 prediction 重新评分。

数值题使用 Decimal 计算相对误差 `e=|pred-gt|/|gt|`（GT=0 时为绝对误差），对
`θ=0.50,0.55,...,0.95` 十个阈值严格判断 `e < 1-θ` 并取平均。四个题型各自在本题型全部样本上平均；
NQ 是两个 numerical 题型等权，MCQ 是两个选择题型等权，Overall 是四题型等权宏平均。全 1,009 题
micro 只作审计。

同次评分在完全独立目录生成当前 SpatialLadder direct-mode 兼容审计：

```text
spbench_si_upstream_7a0d2ee_default_direct_compat_v1
```

它精确保留当前上游的 first-match 提取、MRA `<=` 边界与聚合。audit 不能覆盖主 scorer 的逐行产物，
也不能与主协议逐行混表。

## 6. publication gates 与结果性质

评分前必须重跑 full validator；目录评分持有 output-root 锁。主/audit summary、scored rows、prediction、
metadata、revision、prompt、decoding、input track 与 hash 不一致时拒绝发布。同 profile 多个 publishable
候选也拒绝自动选择。

报告只显示主协议的模型（输入形式写在同一单元格括号内）、四题型、NQ、MCQ 与 Overall，
每个指标列的并列最高分全部加粗；
上游兼容性审计仍保留为独立评分产物和 publication provenance，但不进入汇总 Markdown。
报告的集合完整度不再是发布前置门禁：任意非空子集都可汇总，
但入表的每一轨仍必须通过 full validator、主协议与 audit 完整性、provenance 和
publication gates。默认纳入所有可发布候选；操作者可重复传入 `--exclude-profile`
明确排除已完成或未完成的注册轨。部分报告必须同时列出纳入数、明确排除项与未纳入且未排除项，
不得伪装成全量完成。排除只作用于报告集合，不能跳过或放宽单轨发布门禁。
Gemini 的额度来源续接只在本文和机器 provenance 中披露，不进入两张结果表，不改模型显示名，也不增加
provider/source 列；汇总中仍只有同一行 `Gemini 3.1 Pro | RGB`。

本实现的主结果是“original MRA definition + robust direct parser internal score”，不是当前上游代码的
逐字节输出。prompt、parser、单位、边界、聚合或 publication identity 改变时必须更换 scorer protocol、
更新 ADR 并补回归测试。
