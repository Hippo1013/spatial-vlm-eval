# MSMU-Bench Evaluation Protocol

## 协议身份

当前 canonical scorer：

```text
sdvlm_official_compat_local_judge_v3_grounding_split_strict_quant_length
```

准确描述为：

> SD-VLM official-compatible thresholds/macro-8 + local judge + grounding split fix + strict quantitative length

它用于项目内部消融和横向比较，不是 strict official GPT-4-Turbo score。

参考官方代码：`cpystan/SD-VLM@4023c5c8ee2b909f3556445484e92ffa47bc500f`。

## 数据合同

official test split 固定为 987 条，每条包含一张图片、一轮 QA 和一个 raw type。每个问题恰有一个
字面 `<image>` 占位符。

| Official type | Raw type | Count |
|---|---|---:|
| scale_estimation | width / height / size | 259 |
| absolute_distance | distance | 40 |
| count | count | 96 |
| grounding | position | 87 |
| refer_obj_estimation | refer_two_objects / refer_three_objects | 190 |
| relative_position | left/right | 161 |
| scale_compare | taller_two_object / tall_three_objects | 106 |
| existence | zero | 48 |

grounding 分为 42 条 object→coordinate 和 45 条 coordinate→object。

## 被测模型输入

每条只能输入：

1. 对应图片；
2. 第一条 user question。

不得输入 reference、raw type、task family、其他 QA 或同图历史。`<image>` 只是数据占位符；对
Qwen 必须删除字面 token，并用 structured image content 传图。

不得添加答案格式提示。尤其 `size` reference 是 length × width × height，但该数据集约定不能在
生成前提示模型。

## 当前 Qwen profile

当前登记 `Qwen2.5-VL-7B-Instruct`、`32B-Instruct` 和 `72B-Instruct` 三种参数量；
`qwen25_vl_base` 手工入口明确指 7B。三者输入和 decoding 设置一致，但 model revision、
inference protocol 与输出目录独立。

| Setting | Value |
|---|---|
| Chat template | Qwen2.5-VL native template |
| Implicit system | `You are a helpful assistant.` |
| Image pixels | 12544..112896 |
| Sampling | disabled |
| Beams | 1 |
| Max new tokens | 192 |
| Output order | original index, 0..986 |

这不是 SD-VLM 官方的 sampling profile（temperature 0.2、1024 new tokens），两者不得在同一
对比表中混用。

## 多模型 inference protocol

scorer protocol 与 inference protocol 是两个正交身份。此次新增 adapter 不改变 judge prompt、阈值、
列表长度语义、grounding 路由、cache key 或 macro-8 聚合，因此 canonical scorer id 保持不变。
prompt、图像派生组件或 decoding 不同的轨使用独立 inference protocol：

| Profile | Legal model input | Decoding | Inference protocol |
|---|---|---|---|
| GPT-5 | one RGB + original question | low reasoning, no temperature, 192 completion tokens | `msmu_gpt5_question_only_v1` |
| Gemini 3.1 Pro preview | one RGB + original question | low reasoning, temperature 0, 192 completion tokens | `msmu_gemini31pro_question_only_v1` |
| LLaVA-NeXT Mistral 7B | one RGB + original question | greedy, 192 | `msmu_llava_next_mistral_7b_question_only_v1` |
| LLaVA-NeXT Yi 34B | one RGB + original question | greedy, 192 | `msmu_llava_next_yi_34b_question_only_v1` |
| InternVL3 8B / 38B / 78B | one RGB + original question | greedy, 192 | model-size-specific `msmu_internvl3_*_question_only_v1` |
| Qwen2.5-VL 7B / 32B / 72B | one RGB + original question；structured image content | greedy, 192；72B 双卡 balanced | model-size-specific `msmu_qwen25_vl*_question_only_deterministic_v1` |
| SSR fair | one RGB + original question；无 TOR/MIDI/depth | greedy, 192 | `msmu_ssr_rgb_only_v1` |
| SSR native | one RGB + original question；same-RGB DepthPro + MIDI + 10 TOR | greedy, 192 | `msmu_ssr_native_depthpro_midi_tor10_native_v1` |
| SpatialRGPT | one RGB + original question；无 region/mask/depth | greedy, 192 | `msmu_spatialrgpt_rgb_only_v1` |
| 3DThinker fair | one RGB + original question | greedy, 192 | `msmu_3dthinker_question_only_v1` |
| 3DThinker native | one RGB + original question + official begin-position mental-3D control prompt | greedy, 2048；last complete answer tag | `msmu_3dthinker_native_mental3d_native_v1` |
| SpatialBot fair | one RGB + original question | greedy, 192 | `msmu_spatialbot_rgb_only_v1` |
| SpatialBot native | one RGB + original question；same-RGB ZoeDepth uint16-mm derived depth | greedy, 192 | `msmu_spatialbot_native_zoedepth_rgbd_native_v1` |

3DThinker native 缺少完整 `<answer>...</answer>` 时保留 raw response 作为 prediction 并写 warning，不能
静默返回空值。SpatialBot native 的 depth 只能从当前 RGB 估计，不接受 GT/sensor depth。SpatialRGPT
因 MSMU 无 region/mask 且题干无 region token，不运行 detector 伪造 region。

SpatialBot depth 合同明确为 `clip(round(metres * 1000), 0, 65535)` 后做上游三通道 packing。锁定
ZoeDepth 的 `save_raw_16bit` helper 使用 `depth * 256`，所以本项目的毫米量化不是该 helper 的字节级
复刻；该已知偏差属于当前 inference protocol，若改为 `*256` 必须更换 inference protocol id 和测试。

OpenAI-compatible 请求严格只有一个 user message，其中一个 question text part 和一个 PNG data URI；
不发送 system、history、reference 或答案格式提示。OpenRouter 路由同时设置首方 provider only、
`allow_fallbacks=false`、`require_parameters=true`、`data_collection=deny` 和 `zdr=true`，并在成功写
journal 前校验 generation metadata 的 canonical model、provider 和 `num_media_prompt==1`。

vLLM 0.19 服务必须设置 `--limit-mm-per-prompt.image 1`、锁定 revision/served name/TP/dtype。静态
preflight 分别检查 LLaVA prompt 中恰有一个 `<image>`、InternVL prompt 中恰有一个
`<IMG_CONTEXT>`，且 processor 返回非空 `pixel_values`；服务启动后还必须通过红/蓝合成图 canary。

## Prediction JSONL

每行包含：

```json
{"index":0,"raw_type":"width","task_family":"scale_estimation","question":"What is the width of the wall shelves?","reference":"the wall shelves measures 0.37 meters in width.","prediction":"The shelves are about 0.4 meters wide."}
```

前五个字段由 official row 和固定映射生成，只有 `prediction` 来自模型。输出中的 question 删除
字面 `<image>` 并做首尾 whitespace 清理；reference 做首尾 whitespace 清理。不得手写或改述。
这是精确六字段 schema；`official_type`、`score`、`judge` 或其他任何额外字段均不属于 prediction，
必须作为 hard error 拒绝。`official_type` 只能由 scorer 在完整校验通过后根据 dataset-owned
`raw_type` 经固定映射派生。

## 强制校验

校验器检查：

- 恰好 987 条、index 为 0..986 且无重复；
- 每行恰好包含规定的六个字段，不得缺少或增加字段；
- raw type、task family、question、reference 与 test row 一致；
- 八类数量正确；
- 空 prediction 明确列入 warnings。

空 prediction 不使 `passed` 变为 false，评分时会得到零分或抽取失败。其他结构/provenance 错误
均为 hard error。`--allow-subset` 仅用于独立 validator 的调试，scorer 固定执行 full-split 校验，
无法用该参数绕过。

`--indices` 与 `--limit` 只控制推理 debug target。逐样本 journal 可以断点续跑；正式 JSONL 只有在
所有 target 成功后才原子生成。带 `--indices`/`--limit` 的产物 metadata 必须是
`publishable_inference=false`，pipeline 不允许调用 scorer。正式输出仍必须覆盖 `0..986`。

## Judge 与评分

默认 judge 是本地 Qwen2.5-14B-Instruct，通过 vLLM OpenAI-compatible endpoint 提供，temperature
固定为 0。judge 只看 question/reference/prediction 文本，不看图。

### 数值任务

- 非 grounding：将 reference/prediction 换算为米或 count；计算对称 ratio
  `max(pred/gt, gt/pred)`；`ratio < 1.25` 得 1。
- 多数值：逐位置 ratio 后取平均，再应用 `< 1.25`。
- prediction 列表短于 reference 时失败；多出的尾值忽略。
- coordinate grounding：对应坐标绝对误差的均值 `<= 0.1` 得 1。
- 无法抽取或 malformed 为 0。

### 定性与 object grounding

- 定性任务：`your_mark > 0.5` 得 1。
- object-at-coordinate：grounding 专用 judge prompt 判断对象语义一致性。

后者不同于官方 `all-MiniLM-L6-v2 cosine >= 0.5`。

### 汇总

主指标为八类 accuracy 非加权平均：

```text
official_macro8_accuracy = mean(
  scale_estimation,
  absolute_distance,
  count,
  grounding,
  refer_obj_estimation,
  relative_position,
  scale_compare,
  existence,
)
```

`micro_accuracy` 只作补充。正式 summary 必须同时满足：

- `publishable == true`；
- prediction 完整校验通过；
- `num_samples == 987` 且 index 精确覆盖 `0..986`；
- `missing_official_types == []`；
- `num_judge_failures == 0`。

上述状态由 `publication_gates` 和 `publication_gate_failures` 以机器可读形式记录。未通过门禁的
诊断 summary 不含可引用的正式指标，不得进入结果表。

## Cache 与产物

cache key 包含 protocol、official type、question、reference、prediction、完整 judge prompt、judge
model 和 endpoint。更换上述任一内容不会复用旧 response。每个模型和协议仍必须使用独立目录。

只有能够解析且满足对应任务响应 schema 的 judge 结果才写入 `judge_cache.jsonl`。HTTP/超时、JSON
解析或响应 schema 失败写入独立的 `judge_failures.jsonl`，不得作为完成项缓存；已有 cache 中的
失败记录也必须视为 pending 并在下次运行时重试。只要存在未解决 judge 失败，scorer 必须写出
`publishable == false` 的诊断 summary 并以非零状态退出。该可靠性修复不改变合法 judge response
的评分语义，因此保留当前 scorer/cache protocol id。

评分目录输出：

- `prediction_validation.json`
- `judge_cache.jsonl`
- `judge_failures.jsonl`
- `scored_rows.jsonl`
- `summary.json`
- 运行日志

## 与 strict official 的差异

当前实现与官方至少有以下差异：Qwen 原生 prompt/图像处理、隐式 system prompt、greedy decoding、
本地 Qwen judge、JSON-only 约束、judge temperature、容错解析和 grounding object prompt/语义评分。

此外，锁定的官方 commit 存在语法错误、缺失 `load_dataset` import，以及 inference 输出 raw type
但后续脚本按 official type 路由的接口断裂。因此本仓库不包含或包装官方旧脚本，只维护可运行、
明确标注偏差的内部统一协议。

## 源文件更名

| 服务器源组件 | 本仓库 canonical 组件 |
|---|---|
| `msmu_sft_baseline/data.py` | `benchmarks/msmu/data.py` |
| `msmu_sft_baseline/infer.py` | `models/qwen25_vl/peft_infer.py` |
| `validate_predictions.py` | `prediction_validation.py` |
| `score_official_compat.py` | `scorer.py` |
| `batch_msmu_official_infer.sh` | `scripts/msmu/infer_qwen_peft.sh` |
| `batch_msmu_sft_score_official_compat.sh` | `scripts/msmu/score_predictions.sh` |
| `serve_msmu_local_judge.sh` | `scripts/msmu/serve_local_judge.sh` |

原始 SHA-256 记录在 `docs/source-provenance.json`。
