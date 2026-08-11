# MSMU-Bench Evaluation Protocol

## 协议身份

当前 canonical scorer：

```text
sdvlm_official_compat_local_judge_v4_grounding_split_strict_quant_length_malformed_zero
```

准确描述为：

> SD-VLM official-compatible thresholds/macro-8 + local judge + grounding split fix + strict quantitative length + malformed judge response zero fallback

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

当前补测登记 `Qwen3-VL-2B-Instruct`、`4B-Instruct`、`8B-Instruct` 和 `32B-Instruct`。四者只用
Instruct 权重，不使用 Thinking 或量化变体；model revision、inference protocol 与输出目录独立。

| Setting | Qwen3-VL current supplement | Retained Qwen2.5-VL |
|---|---|---|
| Chat template | Qwen3-VL native structured-image template | Qwen2.5-VL native structured-image template |
| System message | none | native implicit `You are a helpful assistant.` |
| Image pixels | 16384..147456 | 12544..112896 |
| Spatial factor | 32 | 28 |
| Merged visual-token budget | 16..144 | 16..144 |
| Sampling | disabled | disabled |
| Beams | 1 | 1 |
| Max new tokens | 192 | 192 |
| Output order | original index, 0..986 | original index, 0..986 |

Qwen3-VL 官方模型卡推荐 sampling 和更大的默认图像范围；本项目为统一横评锁定 greedy/192 tokens，
并按 Qwen3-VL 的 32-pixel spatial factor 把图像限制为等 visual-token 预算。这不是 Qwen 官方推荐
generation profile，也不是 SD-VLM 官方的 temperature 0.2/1024-token profile；三者不得混表。
Qwen2.5-VL 7B/32B/72B adapter 与已有结果只为复现保留，不属于当前四模型补测范围。

## 多模型 inference protocol

scorer protocol 与 inference protocol 是两个正交身份。此次新增 adapter 不改变 judge prompt、阈值、
列表长度语义、grounding 路由、cache key 或 macro-8 聚合，因此 canonical scorer id 保持不变。
prompt、图像派生组件或 decoding 不同的轨使用独立 inference protocol：

| Profile | Legal model input | Decoding | Inference protocol |
|---|---|---|---|
| GPT-5 | one RGB + original question | low reasoning, no temperature, 192 completion tokens | `msmu_gpt5_question_only_v1` |
| GPT-5 OpenRouter non-ZDR | one RGB + original question；OpenAI only、无 fallback、`data_collection=deny`、不要求 ZDR | medium reasoning, no temperature, 16384 total completion tokens | `msmu_gpt5_question_only_openrouter_non_zdr_v3_medium_16384` |
| Gemini 3.1 Pro preview | one RGB + original question | low reasoning, temperature 0, 192 completion tokens | `msmu_gemini31pro_question_only_v1` |
| Gemini 3.1 Pro preview OpenRouter non-ZDR | one RGB + original question；Google AI Studio only、无 fallback、`data_collection=deny`、不要求 ZDR | medium reasoning, temperature 0, 16384 total completion tokens | `msmu_gemini31pro_question_only_openrouter_non_zdr_v3_medium_16384` |
| LLaVA-NeXT Mistral 7B | one RGB + original question | greedy, 192 | `msmu_llava_next_mistral_7b_question_only_v1` |
| LLaVA-NeXT Yi 34B | one RGB + original question | greedy, 192 | `msmu_llava_next_yi_34b_question_only_v1` |
| InternVL3 8B / 38B / 78B | one RGB + original question | greedy, 192 | model-size-specific `msmu_internvl3_*_question_only_v1` |
| Qwen2.5-VL 7B / 32B / 72B | one RGB + original question；structured image content | greedy, 192；72B 双卡 balanced | model-size-specific `msmu_qwen25_vl*_question_only_deterministic_v1` |
| Qwen3-VL 2B / 4B / 8B / 32B | one RGB + original question；structured image content；no system message | greedy, 192；32B batch size 1 | model-size-specific `msmu_qwen3_vl_*_question_only_deterministic_v1` |
| SSR fair | one RGB + original question；无 TOR/MIDI/depth | greedy, 192 | `msmu_ssr_rgb_only_v1` |
| SSR native | one RGB + original question；same-RGB DepthPro + MIDI + 10 TOR | greedy, 192 | `msmu_ssr_native_depthpro_midi_tor10_native_v1` |
| SpatialRGPT | one RGB + original question；无 region/mask/depth | greedy, 192 | `msmu_spatialrgpt_rgb_only_v1` |
| 3DThinker fair | one RGB + original question | greedy, 192 | `msmu_3dthinker_question_only_v1` |
| 3DThinker native | one RGB + original question + official begin-position mental-3D control prompt | greedy, 2048；last complete answer tag | `msmu_3dthinker_native_mental3d_native_v1` |
| SpatialBot fair | one RGB + original question | greedy, 192 | `msmu_spatialbot_rgb_only_v1` |
| SpatialBot native | one RGB + original question；same-RGB ZoeDepth uint16-mm derived depth | greedy, 192 | `msmu_spatialbot_native_zoedepth_rgbd_native_v1` |
| RoboBrain2.5-8B-NV | one RGB + original question；official `general` structured image | sampling，temperature 0.7、top-p 0.8、768、seed 42 | `msmu_robobrain25_8b_nv_rgb_original_first_question_official_general_sampling_t07_top_p08_768_v1` |
| RoboBrain2.5-8B-MT | one RGB + original question；独立 MT checkpoint | sampling，temperature 0.7、top-p 0.8、768、seed 42 | `msmu_robobrain25_8b_mt_rgb_original_first_question_official_general_sampling_t07_top_p08_768_v1` |
| HiSpatial-3B | current MSMU RGB + 仅由同图 MoGe-2 派生的 XYZ；禁止 GT/题型信息 | official predictor greedy、100、seed 42 | `msmu_hispatial3b_same_rgb_moge2_xyz_original_first_question_official_predictor_greedy100_v1` |
| SpatialLadder-3B direct | one RGB + original question；不加 post prompt | BF16/FA2、left-padded native batch、temperature 0.01、top-p 1、repetition 1.05、128、seed 42 | `msmu_spatialladder3b_rgb_original_first_question_direct_flashattn2_leftpad_native_batch_128_v1` |
| SpatialLadder-3B thinking | one RGB + official generic `THINKING_TEMPLATE` + `special_post_prompt`；不使用选择题/数值题型模板 | 同上但 1024；最后完整 answer tag | `msmu_spatialladder3b_rgb_official_generic_special_thinking_flashattn2_leftpad_native_batch_last_answer_1024_v1` |

3DThinker native 缺少完整 `<answer>...</answer>` 时保留 raw response 作为 prediction 并写 warning，不能
静默返回空值。SpatialBot native 的 depth 只能从当前 RGB 估计，不接受 GT/sensor depth。SpatialRGPT
因 MSMU 无 region/mask 且题干无 region token，不运行 detector 伪造 region。

SpatialBot depth 合同明确为 `clip(round(metres * 1000), 0, 65535)` 后做上游三通道 packing。锁定
ZoeDepth 的 `save_raw_16bit` helper 使用 `depth * 256`，所以本项目的毫米量化不是该 helper 的字节级
复刻；该已知偏差属于当前 inference protocol，若改为 `*256` 必须更换 inference protocol id 和测试。

上述五条 MSMU SOTA supplement 只复用 model-family 的官方 processor/predictor 技术，不继承
CV-Bench、Q-Spatial 或 SPBench-SI 的 benchmark prompt、schema、validator、scorer 或报告语义。
adapter 的运行输入仍只有 `index/image/question`。RoboBrain 严格走锁定上游 `general` 路由；HiSpatial
必须记录同一源 RGB digest、派生 XYZ digest、HiSpatial/MoGe-2/utils3d revision，且不能接收 MSMU
reference、raw type 或 task family。SpatialLadder 保留 tied embeddings、官方像素范围
`12544..401408`、BF16/FlashAttention2 与 tokenizer left padding；native batch canary 使用异长 red/blue
prompt 做 `16→8→4→2→1` 容量探测。thinking 轨只取最后一个完整 `<answer>...</answer>` 内容；没有
完整标签时原始响应成为 prediction，同时 journal 记录 warning、原始响应 SHA-256、字符数和抽取状态。
它是补充轨，不进入 MSMU 主矩阵完成数。

五条轨的 inference protocol 完全独立，但 scorer、阈值、judge prompt、cache key 和 macro-8 聚合均不
改变，继续使用本页 canonical scorer protocol。任何一个新 prompt、派生输入或 decoding 的 journal
都不得被另一条轨恢复。

OpenAI-compatible 请求严格只有一个 user message，其中一个 question text part 和一个 PNG data URI；
不发送 system、history、reference 或答案格式提示。OpenRouter 路由同时设置首方 provider only、
`allow_fallbacks=false`、`require_parameters=true`、`data_collection=deny` 和 `zdr=true`，并在成功写
journal 前校验 generation metadata 的 canonical model、provider 和 `num_media_prompt==1`。请求使用
OpenRouter model alias，但返回身份精确锁定 catalog canonical revision：GPT-5 为
`openai/gpt-5-2025-08-07`，Gemini 3.1 Pro preview 为
`google/gemini-3.1-pro-preview-20260219`；别名本身或其他 revision 均不得通过校验。

当目标首方 endpoint 没有 ZDR 路由时，只能在用户明确同意后使用独立的
`gpt5_openrouter_non_zdr` / `gemini31pro_openrouter_non_zdr` profile。两条例外轨仅把请求级
`zdr` 设为 false；首方 provider、无 fallback、参数完整和 `data_collection=deny` 保持不变。live canary
确认 reasoning model 的 completion 上限同时包含 hidden reasoning：v1 的 192 tokens 被 GPT-5 全部用作
reasoning 并产生空文本。non-ZDR v2 曾保留 low reasoning、将总 completion budget 提升到 512，但正式
运行仍出现 hidden reasoning 耗尽预算的空 prediction 和被截断回答。当前两条 non-ZDR v3 能力轨均按
用户确认锁定为 medium reasoning、16384 total completion tokens；GPT-5 的选择同时与 EASI 对同一
`gpt-5-2025-08-07` revision 的正式空间评测设置一致。原 ZDR/v1 decoding 保持不变。各轨使用独立
inference protocol、run slug、journal 和输出目录，不得恢复或覆盖其他 decoding 轨的 journal。

阶段一的非 MSMU 组合视觉 canary 固定使用一张 512×512 抗锯齿白底 RGB：左上红圆、右下蓝方块；
位置证据可使用正确关联的英文方位词或合法归一化 bbox，二者同时出现时不得冲突；
图像由 4× 超采样后 LANCZOS 缩小确定性生成。问题只要求描述每个彩色形状及位置，不包含正确答案。
回答必须同时建立 red-circle/top-left 与
blue-square/bottom-right 两个语义对应，否则 fail closed。该诊断对每个模型只做一次 generation，不写入
MSMU prediction journal、不参与评分，也不改变 inference/scorer protocol。

SOTA supplement 另由 `scripts/msmu/run_sota_supplement.sh` 冻结双 lane：GPU0 为 RoboBrain NV →
HiSpatial → SpatialLadder direct，GPU1 为 RoboBrain MT → SpatialLadder thinking。每条 lane 内依次通过
组合视觉 canary、固定 smoke8 和 full-987；两条 lane 都 COMPLETE 且五份 prediction/validator/metadata
合法后，才释放推理模型并只启动一次本地 judge，按 NV、MT、HiSpatial、direct、thinking 顺序评分。
两个 watcher 只由继承 pipe 的状态事件唤醒并读取仓库外 `status.tsv`，不调用 LLM。任一 lane 失败只
终止控制器自有进程组；合法 journal/正式产物保留，非法 finalized 产物原地 fail closed 且不覆盖。

vLLM 0.19 服务必须设置 `--limit-mm-per-prompt.image 1`、锁定 revision/served name/TP/dtype。静态
preflight 分别检查 LLaVA prompt 中恰有一个 `<image>`、InternVL prompt 中恰有一个
`<IMG_CONTEXT>`，且 processor 返回非空 `pixel_values`；服务启动后还必须通过上述组合视觉 canary。
Qwen stage 1 通过同一个已加载模型和 processor 执行该检查并要求恰好一个 image tensor。GPT-5/Gemini
stage 1 则通过同一个 OpenAI-compatible adapter 执行，并继续应用 OpenRouter 的首方 provider、精确
canonical model 和 `num_media_prompt==1` 门禁。

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

结果表的八类列顺序与官方论文 Table 1 一致，表头使用中文，数值以百分比保留两位小数。展示表仅
允许一个 scorer protocol；默认当前 canonical v4，多 scorer protocol 请求必须失败，不能把历史与
当前评分静默混合。表格只显示模型名称、八类指标和平均值，专用模型直接在模型名称中写实际输入或
提示配置：SSR 为 `RGB` / `RGB + 深度估计`，SpatialRGPT 保持模型原名且不加注释，3DThinker 为
`RGB` / `RGB + Mental-3D 提示词`，SpatialBot 为 `RGB` / `RGB + 深度估计`。标题下固定用一行注释
说明括号内容的含义，并声明“RGB + 深度估计”中的深度由当前 MSMU RGB 图像估算，不使用 GT 深度、
reference 或额外标注。精确 revision、inference protocol、scorer protocol 和结果性质仍由生成前
强制校验的 metadata、summary 与结果目录追溯，不得从展示表反推。未知双轨 profile 若没有显式展示
配置必须 fail closed，不能退回含混的“公平版/原生版”。

SOTA supplement 的固定展示包含五行：RoboBrain NV/MT 为 `RGB`，HiSpatial 为
`RGB + MoGe-2 XYZ`，SpatialLadder 为 `RGB / direct` 与
`RGB + 官方通用 thinking 提示词`。报告写入前的 `--check` 必须确认既有 18 行和新增 5 行在当前 scorer
protocol 下各有且只有一个完整 publication-gated summary；成功后才原子重建 23 行报告。

## Cache 与产物

cache key 包含 protocol、official type、question、reference、prediction、完整 judge prompt、judge
model 和 endpoint。更换上述任一内容不会复用旧 response。每个模型和协议仍必须使用独立目录。

能够解析且满足对应任务响应 schema 的 judge 结果正常写入 `judge_cache.jsonl`。完成配置的重试后，
如果 judge endpoint 已经返回文本，但该文本仍无法解析或不满足当前任务的响应 schema，则按服务器
参考 scorer 的保守逻辑，将它标记为 `malformed_judge_response_zero`、写入 cache，并在所有 judge
路径中把该样本记为 0 分。该 fallback 覆盖非 grounding 数值、coordinate grounding、
object-at-coordinate grounding 和定性任务；它不进入 `judge_failures.jsonl`，因此不会阻断后续样本
或整轨 publication。summary 必须报告 fallback 数量与 index，逐样本记录必须保留解析错误和 judge
原始文本，便于以后人工复核。

如果所有重试均没有得到任何 judge 文本，例如连接失败或超时，则仍写入
`judge_failures.jsonl`，不得缓存或记成模型 0 分。未处理的 worker 异常也保持 hard failure。只要
存在这类未解决基础设施失败，scorer 必须写出 `publishable == false` 的诊断 summary 并以非零状态
退出。

上述 fallback 改变了 malformed judge response 的评分与 cache 语义，因此 scorer protocol 从 v3
升级为 v4，judge cache protocol 升级为
`sdvlm_official_compat_local_judge_v3_grounding_split_malformed_zero`。旧 protocol 的 summary、
scored rows 和 judge cache 不得作为新 protocol 结果复用。

judge JSON 解析优先接受完整对象。确定性恢复仅额外接受响应开头、带引号且独占第一行的
`"your_mark": 0` 或 `"your_mark": 1`，允许其后存在解释文字；不得从任意 prose 中搜索 mark，也
不得通过该分支接受其他 key、`null`、小数或 `0/1` 之外的值。该恢复只读取 judge 已明确给出的
离散 mark，不改变 judge response、评分阈值或合法响应语义，因此保留当前 scorer/cache protocol id。
每次触发该恢复都会在 `score.log` 和上层串行批次日志中记录 index、恢复策略、离散 mark 和忽略
尾随文字的事实；正常 JSON 响应不写恢复 warning，日志也不重复保存完整 judge response。

评分目录输出：

- `prediction_validation.json`
- `judge_cache.jsonl`
- `judge_failures.jsonl`
- `scored_rows.jsonl`
- `summary.json`
- 运行日志

## 与 strict official 的差异

当前实现与官方至少有以下差异：Qwen 原生 prompt/项目锁定图像范围、greedy decoding、本地 Qwen
judge、JSON-only 约束、judge temperature、容错解析和 grounding object prompt/语义评分。保留的
Qwen2.5-VL 轨还有原生隐式 system prompt；Qwen3-VL 轨不添加 system message。

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
