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

## Prediction JSONL

每行包含：

```json
{"index":0,"raw_type":"width","task_family":"scale_estimation","question":"What is the width of the wall shelves?","reference":"the wall shelves measures 0.37 meters in width.","prediction":"The shelves are about 0.4 meters wide."}
```

前五个字段由 official row 和固定映射生成，只有 `prediction` 来自模型。输出中的 question 删除
字面 `<image>` 并做首尾 whitespace 清理；reference 做首尾 whitespace 清理。不得手写或改述。

## 强制校验

校验器检查：

- 恰好 987 条、index 为 0..986 且无重复；
- raw type、task family、question、reference 与 test row 一致；
- 八类数量正确；
- 空 prediction 明确列入 warnings。

空 prediction 不使 `passed` 变为 false，评分时会得到零分或抽取失败。其他结构/provenance 错误
均为 hard error。`--allow-subset` 仅用于独立 validator 的调试，scorer 固定执行 full-split 校验，
无法用该参数绕过。

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

`micro_accuracy` 只作补充。正式 summary 必须满足 `num_samples == 987` 且
`missing_official_types == []`。

## Cache 与产物

cache key 包含 protocol、official type、question、reference、prediction、完整 judge prompt、judge
model 和 endpoint。更换上述任一内容不会复用旧 response。每个模型和协议仍必须使用独立目录。

评分目录输出：

- `prediction_validation.json`
- `judge_cache.jsonl`
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
