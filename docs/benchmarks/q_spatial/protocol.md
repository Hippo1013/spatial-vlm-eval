# Q-Spatial Bench canonical protocol

本文是本仓库 Q-Spatial Bench 输入、推理、校验、评分和发布语义的唯一规范。机器真值位于
`src/spatial_vlm_eval/benchmarks/q_spatial/`；两者冲突时停止运行并同时修复代码、测试与本文。

## 1. 锁定来源与数据合同

- 官方代码：[`andrewliao11/Q-Spatial-Bench-code`](https://github.com/andrewliao11/Q-Spatial-Bench-code)
  commit `ebe8137eae9781aaf7e29691ce8bc68b2a498a83`。
- 数据：[`andrewliao11/Q-Spatial-Bench`](https://huggingface.co/datasets/andrewliao11/Q-Spatial-Bench)
  revision `17b92e470d58fa46859ebd48ff35a1669828c9be`。
- 论文：仓库外未跟踪的 `benchmark_paper/Q-Spatial.pdf`。

正式运行必须显式提供两个数据根，不能从一个根猜另一个根，也不能把 ScanNet 许可内容复制进仓库：

```text
QSPATIAL_PARQUET_ROOT=/media/datasets/tangzecong/huggingface/dataset/Q-Spatial-Bench
QSPATIAL_SCANNET_RGB_ROOT=/media/datasets/lihaoran/huggingface/datasets/Q-Spatial-Bench/QSpatial_scannet/images
```

第一个根是 legacy 只读 Parquet；第二个根是已授权 ScanNet RGB 的显式入口。全局 index 固定为
QSpatial-ScanNet `0..169`、Q-Spatial++ `170..270`。两份 Parquet 分别为 170/101 行：

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/QSpatial_scannet-00000-of-00001.parquet` | 12,022 | `a5b0a37443b4ae18c837e4df7fe60411f869f282aa5803b8a7d509ba381286ba` |
| `data/QSpatial_plus-00000-of-00001.parquet` | 129,408,418 | `30ff075480f7fe0497122c8251f5d529f2241dda1387038e2a0ed802ae8615e2` |

两者 schema 必须恰为 `question`、`answer_value`、`answer_unit`、`question_type`、`image_path`、
`image`。ScanNet 为 66 scenes / 99 frames；外部 RGB 目录逐文件组合 SHA-256 是
`4485132ff448f43bdfb1283743825995823487a37f74ae4ab5a8e9d4b653751b`。Q-Spatial++ 有 101 个
QA、87 张不同图片；raw type 为 98 `horizontal_distance`、2 `vertical_distance`、1
`1d_horizontal`。最后一类只在派生统计中映射为 `object_width`，不改原始字段。

## 2. 防泄漏输入与 Prompt

adapter 只能收到：

```text
QSpatialModelInput(index, image, system_prompt, user_prompt)
```

其中 `image` 是恰好一张 RGB；`user_prompt` 是 `Question: {question}`；system prompt 的 UTF-8
SHA-256 必须为 `b3da32feb428a7840ecaf1d08ef095b9cd72ff6ef34d5b2b05ec1c1599bb613c`。完整文本为：

```text
You will be provided with a question and a 2D image. The question involves measuring the precise distance in 3D space through a 2D image. You will answer the question by providing a numeric answer consisting of a scalar and a distance unit in the format of """\scalar{scalar} \distance_unit{distance unit}""" at the end of your response.
```

支持 system role 的 backend 使用 system + user 两个 turn；官方 runner 不支持时按
`system_prompt + "\n\n" + user_prompt` 无损折叠，并记录 `system_role_supported=false`、模板摘要和
prompt SHA。任何 adapter 都不得看到 answer、unit、split、question type、完整 row 或同图其他问题。
prediction JSONL 每行只允许：

```json
{"index": 0, "raw_prediction": "..."}
```

空 prediction 保留并告警，评分为零；网络/runner 失败不能伪装成空 prediction。

## 3. 21 条推理轨

唯一机器清单是 `profiles.PROFILE_SEQUENCE`。它含 18 条 RGB 与 3 条派生输入轨，不含
3DThinker Mental-3D 或 SpatialLadder thinking prompt：

- LLaVA-NeXT 7B/34B：vLLM TP=1/2，greedy、512、seed 42；只对这两轨启用官方两阶段格式修复，
  第二次最多 64 tokens，两次都必须传同一张图片。
- InternVL3 8B/38B/78B：vLLM TP=1/2/4，BF16、greedy、512、seed 42；78B 必须是四张 80GB GPU。
- Qwen3-VL 2B/4B/8B/32B：TP=1/1/1/2；temperature 0.7、top-p 0.8、top-k 20、presence
  penalty 1.5、1024 tokens、每请求 seed 3407。
- GPT-5 / Gemini 3.1 Pro：OpenRouter 首方 non-ZDR、medium reasoning、16384 tokens；GPT-5 不发
  temperature，Gemini 为 0，provider 不保证确定性。
- SSR RGB/native、SpatialRGPT RGB、3DThinker RGB、SpatialBot RGB/ZoeDepth、RoboBrain2.5
  NV/MT、HiSpatial-3B MoGe-2 XYZ、SpatialLadder-3B RGB 按 registry 锁定的官方 runner、revision、
  decoding 与 input track 运行。HiSpatial 两个 split 都禁止 ScanNet GT depth，只用当前 RGB 经 MoGe-2
  估计 XYZ。

采样本地 backend 必须逐请求重置固定 base seed；无法做到的 specialized backend 只能单 persistent
runner。API 明确记录 `provider_nondeterministic=true`。本轮每轨只跑一次；多 seed 问题见
[遗留小问题](known-minor-issues.md)。

## 4. test gate 与 full

`--stage test` 先完成全量 contract、图片解码、fingerprint 与无泄漏检查，再执行 512×512 纯红/纯蓝
RGB canary。每个模型必须分别识别 red/blue，并证明恰好一张源 RGB 进入 adapter；普通 RGB 轨还必须
证明恰好一个 model image tensor/media。profile 锁定的派生轨必须分别证明派生模态来自同一张源 RGB；
其中 `spatialbot_zoedepth` 严格要求一个 RGB tensor 加一个同图派生 depth tensor，不视为第二张源图。
随后运行固定 smoke8：`0,1,3,9,14,205,247,250`。smoke 只门禁传输、模板、journal、subset validator
和输入审计；答案正确率与 numeric parser 状态只作诊断。

gate 绑定 dataset/prompt/profile/model/upstream revision、adapter/processor digest、decoding、seed、backend、
GPU、并发和 endpoint/sharding。full 必须读取同一绑定 gate；变化或过期即拒绝。所有 vLLM 轨都使用一个
endpoint：TP=1 endpoint 只绑定一张 GPU，并在 endpoint 内执行请求并发；TP=2/4 使用一个 tensor-parallel
endpoint。旧的 TP=1 双 endpoint/奇偶分片 gate 因 binding 不同自动失效，不允许迁移。已有完成 gate
因 binding 变化而失效时，test 先把旧 `test_artifacts/` 与 gate 原样轮换为带旧 binding
digest 的 `stale-*` 归档，再从空 test 目录建立新 signature；不得跨 signature 恢复或删除旧 journal。
vLLM `max_model_len` 也是 binding 的一部分；服务器默认 32768，以覆盖 Qwen3-VL 图像 token 后的
smoke prompt 与锁定的 1024 输出预算。LLaVA-NeXT Yi/Mistral 必须按 checkpoint 的合法 4096 上限
启动；调度器逐 profile 覆盖该值，不能把 32768 全局强加给这两条轨。
vLLM capacity 候选默认为 `32→16→8→4→2→1`；API 独立固定上限候选 `8→4→2→1`；specialized track 固定单
persistent runner。journal、resume signature、逐事件 fsync、原子 prediction 与 metadata 由公共 runtime
提供。controller 只清理自己记录的进程组，不接管端口、不终止已有 GPU 进程，也不自动评分。
3DThinker 仍接收同一张 RGB，但其 checkpoint processor 显式绑定 `12544..401408` pixels，避免
Q-Spatial++ 大图令 Qwen2.5-VL 视觉 attention 超出单卡显存；缩放发生在官方 processor 内并写入
adapter metadata/test binding，不产生额外图像或深度输入。

20 条当前双卡可运行轨的冻结调度协议是 `q_spatial_2xa800_staged_lanes_v1`：阶段 A 双卡 lane 顺序运行
InternVL3-38B、LLaVA-NeXT-Yi-34B、Qwen3-VL-32B；API lane 同时启动但严格串行 GPT-5、Gemini 3.1
Pro。双卡 lane 全部成功并释放 GPU 后，阶段 B 才并行启动 GPU 0/1 两条独立单卡 lane；它不等待 API
lane。InternVL3-78B 继续保持 TP=4 resource blocked。任一 lane 失败只停止本 lane；双卡 lane 失败会阻止
阶段 B，API 仍可继续。控制状态、冻结计划和分轨日志位于 `QSPATIAL_OUTPUT_ROOT/_scheduled_batch/`。
`--skip-completed` 只有在 271 行 validator、metadata、revision、protocol、dataset、binding 和当前 gate
全部复核后才跳过。
调度器的 `--stage test` 复用同一冻结分队、阶段屏障和失败隔离，但每轨在当前 gate 通过后立即结束，
绝不启动 full 或正式 validator；已有合法 gate 自动复用，`--skip-completed` 仅适用于 `--stage full`。

InternVL3-78B 只允许由 `run_internvl3_78b_evaluation.sh` 在四张 80GB GPU 上独立补测。该入口不改变
21 轨 registry，也不新建正式结果根：从当前 profile/binding/scorer protocol 解析 canonical 路径，
完成 test/full-271/validator 后精确评分该轨，并在原 `QSPATIAL_OUTPUT_ROOT` 原地重建同一个报告。
已有 prediction 只有在当前 inference/scorer 声明、revision、dataset、binding、gate 与 artifact hash
全部匹配时才可跳过模型推理。

## 5. Scorer protocol

当前 scorer protocol：

```text
q_spatial_robust_numeric_v2_standard_prompt_declared_final_equivalent_tags_controlled_wrappers_paper_inclusive_ratio
```

解析按以下顺序执行：

1. 优先检查模型明确声明的最终区域：最后一个 `<answer>...</answer>`、三引号、代码块、display/inline
   math、最后非空行，以及最后一个 `final answer` / `answer:` / `in conclusion` 后缀；只构造解析视图，
   `raw_prediction` 永不改写。
2. 标准 scalar/unit 标签保持最高证据等级。单组合法标签直接接受；多组标签只有全部换算后等价时才折叠。
   如果前文标签用于计算，最后明确 final 区域内恰有一个合法结果，则只接受该 final 声明；相互冲突、
   范围或多 final 仍判零。只有唯一 `distance_unit` 标签且其前紧邻一个 scalar 时也可接受。
3. 无有效完整标签时，受控识别真实模型已产生的 `N unit`、`N\unit`、`N{unit}`、`N, unit`、
   `\boxed{N \text{unit}}`、`distance{N \text{unit}}` 与 `\diameter{N} \units{unit}`；重复证据必须换算等价。
   `PS4` 和 `Region [0]` 这类对象/区域标识中的数字不作为竞争答案。
4. scalar 支持有限十进制和简单正分数；范围、复合表达式、科学计数法、多候选、缺单位或冲突答案均为零。
   明确的零/负数仍保留其模型声明值和单位用于审计，但因 Q-Spatial 真值和 ratio 要求正数而计零。
5. 支持 m/metre、cm、mm、ft/foot、inch 及单复数；pixels、角度和其他未知单位保留原值/单位但不换算、
   不得进入主分。
6. Decimal 确定性换算为厘米，`δ=max(pred/gt, gt/pred)`。主指标使用论文边界 `δ≤2`，同时记录
   `δ≤1.25`。
7. Overall 是 ScanNet 与 Q-Spatial++ 成功率等权平均；271 条 micro accuracy 仅作审计。

v2 是评分语义变更，但不改变任何已有模型输出。推理与评分协议保持分离：v2 scorer 只允许读取 inference
metadata 中声明的当前 v2 或历史 v1 scorer ID，并把原声明写入 summary；既有 v1 完整 prediction 不改写、
不需要重新推理，新推理默认声明 v2。其他未知 scorer 声明继续 fail closed。

同次评分还保存旧 notebook 审计：取最后一组标签、scalar 多数字求平均、未知单位按厘米、阈值严格
`<`；malformed 只形成 audit parse failure。`scored_rows.jsonl` 必须保留 raw output、两种解析状态、
换算值、GT、ratio、阈值、split/raw/canonical type 和主/旧差异。

## 6. Publication gates 与报告

subset 永不评分。目录评分先强制 full validator，再持有批次锁，只发现当前 scorer protocol 下未完成的
完整轨。summary、scored rows、prediction、metadata、revision、decoding、input track、prompt 和所有
hash 任一不一致都拒绝报告；同 profile 多个 publishable 候选也拒绝。

报告主表固定展示 21 轨的模型、两个 split 的 `δ≤2` 与 split-macro Overall；不再单列 input track 和
comparison group，而按 MSMU 汇总命名方式在同模型多输入轨或非默认派生输入的模型名后用中文括号标明
实际配置。未显式映射的双轨/派生输入必须 fail closed。另有 ScanNet 五类表和使用同一当前 scorer 的
`δ≤1.25` 严格阈值表。旧 notebook 兼容性审计只保留在逐行评分与 summary 产物中，不进入 Markdown
汇总。加粗仍只在同一 comparison group 内比较。报告必须分别显示 RGB `n/18` 与全轨 `n/21`，缺失
78B 或任何其他轨都要显式列出。

## 7. 结果性质

本实现锁定论文公式、公开发布资产与独立 v2 robust parser，但不是官方 notebook 的逐字节复制，因此结果
标记为 `official formula + robust numeric parser internal score`。任何 prompt、解析、单位、阈值、
聚合或 cache identity 变化都必须更换 scorer protocol、补回归测试并更新本文与 ADR。
