# Q-Spatial Bench canonical protocol

本文是本仓库 Q-Spatial Bench 输入、推理、校验、评分和发布语义的唯一规范。机器真值位于
`src/spatial_vlm_eval/benchmarks/q_spatial/`；两者冲突时停止运行并同时修复代码、测试与本文。

## 1. 锁定来源与数据合同

- 官方代码：[`andrewliao11/Q-Spatial-Bench-code`](https://github.com/andrewliao11/Q-Spatial-Bench-code)
  commit `ebe8137eae9781aaf7e29691ce8bc68b2a498a83`。
- 数据：[`andrewliao11/Q-Spatial-Bench`](https://huggingface.co/datasets/andrewliao11/Q-Spatial-Bench)
  revision `17b92e470d58fa46859ebd48ff35a1669828c9be`。
- 论文：本地 [`benchmark_paper/Q-Spatial.pdf`](../../../benchmark_paper/Q-Spatial.pdf)。

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
RGB canary。每个模型必须分别识别 red/blue，并提供恰好一张图进入 processor/tensor/media 边界的
证据。随后运行固定 smoke8：`0,1,3,9,14,205,247,250`。smoke 只门禁传输、模板、journal、subset
validator 和输入审计；答案正确率与 numeric parser 状态只作诊断。

gate 绑定 dataset/prompt/profile/model/upstream revision、adapter/processor digest、decoding、seed、backend、
GPU、并发和 sharding。full 必须读取同一绑定 gate；变化或过期即拒绝。TP=1 vLLM 使用两个独立 endpoint
固定奇偶分片；TP>1 使用单 endpoint。journal、resume signature、逐事件 fsync、原子 prediction 与 metadata
由公共 runtime 提供。controller 只清理自己启动的 adapter/runner，不接管端口、不终止已有 GPU 进程。

## 5. Scorer protocol

当前 scorer protocol：

```text
q_spatial_robust_numeric_v1_standard_prompt_tag_first_unique_fallback_paper_inclusive_ratio
```

解析按以下顺序执行：

1. 输出只要出现 scalar/unit 标签痕迹即进入标签模式；partial、数量不一或多组标签直接无效，不 fallback。
2. 标签模式要求恰好一个 scalar 和一个 unit；scalar 只能是一个正有限十进制数。
3. 完全无标签时依次优先取最后一个 `final answer`、`answer:`、`in conclusion` 后的区域；均无则只看
   最后一个非空行。
4. fallback 区域必须只有一个相邻 numeric-unit pair；范围、复合单位、多候选和冲突答案均为零。
5. 支持 m/metre、cm、mm、ft/foot、inch 及单复数；未知单位、零、负数、分数和科学计数法为零。
6. Decimal 确定性换算为厘米，`δ=max(pred/gt, gt/pred)`。主指标使用论文边界 `δ≤2`，同时记录
   `δ≤1.25`。
7. Overall 是 ScanNet 与 Q-Spatial++ 成功率等权平均；271 条 micro accuracy 仅作审计。

同次评分还保存旧 notebook 审计：取最后一组标签、scalar 多数字求平均、未知单位按厘米、阈值严格
`<`；malformed 只形成 audit parse failure。`scored_rows.jsonl` 必须保留 raw output、两种解析状态、
换算值、GT、ratio、阈值、split/raw/canonical type 和主/旧差异。

## 6. Publication gates 与报告

subset 永不评分。目录评分先强制 full validator，再持有批次锁，只发现当前 scorer protocol 下未完成的
完整轨。summary、scored rows、prediction、metadata、revision、decoding、input track、prompt 和所有
hash 任一不一致都拒绝报告；同 profile 多个 publishable 候选也拒绝。

报告主表固定展示 21 轨的模型、input track、comparison group、两个 split 的 `δ≤2` 与 split-macro
Overall；另有 ScanNet 五类表以及 `δ≤1.25`、旧 notebook 和差异 profile 审计。加粗只在同一
comparison group 内比较。报告必须分别显示 RGB `n/18` 与全轨 `n/21`，缺失 78B 或任何其他轨都要
显式列出。

## 7. 结果性质

本实现锁定论文公式、公开发布资产与独立 robust parser，但不是官方 notebook 的逐字节复制，因此结果
标记为 `official formula + robust numeric parser internal score`。任何 prompt、解析、单位、阈值、
聚合或 cache identity 变化都必须更换 scorer protocol、补回归测试并更新本文与 ADR。
