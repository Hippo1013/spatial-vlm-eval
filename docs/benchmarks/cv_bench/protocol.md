# CV-Bench Canonical Protocol

本文件是仓库内 CV-Bench 数据、模型输入、prediction、评分和发布门禁的唯一规范。实现真值位于
`src/spatial_vlm_eval/benchmarks/cv_bench/`；二者冲突时必须停止运行并一并修复代码、测试和本文。

## 上游身份

- 数据集：[nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench)，revision
  `bc284db50d036958861cb60cdd7b77612052ce0d`。
- 官方评测参考：[cambrian-mllm/cambrian](https://github.com/cambrian-mllm/cambrian/tree/539ffc3254bba004e5d012b65c0ad6cb308897c5/eval)，
  commit `539ffc3254bba004e5d012b65c0ad6cb308897c5`。
- 论文：仓库外未跟踪的 `benchmark_paper/CV-Bench.pdf`。论文、数据和模型权重不提交 Git。

锁定数据由两个 Parquet 按 2D 后 3D 的顺序拼接，形成全局 index `0..2637`：

服务器 legacy 只读位置于 2026-08-03 核验为
`/media/datasets/tangzecong/huggingface/datasets/CV-Bench`；新下载不得写入该 namespace。

| 文件 | 行数 | 字节数 | SHA-256 |
|---|---:|---:|---|
| `test_2d.parquet` | 1438 | 184906137 | `33196034ef4bf3265cae4a7ff5c4071b2ff1cc21123e8e285c6a91393897ecbc` |
| `test_3d.parquet` | 1200 | 219902227 | `ef91fe8b5392eb2a16e318ca68fa02449d45ba1e152afece12a0a526e9fbbc25` |

contract 同时校验 schema、文件哈希、行数、RGB 解码、prompt 重建、任务/来源分布、逐图像素摘要和
全数据 fingerprint。固定分布为 Relation 650、Count 788、Depth 600、Distance 600；ADE20K 633、
COCO 805、Omni3D 1200。

## 模型输入边界

`CVBenchTestContract` 私有持有完整数据行。adapter 只能收到冻结对象：

```text
CVBenchModelInput(index, one RGB image, question)
```

其中 `question` 是数据集 `prompt`（题目和有序选项）加上官方直接答题后缀：

```text
Answer with the option's letter from the given choices directly.
```

adapter 不得接触 `answer`、`task`、`source`、bbox 或其他评分字段。每个调用必须在 journal 或 processor
审计中证明 prompt、图片 mode/尺寸/像素 SHA-256、模板摘要，以及恰好一个 media prompt 或模型图像
tensor。SpatialRGPT 不伪造 region/mask/depth；HiSpatial 只建立其架构合法的 RGB + MoGe-2 XYZ 轨。

## 推理 profile 与测试门禁

23 条目标轨及其 revision、input track、decoding 和 inference protocol 由
`benchmarks.cv_bench.profiles.PROFILE_SEQUENCE` / `PROFILES` 唯一维护，展示矩阵见
[模型矩阵](../../model-matrix.md#cv-bench-当前-23-条目标-inference-profile)。

通用开源模型锁定 greedy、`temperature=0`、`top_p=None`、beam 1、512 tokens、seed 42。OpenRouter
两轨锁定首方 non-ZDR、禁止 provider fallback、medium reasoning 和 16384 completion budget；GPT-5
不发送 temperature，Gemini 发送 0。专用模型使用各自上游 generation kwargs；SSR、3DThinker direct
和 SpatialLadder 中无法由 registry 静态确定的字段必须由绑定 checkpoint/upstream 的 generation
manifest 解析，缺失时 fail closed。

`--stage test` 必须完成：

1. 锁定数据的完整 fingerprint/schema/prompt/image 审计；
2. 最低视觉接收门禁：分别输入一张 512×512 纯红 RGB 图和一张纯蓝 RGB 图并询问颜色，两次回答
   必须分别唯一指向 red 与 blue，且都证明模型边界恰好接收一张图；不再测试形状、方位或空间描述
   能力。已通过旧版红圆/蓝方块严格 canary 的结果可在逐项复核答案、单图计数、smoke8 和其余绑定
   后迁移为当前 gate，并明确记录 `stricter_legacy_evidence`，无需重新调用模型；
3. 固定 smoke8：`0,633,342,1080,1438,1442,2038,2042`，四任务各两条并覆盖三个来源；
4. Transformers processor/template 的单图审计；vLLM 不一致时只能回退到显式锁定的 upstream runner；
5. vLLM 并发候选 `32,16,8,4,2,1` 容量探测；
6. subset validator、模型边界单图计数和只读 GPU inventory/process 审计。

测试 gate 绑定 dataset revision/fingerprint、模型 revision、profile registry digest、adapter digest、
upstream commit、processor 摘要、decoding、sharding 和显式 GPU selection。任一字段改变，full 阶段拒绝
旧 gate。InternVL3-78B 只能在明确枚举四张至少 79000 MiB 的 GPU 后以 TP=4 运行，不做量化替代。

## Prediction 与 validator

`predictions.jsonl` 每行严格只有：

```json
{"index":0,"raw_prediction":"A"}
```

真值只在评分时按 index 从锁定数据重新关联。正式 prediction 必须无重复地精确覆盖 `0..2637`；空
prediction 保留并告警。subset 只能出现在 `test_runs/`，即使 validator 通过也不得评分或发布。

## 评分

当前 scorer protocol：

```text
cv_bench_robust_mcq_v2_answer_tag_unique_letter_or_exact_option_text
```

解析只接受唯一合法的显式选项字母，或与单个选项完整匹配的文本；带官方 thinking prompt 的轨可用
唯一完整 `<answer>...</answer>` 界定最终答案，多个 answer tag 仍判无效。字母和文本同时出现且指向同一
选项时有效；冲突、多答案、越界、空值或无法解析均记零分。`scored_rows.jsonl` 保存
`parsed_answer`、`parse_status`、gold、correctness 和分组字段，不覆盖原始输出。

本实现沿用官方指标定义，但解析器比 Cambrian 旧版的首字符解析更严格稳健，因此结果类型是
`cv_bench_official_formula_robust_parser_internal_score`，不是旧脚本的逐字节复刻。聚合固定为：

```text
2D      = (ADE20K accuracy + COCO accuracy) / 2
3D      = Omni3D accuracy
Overall = (2D + 3D) / 2
```

2638 条 micro accuracy 只写入 `micro_accuracy_audit_only`，不得作为 Overall。报告额外显示 Spatial
Relationship、Object Count、Depth Order 和 Relative Distance 四项准确率。

## 产物与发布

正式输出位于仓库外的 `CVBENCH_OUTPUT_ROOT`：

```text
runs/PROFILE/MODEL_REVISION/INFERENCE_PROTOCOL/
├── test_gate.json
├── predictions.jsonl
├── predictions.jsonl.journal.jsonl
├── predictions.jsonl.metadata.json
├── prediction_validation.json
└── scores/SCORER_PROTOCOL/
    ├── prediction_validation.json
    ├── scored_rows.jsonl
    ├── summary.json
    └── publication_gates.json
```

目录驱动评分不得维护模型名单。报告只发现当前 scorer protocol 下通过 publication gates 的完整结果，
并逐行校验 model revision、input track、decoding 和 inference protocol；同一 profile 有多个可发布结果
时 fail closed。只有 23 条轨全部通过时，`cv-bench-result.md` 才标为“目标矩阵完整”。

HiSpatial 上游 CV 脚本只覆盖 2D Relation 和 3D；本项目为了统一矩阵，对合法 RGB 派生 XYZ 输入运行
完整 2638 条并明确记录这一偏差。专用 runner 的实现不由 benchmark 模块猜测：服务器必须提供锁定
上游环境中的 persistent JSONL runner、实现 SHA-256 和（需要时）generation manifest。
