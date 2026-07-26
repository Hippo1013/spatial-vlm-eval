# Repository Architecture

## 设计原则

仓库采用两个正交扩展轴：benchmark 定义“什么输入合法、如何校验、如何评分”，模型适配器定义
“如何把同一条 benchmark 样本变成模型原生输入并生成文本”。shell 脚本仅编排两者。

```text
Benchmark dataset
  → benchmark data adapter
  → model-family inference adapter
  → predictions.jsonl
  → benchmark validator (mandatory)
  → benchmark scorer / judge
  → scored rows + summary
```

这种分层避免为每个“模型 × benchmark”复制一套评分代码，也避免模型专用 prompt 污染 benchmark
协议。

## Python 包

### `spatial_vlm_eval.benchmarks`

每个子包拥有：

- 数据读取和 official type 映射；
- prediction schema；
- 完整性与 provenance 校验；
- judge prompt、cache identity、阈值与汇总；
- protocol 常量。

### `spatial_vlm_eval.models`

每个子包拥有：

- 官方 processor/tokenizer/chat template；
- structured image/video 输入；
- base/checkpoint 加载；
- generation profile；
- 推理运行 metadata。

模型适配器可以依赖 benchmark 的数据接口，但 scorer 不得依赖具体模型。

## shell 层

`scripts/<benchmark>/` 提供可审计入口：

- `infer_<model>.sh`
- `validate_predictions.sh`
- `serve_local_judge.sh`（若需要）
- `score_predictions.sh`
- `run_<model>_pipeline.sh`

shell 层不实现指标，不自动下载模型或数据，不在仓库内保存凭证。

## 输出目录建议

```text
outputs/<benchmark>/<model>/<run_id>/
├── predictions.jsonl
├── predictions.jsonl.metadata.json
├── prediction_validation.json
├── infer.log
└── scores/<protocol>/
    ├── prediction_validation.json
    ├── judge_cache.jsonl
    ├── scored_rows.jsonl
    ├── summary.json
    └── score.log
```

生成物不进入 Git；可使用对象存储、实验平台或 GitHub Release 单独归档。
