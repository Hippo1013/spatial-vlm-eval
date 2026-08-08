# InternVL3-78B 三 Benchmark 一键测评

用于四张 80GB GPU 服务器，一次加载 `OpenGVLab/InternVL3-78B-hf`，依次运行：

```text
Q-Spatial 271 -> SPBench-SI 1009 -> CV-Bench 2638
```

固定配置：vLLM `0.19.0`、BF16、TP=4、GPU `0,1,2,3`、served model
`internvl3-78b-three-bench`。

## 配置

服务器首次使用时，根据[公共配置模板](../configs/internvl3-78b-three-bench.env.example)将缺失项写入
仓库根目录未跟踪的 `.env.server`。入口会自动加载该文件，之后无需重复配置。

```bash
cd /media/datasets/lihaoran/latent_reasoning/spatial-vlm-eval
```

只有使用其他配置文件时才需要：

```bash
export INTERNVL3_78B_THREE_BENCH_ENV_FILE=/absolute/path/to/untracked-three-bench.env
```

## 检查

```bash
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --dry-run
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --status
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --check
```

GPU、端口或锁不可用时，`--check` 退出 `4`。正式运行前必须保证四张 GPU 空闲；其他模型的既有
评分或报告不再是 78B 推理与评分的前置门禁。

## 正式运行

```bash
bash scripts/internvl3_78b/run_three_bench_evaluation.sh
```

脚本自动完成共享 vLLM 启停，以及三个 benchmark 各自的 test/full、validator 和 78B 精确单轨评分。
每个 benchmark 独立决定是否继续汇总：

- Q-Spatial 既有报告源为 `20/21` 且只缺 `internvl3_78b` 时，评分后重建 `21/21`；
- SPBench-SI 既有报告源为 `20/21` 且只缺 `internvl3_78b` 时，评分后重建 `21/21`；
- CV-Bench 既有报告源为 `22/23` 且只缺 `internvl3_78b` 时，评分后重建 `23/23`；
- 任一 benchmark 未达到上述基线，或其他结果无法发现，只完成该 benchmark 的 78B 评分并记录
  `report=skipped`，不会阻塞后续 benchmark。

评分仍须通过该 benchmark 的完整 validator、summary、scored rows 和 publication gates；跳过报告不等于
跳过评分验收。

## 查看与恢复

```bash
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --status

# 中断后直接重新执行，脚本自动检查并恢复已有产物
bash scripts/internvl3_78b/run_three_bench_evaluation.sh
```

完成后确认：

- control root 的 `status.tsv` 最后一条为 `workflow final COMPLETE`；
- 三条 `internvl3_78b` score 均为 `complete`；
- `status.tsv` 分别标记 `report=complete` 或 `report=skipped`；
- 只有 `report=complete` 的 benchmark 才要求对应 Markdown 报告达到 21/21 或 23/23。
