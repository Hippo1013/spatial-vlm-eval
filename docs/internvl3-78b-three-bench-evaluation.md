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

GPU、端口或锁不可用时，`--check` 退出 `4`。正式运行前必须保证四张 GPU 空闲且三个 benchmark
的既有基线完整。

## 正式运行

```bash
bash scripts/internvl3_78b/run_three_bench_evaluation.sh
```

脚本自动完成共享 vLLM 启停、三个 benchmark 的 test/full、validator、评分和报告生成。

## 查看与恢复

```bash
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --status

# 中断后直接重新执行，脚本自动检查并恢复已有产物
bash scripts/internvl3_78b/run_three_bench_evaluation.sh
```

完成后确认：

- control root 的 `status.tsv` 最后一条为 `workflow final COMPLETE`；
- `q-spatial-result.md` 为 21/21；
- `spbench-si-result.md` 为 21/21；
- `cv-bench-result.md` 为 23/23。
