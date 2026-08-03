# CV-Bench 简明运行指令

详细前置条件见 [两阶段 runbook](cv-bench-two-stage-runbook.md)。以下命令均在仓库根目录运行。

## 1. 准备

```bash
conda env list
test -e .env.cvbench.server || cp configs/cv-bench-server.env.example .env.cvbench.server
export CVBENCH_ENV_FILE="$PWD/.env.cvbench.server"
```

核对模板中的 legacy `CVBENCH_DATASET_ROOT`，再填好模型/endpoint 或专用 runner 配置。

## 2. 查看与检查

```bash
bash scripts/cv_bench/run_inference.sh --list
bash scripts/cv_bench/run_inference.sh --check --model PROFILE
bash scripts/cv_bench/run_inference.sh --status
```

## 3. 测试阶段

```bash
bash scripts/cv_bench/run_inference.sh --stage test --model PROFILE
bash scripts/cv_bench/run_inference.sh --stage test --models PROFILE1,PROFILE2
bash scripts/cv_bench/run_inference.sh --stage test --all
```

test 会依次完成 GPU/processor、视觉 canary、smoke8、输入审计和绑定 gate。未通过 gate 不得跑 full。

## 4. 正式 2638 条

```bash
bash scripts/cv_bench/run_inference.sh --stage full --model PROFILE
bash scripts/cv_bench/run_inference.sh --stage full --models PROFILE1,PROFILE2
bash scripts/cv_bench/run_inference.sh --stage full --all
```

多模型按 registry 顺序串行；本轮实施不自动执行这些 full 命令。

## 5. 评分

只评分一个刚完成的结果：

```bash
bash scripts/cv_bench/score_results.sh --predictions /absolute/path/to/predictions.jsonl
```

查看并评分输出根中的全部待评分结果：

```bash
bash scripts/cv_bench/score_results.sh --list
bash scripts/cv_bench/score_results.sh
```

## 6. 汇总

```bash
bash scripts/cv_bench/build_results_report.sh --check
bash scripts/cv_bench/build_results_report.sh
```

报告写入 `$CVBENCH_OUTPUT_ROOT/cv-bench-result.md`；只有 23/23 条轨通过发布门禁才标记矩阵完整。
