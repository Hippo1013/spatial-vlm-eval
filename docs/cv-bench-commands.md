# CV-Bench 简明运行指令

详细前置条件见 [两阶段 runbook](cv-bench-two-stage-runbook.md)。以下命令均在仓库根目录运行。

## 1. 准备

```bash
conda env list
test -e .env.cvbench.server || cp configs/cv-bench-server.env.example .env.cvbench.server
export CVBENCH_ENV_FILE="$PWD/.env.cvbench.server"
```

核对模板中的 legacy `CVBENCH_DATASET_ROOT`，再填好模型/endpoint 或专用 runner 配置。
专用 runner 的一次性 digest/command 配置示例见[两阶段 runbook 的第 2 节](cv-bench-two-stage-runbook.md#2-专用模型-runner)。

OpenRouter 两条轨的 key 用隐藏输入工具写入服务器的未跟踪 `.env.server`，不要把 key 放进命令参数：

```bash
ssh -t msmu-a800 'cd /media/datasets/lihaoran/latent_reasoning/spatial-vlm-eval && bash scripts/cv_bench/set_openrouter_key.sh'
```

## 2. 查看与检查

```bash
bash scripts/cv_bench/run_inference.sh --list
bash scripts/cv_bench/run_inference.sh --check --model PROFILE
bash scripts/cv_bench/run_inference.sh --status
```

## 3. 测试阶段

通用开源模型先启动 endpoint。TP=1 在两个终端各启动一个；TP=2/4 只启动一个：

```bash
# TP=1：另一个终端把 GPU/PORT 改为 1/18102
bash scripts/cv_bench/serve_vllm_profile.sh --model qwen3_vl_8b --gpu-ids 0 --port 18101

# TP=2
bash scripts/cv_bench/serve_vllm_profile.sh --model qwen3_vl_32b --gpu-ids 0,1 --port 18101
```

确认服务就绪后，在新终端运行 test：

```bash
bash scripts/cv_bench/run_inference.sh --stage test --model PROFILE
bash scripts/cv_bench/run_inference.sh --stage test --models PROFILE1,PROFILE2
bash scripts/cv_bench/run_inference.sh --stage test --all
```

test 会依次完成 GPU/processor、视觉 canary、smoke8、输入审计和绑定 gate。未通过 gate 不得跑 full。

## 4. 正式 2638 条

截至 2026-08-04，排除四卡 InternVL3-78B 后的串行批次正在运行。控制器存活时不要重复执行启动命令；
先用 `--status` 和 `$CVBENCH_OUTPUT_ROOT/_serial_full/status.tsv` 查看现场状态。

```bash
bash scripts/cv_bench/run_inference.sh --stage full --model PROFILE
bash scripts/cv_bench/run_inference.sh --stage full --models PROFILE1,PROFILE2
bash scripts/cv_bench/run_inference.sh --stage full --all
```

单模型失败后可在服务恢复就绪时原样重跑同一个 `--model PROFILE`；同签名 journal 会跳过成功项并只补
缺失 index。模型名称和顺序继续通过 `--list` 查询，本页不重复维护逐 profile 命令。

多模型按 registry 顺序串行。需要自动轮换通用模型 vLLM 服务并跳过四卡 InternVL3-78B 时：

```bash
bash scripts/cv_bench/run_full_serial.sh --without-internvl78 --skip-completed
```

该入口按 registry 顺序运行其余 22 条轨、只管理自己启动的 vLLM 服务，任一轨失败立即停止且不评分。
`--skip-completed` 会先用锁定数据重新执行完整 validator，只有实际 prediction 仍通过时才在启动模型前
写入 `SKIP_COMPLETE` 并跳过。运行前须关闭 GPU burn，并在当前 shell 启用 API 代理。

## 5. 只读查看逐条结果

自动跟随当前及后续串行模型，只打印启动监听后新写入 journal 的逐条 success/failure：

```bash
bash scripts/cv_bench/watch_live_predictions.sh
```

双 GPU 独立 lane 运行时分别打开两个只读进度窗口；每个窗口只跟随对应 lane 的当前模型，并在该 lane
换模后自动切换 journal：

```bash
bash scripts/cv_bench/watch_live_predictions.sh --lane gpu0
bash scripts/cv_bench/watch_live_predictions.sh --lane gpu1
```

需要从头重放当前模型的既有 journal 时增加 `--from-start`。按 `Ctrl-C` 只会退出 watcher，不会中断
tmux 中的推理。

## 6. 校验

正式结果必须先通过完整 2638 条校验；`score_results.sh` 也会在评分前强制重复这一步：

```bash
bash scripts/cv_bench/validate_predictions.sh --predictions /absolute/path/to/predictions.jsonl
```

## 7. 评分

只评分一个刚完成的结果：

```bash
bash scripts/cv_bench/score_results.sh --predictions /absolute/path/to/predictions.jsonl
```

查看并评分输出根中的全部待评分结果：

```bash
bash scripts/cv_bench/score_results.sh --list
bash scripts/cv_bench/score_results.sh
```

## 8. 汇总

```bash
bash scripts/cv_bench/build_results_report.sh --check
bash scripts/cv_bench/build_results_report.sh
```

报告写入 `$CVBENCH_OUTPUT_ROOT/cv-bench-result.md`；只有 23/23 条轨通过发布门禁才标记矩阵完整。
