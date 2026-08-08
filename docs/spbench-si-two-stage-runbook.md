# SPBench-SI 两阶段运行手册

本手册只说明操作顺序。输入、scorer 与发布语义以
[canonical protocol](benchmarks/spbench_si/protocol.md)为准。GPU test/full、付费 API 和正式评分是三个
独立授权边界；代码就绪不构成任一授权。

## 1. 只读准备

先合并 [服务器配置模板](../configs/spbench-si-server.env.example)到未跟踪 `.env.server`，复用
`LATENT_PYTHON` 和现有 Conda 环境。安装依赖前先运行 `conda env list`。两个 SPBench 文件继续从
`/media/datasets/tangzecong/` 只读加载；输出、任何新下载、环境、cache、upstream/checkpoint 必须位于
`/media/datasets/lihaoran/`。`SPBENCH_SI_VLLM_RUNTIME_VERSION` 必须写当前审计的精确 0.19.x 版本；
launcher 会对照 `vllm --version`，该值也进入 gate binding。

```bash
set -a
source .env.server
set +a
bash scripts/spbench_si/run_inference.sh --list
bash scripts/spbench_si/run_scheduled_batch.sh --list
bash scripts/spbench_si/run_scheduled_batch.sh --check
bash scripts/spbench_si/run_scheduled_batch.sh --dry-run
```

`--list`/`--check`/`--dry-run` 不加载 dataset/model，不占 GPU，不调用 API，不写正式 prediction。

## 2. 单轨 test 与 full

通用开源轨先在受控 session 启动 registry-driven vLLM；例如 Qwen3-VL-8B：

```bash
bash scripts/spbench_si/serve_vllm_profile.sh \
  --model qwen3_vl_8b --gpu-ids 0 --port 18101
```

对应 `SPBENCH_SI_QWEN3_VL_8B_GPU_IDS=0` 与
`SPBENCH_SI_QWEN3_VL_8B_BASE_URLS=http://127.0.0.1:18101/v1` 必须已加载。processor/template 对照和
read-only GPU preflight 在模型请求前执行；已有端口或 compute process 不会被接管。

获得 GPU 或 API test 授权后先建立 test gate：

```bash
bash scripts/spbench_si/run_inference.sh --stage test --model qwen3_vl_8b
```

检查该轨 `test_gate.json`：红/蓝 canary、smoke8 subset validator、processor/template、单图证据、模型
revision、runtime、GPU/TP、capacity/batch 与全部 binding 必须通过。smoke 分数仅作诊断。

SpatialLadder 例外地使用官方 native batch 16→8→4→2→1 探测。当前 v2 gate 必须显示
`tokenizer_padding_side=left`，并证明同一批中两种不同 prompt 长度的 red/blue canary 都通过。锁定
checkpoint 的 tokenizer 默认是 right padding，因此只看到普通 canary PASS 或容量 16 并不充分；任何
right-padding warning 都是硬故障。当前 inference protocol 是
`spbench_si_spatialladder3b_rgb_rgb_default_direct_folded_user_upstream_locked_v2`；旧
`...upstream_locked_v1` gate/full 已作废，不能 resume 或评分。

获得 full 授权后：

```bash
bash scripts/spbench_si/run_inference.sh --stage full --model qwen3_vl_8b
bash scripts/spbench_si/validate_predictions.sh \
  --predictions /absolute/path/to/predictions.jsonl
```

full 不接受过期 gate；journal 可在完全相同 signature 下恢复。完成时必须有 1,009 行、index
`0..1008`、prediction 每行只有 `index,raw_prediction`，且 `prediction_validation.json` 通过。

## 3. 双卡 20 轨调度

仅 test 的冻结计划：

```bash
bash scripts/spbench_si/run_scheduled_batch.sh --stage test --dry-run
bash scripts/spbench_si/run_scheduled_batch.sh \
  --stage test --without-internvl78 --with-paid-api
```

full 调度会先逐轨建立或复用完全匹配的 test gate，再继续 full；仍需单独获得 full 和付费 API 授权：

```bash
bash scripts/spbench_si/run_scheduled_batch.sh \
  --stage full --without-internvl78 --with-paid-api
```

控制器在 `SPBENCH_SI_OUTPUT_ROOT/_scheduled_batch/` 保存冻结计划、`status.tsv` 和分轨日志。Phase B 只在
双卡 lane 成功、其自有进程退出且端口释放后启动。任一 lane 失败只停止该 lane；不得用人工 `killall`
清理未知进程。

每个 lane 的 watcher 由控制器独立启动。需要人工只读附着时，从 `status.tsv` 取 run id：

```bash
bash scripts/spbench_si/watch_scheduled_health.sh \
  --control-root "$SPBENCH_SI_OUTPUT_ROOT/_scheduled_batch" \
  --run-id RUN_ID --lane gpu0
```

watcher 只读状态并报告 PASS/FAIL/COMPLETE，不重启、不推理、不评分。

## 4. InternVL3-78B 四卡入口

`internvl3_78b` 不在双卡批次中，固定 TP=4、四张 80GB GPU；禁止以量化或 TP=2 替代。四卡环境中
优先使用独立全链路入口；它只管理自有 vLLM，依次执行 test、full-1009、validator、精确双协议评分
和全局报告重建：

```bash
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --check
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --dry-run
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh
bash scripts/spbench_si/run_internvl3_78b_evaluation.sh --status
```

交互式简明问答使用 `--faq`；完整迁移准备、tmux、恢复与产物说明见
[InternVL3-78B 四卡完整评测](spbench-si-internvl3-78b-evaluation.md)。诊断时也可手工启动
registry-driven vLLM，再执行同一 test/full 入口：

```bash
bash scripts/spbench_si/serve_vllm_profile.sh \
  --model internvl3_78b --gpu-ids 0,1,2,3 --port 18102
bash scripts/spbench_si/run_inference.sh --stage test --model internvl3_78b
bash scripts/spbench_si/run_inference.sh --stage full --model internvl3_78b
```

手工模式下服务和推理应在分离的受控 session 中运行；端口、endpoint、TP、GPU 与 revision 必须进入
同一 gate。全链路入口不会停止 burn 或未知任务，资源忙时在模型加载前 fail closed。

若还要同时补齐 Q-Spatial 和 CV-Bench，可改用
[三 Benchmark 一键测评](internvl3-78b-three-bench-evaluation.md)共享一次模型加载。原单 benchmark
入口仍保留；SPBench-SI scheduler、单模型入口和三 benchmark 入口共同竞争输出根锁，不能并发写入。

## 5. 评分与报告

正式评分需再取得授权。先只读检查候选：

```bash
bash scripts/spbench_si/score_results.sh --check
bash scripts/spbench_si/score_results.sh --dry-run
```

获准后可精确评分一条，或默认递归评分全部完整未评分结果：

```bash
bash scripts/spbench_si/score_results.sh --predictions /absolute/path/to/predictions.jsonl
bash scripts/spbench_si/score_results.sh
bash scripts/spbench_si/build_results_report.sh
```

评分器先重跑 full validator，然后分别生成主协议和 upstream audit 目录。报告只发现通过 publication
gates 的唯一候选；20/21 暂行报告只允许明确缺少 `internvl3_78b`，四卡结果补齐后原地重建为 21/21。

## 6. 验收证据

发布前逐轨现场检查：

- `test_gate.json` 与 `test_artifacts/prediction_validation.json`；
- `predictions.jsonl`、`.metadata.json` 与 full `prediction_validation.json`；
- 两个 scorer protocol 各自的 `summary.json`，以及主目录 `publication_gates.json`；
- 全局报告的状态、missing 列表、输入 track 与 21 条唯一候选。

日志完成、tmux 安静或推理结束都不等于评分/发布完成。
