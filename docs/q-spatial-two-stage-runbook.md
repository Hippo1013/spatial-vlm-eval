# Q-Spatial 两阶段推理、评分与报告 runbook

canonical 语义见 [`docs/benchmarks/q_spatial/protocol.md`](benchmarks/q_spatial/protocol.md)，本文件只说明
21 条目标轨的服务器操作顺序。

## 1. 运行前检查

```bash
git status --short
conda env list
set -a
source .env.server
set +a
bash scripts/q_spatial/run_inference.sh --list
```

公共入口按 `QSPATIAL_PYTHON`、`PYTHON`、`LATENT_PYTHON` 的顺序选择解释器；现有服务器配置可直接
复用 `LATENT_PYTHON`，不得因系统 Python 缺包而向默认解释器安装依赖。

确认 Parquet legacy 根只读、ScanNet RGB 访问已获授权、`QSPATIAL_OUTPUT_ROOT` 位于
`/media/datasets/lihaoran/`。脚本不会下载、移动或打包 ScanNet。先运行 `--check`；它会验证完整数据、
processor/template、revision、GPU inventory 与 profile binding，但不会启动服务或付费请求。

## 2. vLLM endpoint

所有 vLLM 轨只使用一个 endpoint。TP=1 把该 endpoint 绑定到一张 GPU；TP=2/4 把一个 endpoint 绑定到
恰好 TP 张 GPU：

```bash
bash scripts/q_spatial/serve_vllm_profile.sh \
  --model qwen3_vl_8b --gpu-ids 0 --port 18101

export QSPATIAL_QWEN3_VL_8B_BASE_URLS=http://127.0.0.1:18101/v1
export QSPATIAL_QWEN3_VL_8B_GPU_IDS=0
```

启动脚本先检查端口、GPU 空闲、显存、utilization 和已有 compute process；失败时保留现有进程。服务
固定 revision、served name、BF16、TP 和每 prompt 一张图。controller 不启动或接管这些 shell 中的
服务，操作员只停止自己启动的进程。

InternVL3-78B 必须显式选择四张 80GB GPU，例如 `0,1,2,3`；两卡会 fail closed。当前硬件不足时只可
在批量命令上显式使用 `--skip-resource-blocked`。

## 3. test stage

```bash
bash scripts/q_spatial/run_inference.sh --stage test --model qwen3_vl_8b
```

成功产物位于该 profile/protocol 的 `test_artifacts/`：dataset manifest、GPU preflight、processor audit、
一个 endpoint 的 red/blue canary、capacity probe、smoke8 journal/prediction/metadata、subset validator、
输入审计和 `test_gate.json`。capacity 从 `32,16,8,4,2,1` 选最高稳定值。smoke numeric parse/准确率只作
诊断，传输、单图、模板或 validator 失败才阻止 full。API capacity 使用独立的 `8,4,2,1` 候选；专用
runner 永远为 1。

## 4. full stage 与恢复

```bash
bash scripts/q_spatial/run_inference.sh --stage full --model qwen3_vl_8b
```

full 先重新计算 binding 并读取 test gate；任何 dataset、prompt、revision、adapter、processor、decoding、
GPU、endpoint 或 sharding 变化都拒绝复用。TP=1 的 `0..270` 在同一个单卡 endpoint 内并发请求，不再
生成或合并奇偶 shard。中断后用完全相同命令恢复；不要编辑 journal。只有 271 条成功后才生成正式
`predictions.jsonl` 和 `publishable_inference=true` metadata。旧双 endpoint gate 会因 binding 不同自动
失效，必须重新 test。

specialized track 默认单 persistent runner。请求 JSONL 只传 index、分离的 system/user prompt、一张
PNG、profile/revision/protocol/decoding；response 必须返回模板/prompt SHA 与一个源 RGB 的证据。普通
RGB 轨还须证明一个 model image tensor/media；`spatialbot_zoedepth` 必须证明一个 RGB tensor 加一个
由同一 RGB 派生的 depth tensor。只有逐请求 seed、metadata 与 processor 审计都证明等价后，未来才能
启用两个 runner。
配置前用对应隔离解释器计算当前 adapter digest，并把相同 generation manifest 同时交给 controller 与
runner，例如：

```bash
python -m spatial_vlm_eval.benchmarks.q_spatial.specialized_runner \
  --profile ssr_rgb --print-adapter-digest
```

## 5. 双卡/API 分阶段批次

先列出冻结计划，再做只读检查和零调用 dry-run：

```bash
bash scripts/q_spatial/run_scheduled_batch.sh --list
bash scripts/q_spatial/run_scheduled_batch.sh --check
bash scripts/q_spatial/run_scheduled_batch.sh --dry-run
bash scripts/q_spatial/run_scheduled_batch.sh --stage test --dry-run
```

计划恰好覆盖除 `internvl3_78b` 外的 20 轨：阶段 A 的 TP=2 双卡 lane 与 API lane 并行；API 内部固定
GPT-5 后 Gemini，绝不重叠。双卡 lane 成功并释放 GPU 后立即启动阶段 B 的 GPU 0/1 两条 lane，不等待
API。GPU 0 为 3DThinker、SpatialRGPT、LLaVA-Mistral、InternVL3-8B、Qwen3-VL 8B/4B/2B；GPU 1 为
SSR native/RGB、HiSpatial、SpatialBot depth/RGB、RoboBrain NV/MT、SpatialLadder。

正式批次必须在 tmux 中运行，并同时显式确认 78B resource block 与付费 API：

```bash
# 只完成或复用 20 轨 test gate，不进入 full
bash scripts/q_spatial/run_scheduled_batch.sh \
  --stage test --without-internvl78 --with-paid-api

# gate 后继续 full-271
bash scripts/q_spatial/run_scheduled_batch.sh \
  --stage full --without-internvl78 --with-paid-api --skip-completed
```

每个 job 先复核现有 test gate；合法则复用，否则重新 test，然后 full 与正式 271 行 validator。严格
`--skip-completed` 还会复核 metadata、model revision、inference/scorer protocol、dataset fingerprint/
files、当前 binding、prediction hash 和 test gate。API、双卡、GPU 0、GPU 1 是独立进程组：API 失败
不影响 GPU；双卡失败只阻止阶段 B；阶段 B 任一 lane 失败不终止另一 lane；最后等待所有已启动 lane
结束并汇总非零状态。控制器只终止自己记录的进程组，端口或 GPU 已占用时 fail closed。
`--stage test` 在 gate 合法后直接记 PASS，不执行 full 或完整 validator；合法 gate 自动复用，因此该模式
拒绝 `--skip-completed`。

冻结计划、状态与日志：

```text
QSPATIAL_OUTPUT_ROOT/_scheduled_batch/
├── plan.json
├── status.tsv
├── active/LANE.json
└── logs/RUN_ID.LANE.PROFILE.*.log
```

每条 lane 启动时会同时启动独立只读 health watcher。watcher 用 `tmux wait-for` 等待状态事件，只输出
`PASS/FAIL/COMPLETE`、异常退出、保守停滞或异常 GPU 空闲；它不读取/打印 prediction 内容，不发信号给
推理进程，不重启、不评分。也可手工附加同一事件流：

```bash
bash scripts/q_spatial/watch_scheduled_health.sh --lane dual
bash scripts/q_spatial/watch_scheduled_health.sh --lane api
bash scripts/q_spatial/watch_scheduled_health.sh --lane gpu0
bash scripts/q_spatial/watch_scheduled_health.sh --lane gpu1
```

## 6. InternVL3-78B 四卡补测

`internvl3_78b` 不并入上述双卡计划，也不允许 TP=2/量化替代。迁移时先同步现有 20 条正式结果，使
`QSPATIAL_OUTPUT_ROOT` 仍指向原 20/21 输出根，再运行独立入口：

```bash
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --check
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh --faq
bash scripts/q_spatial/run_internvl3_78b_evaluation.sh
```

入口固定四张 80GB GPU、BF16/TP=4，并从当前 registry、binding 和 `SCORER_PROTOCOL` 解析最新版
协议。它顺序执行 test/full-271/validator，释放自有 vLLM 后只评分该 canonical prediction，再把原有
`q-spatial-result.md` 原地重建为 21/21。正式 prediction/score 不另开结果根；只有控制日志写入
`_single_model_evaluation/logs/`。完整迁移、burn、恢复和验收见
[InternVL3-78B 四卡补测](q-spatial-internvl3-78b-evaluation.md)。

若还要同时补齐 SPBench-SI 和 CV-Bench，可用
[三 Benchmark 单次 vLLM 入口](internvl3-78b-three-bench-evaluation.md)共享一次模型加载；原单 benchmark
入口仍可独立运行。两种入口使用同一 served name、同一 Q-Spatial canonical 结果路径和互斥锁。

## 7. 评分与报告

```bash
bash scripts/q_spatial/score_results.sh --predictions /absolute/path/predictions.jsonl
bash scripts/q_spatial/score_results.sh --check
bash scripts/q_spatial/score_results.sh
bash scripts/q_spatial/build_results_report.sh --check
bash scripts/q_spatial/build_results_report.sh
```

目录评分不维护模型名单，只发现当前 scorer protocol 的完整、未评分 prediction，并使用 output-root
批次锁。评分后检查 `prediction_validation.json`、271 条 `scored_rows.jsonl`、两个 split、ScanNet 五类、
`summary.json` 与 `publication_gates.json`。报告只收录所有 provenance 与 artifact hash 一致的唯一
候选；未完成 78B 时必须显示缺失，不能用空行冒充结果。

## 8. 当前执行边界

GPU/model/API 实跑必须在已授权服务器资源空闲后进行。API test/full 需要用户提供有效 key 并明确批准
真实付费调用。本地 contract/scorer/mock 测试或服务器只读 preflight 不代表任何 profile 已通过 test、
full、评分或发布门禁；状态只能从当前 output-root 的 gate/validator/metadata/summary 现场确认。
