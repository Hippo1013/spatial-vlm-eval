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

确认 Parquet legacy 根只读、ScanNet RGB 访问已获授权、`QSPATIAL_OUTPUT_ROOT` 位于
`/media/datasets/lihaoran/`。脚本不会下载、移动或打包 ScanNet。先运行 `--check`；它会验证完整数据、
processor/template、revision、GPU inventory 与 profile binding，但不会启动服务或付费请求。

## 2. vLLM endpoint

TP=1 轨需要两张卡上的两个独立 endpoint；TP=2/4 轨只需一个 endpoint：

```bash
bash scripts/q_spatial/serve_vllm_profile.sh \
  --model qwen3_vl_8b --gpu-ids 0 --port 18101

bash scripts/q_spatial/serve_vllm_profile.sh \
  --model qwen3_vl_8b --gpu-ids 1 --port 18102

export QSPATIAL_QWEN3_VL_8B_BASE_URLS=http://127.0.0.1:18101/v1,http://127.0.0.1:18102/v1
export QSPATIAL_QWEN3_VL_8B_GPU_IDS=0,1
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
两个 endpoint 的 red/blue canary、capacity probe、smoke8 journal/prediction/metadata、subset validator、
输入审计和 `test_gate.json`。capacity 从 `32,16,8,4,2,1` 选最高稳定值。smoke numeric parse/准确率只作
诊断，传输、单图、模板或 validator 失败才阻止 full。

## 4. full stage 与恢复

```bash
bash scripts/q_spatial/run_inference.sh --stage full --model qwen3_vl_8b
```

full 先重新计算 binding 并读取 test gate；任何 dataset、prompt、revision、adapter、processor、decoding、
GPU、endpoint 或 sharding 变化都拒绝复用。TP=1 固定偶数 index 到 endpoint 0、奇数到 endpoint 1，最后
原子合并 `0..270`。中断后用完全相同命令恢复；不要编辑 journal。只有 271 条成功后才生成正式
`predictions.jsonl` 和 `publishable_inference=true` metadata。

specialized track 默认单 persistent runner。请求 JSONL 只传 index、分离的 system/user prompt、一张
PNG、profile/revision/protocol/decoding；response 必须返回模板/prompt SHA 与一个 model image tensor 或
media 的证据。只有逐请求 seed、metadata 与 processor 审计都证明等价后，未来才能启用两个 runner。
配置前用对应隔离解释器计算当前 adapter digest，并把相同 generation manifest 同时交给 controller 与
runner，例如：

```bash
python -m spatial_vlm_eval.benchmarks.q_spatial.specialized_runner \
  --profile ssr_rgb --print-adapter-digest
```

## 5. 评分与报告

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

## 6. 当前执行边界

GPU/model/API 实跑必须在已授权服务器资源空闲后进行。API test/full 需要用户提供有效 key 并明确批准
真实付费调用。本地 contract/scorer/mock 测试或服务器只读 preflight 不代表任何 profile 已通过 test、
full、评分或发布门禁；状态只能从当前 output-root 的 gate/validator/metadata/summary 现场确认。
