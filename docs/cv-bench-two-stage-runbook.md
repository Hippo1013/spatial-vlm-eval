# CV-Bench 两阶段推理、评分与汇总

本手册只保留操作者需要的命令。协议与产物定义见
[CV-Bench canonical protocol](benchmarks/cv_bench/protocol.md)。本轮只实现和验证链路，不自动启动
2638 条正式推理。只需复制命令时直接看 [CV-Bench 简明运行指令](cv-bench-commands.md)。

## 1. 环境与只读检查

先检查已有环境，不向默认 Python 安装依赖：

```bash
conda env list
test -e .env.cvbench.server || cp configs/cv-bench-server.env.example .env.cvbench.server
export CVBENCH_ENV_FILE="$PWD/.env.cvbench.server"
```

模板中的 `CVBENCH_DATASET_ROOT` 已指向 2026-08-03 现场核验的 legacy 只读目录
`/media/datasets/tangzecong/huggingface/datasets/CV-Bench`；运行前仍须用 `--check` 复核两个文件。新环境、upstream、
checkpoint、cache 和输出只能写入 `/media/datasets/lihaoran/`。然后列出目标轨：

```bash
bash scripts/cv_bench/run_inference.sh --list
bash scripts/cv_bench/run_inference.sh --stage test --model qwen3_vl_8b --dry-run
```

vLLM 模型需先在已协调 GPU 上启动 OpenAI-compatible endpoint；TP=1 轨配置两个 endpoint，TP=2/4 轨
配置一个 endpoint。统一启动器会从 registry 读取 model revision、served name、TP 和显存门槛，并在
加载权重前拒绝忙碌 GPU：

```bash
# TP=1：分别在 GPU 0/1 启动 18101/18102
bash scripts/cv_bench/serve_vllm_profile.sh --model qwen3_vl_8b --gpu-ids 0 --port 18101
bash scripts/cv_bench/serve_vllm_profile.sh --model qwen3_vl_8b --gpu-ids 1 --port 18102

# TP=2：一个 endpoint
bash scripts/cv_bench/serve_vllm_profile.sh --model qwen3_vl_32b --gpu-ids 0,1 --port 18101
```

`--check` 会加载锁定数据、执行官方 processor/template 审计并核对全部绑定字段：

```bash
bash scripts/cv_bench/run_inference.sh --check --model qwen3_vl_8b
```

InternVL3-78B 必须显式设置：

```bash
export CVBENCH_INTERNVL3_78B_GPU_IDS=0,1,2,3
```

脚本只读取 `nvidia-smi` inventory 与 compute process，不会停止任何现有进程。GPU 已占用或 endpoint
不属于本任务时，操作者先协调资源，不得接管或终止别人的进程。

## 2. 专用模型 runner

12 条专用轨通过仓库的 dataset-blind persistent JSONL runner 接入，并使用各 family 的锁定解释器。
先用同一解释器计算组合实现 SHA-256，再为每条轨设置 command；SSR、3DThinker direct、
SpatialLadder 还必须指向 `configs/cv-bench-generation/` 中从锁定上游/checkpoint 解析的 manifest：

```bash
export RUNNER_PYTHON=/absolute/path/to/family/python
export CVBENCH_SPATIALRGPT_RGB_COMMAND="$RUNNER_PYTHON -u -m spatial_vlm_eval.benchmarks.cv_bench.specialized_runner --profile spatialrgpt_rgb"
export CVBENCH_SPATIALRGPT_RGB_ADAPTER_DIGEST="$($RUNNER_PYTHON -m spatial_vlm_eval.benchmarks.cv_bench.specialized_runner --profile spatialrgpt_rgb --print-adapter-digest)"
```

runner 每行只接收 index、最终 prompt、一个 PNG data URI 和锁定 profile metadata；响应必须回传同一
profile/revision/protocol/decoding、原始输出、模板 SHA-256，并证明一个 media 或 image tensor。缺少
任一证明时测试 gate 失败。不得通过 runner 读取原始 Parquet 或答案/任务/来源字段。

HiSpatial 额外锁定 `Ruicheng/moge-2-vitl-normal@b135031bae30b5ac2ae141a0e68717795ce38340`
和 MoGe 上游 `925b8ed835a7a9cdb7578ba15c658a0afc969030`；runner 会同时验证 HiSpatial、
MoGe-2 checkpoint 和两个上游 checkout，任一 revision 不符即停止。

## 3. 测试阶段

先按 backend/family 各选一轨；这些通过后再逐轨建立 gate：

```bash
bash scripts/cv_bench/run_inference.sh --stage test --model llava_next_mistral_7b
bash scripts/cv_bench/run_inference.sh --stage test --model qwen3_vl_2b
bash scripts/cv_bench/run_inference.sh --stage test --model gpt5_openrouter_non_zdr
bash scripts/cv_bench/run_inference.sh --stage test --model spatialrgpt_rgb
```

也可按 registry 顺序串行选择多轨或全部：

```bash
bash scripts/cv_bench/run_inference.sh --stage test --models qwen3_vl_2b,qwen3_vl_4b
bash scripts/cv_bench/run_inference.sh --stage test --all
```

状态检查：

```bash
bash scripts/cv_bench/run_inference.sh --status
```

确认每轨 `test_gate.json`、dataset/processor/input audit、vision canary、capacity probe 和 smoke8 subset
validator 全部通过。API test 会产生真实付费调用；journal 只跳过已经成功的同签名 index。completion
已经成功但 provider metadata/契约验证失败时不会重发付费 POST，必须先人工核对再决定后续处理。

## 4. 正式全量

只有当前绑定的 test gate 可以解锁 full：

```bash
bash scripts/cv_bench/run_inference.sh --stage full --model qwen3_vl_8b
bash scripts/cv_bench/run_inference.sh --stage full --models qwen3_vl_2b,qwen3_vl_4b
bash scripts/cv_bench/run_inference.sh --stage full --all
```

多模型严格按 registry 顺序串行。单卡可容纳的 deterministic 通用模型以两个 endpoint 固定偶/奇 index
分片并确定性合并；TP=2/4、API 和不支持并行的专用 sampling 轨保持单 endpoint/worker 策略。正式文件
必须精确覆盖 2638 条并通过 `prediction_validation.json`；subset 不得复制到正式目录。

## 5. 评分与报告

只评分指定 prediction：

```bash
bash scripts/cv_bench/score_results.sh --predictions /absolute/path/to/predictions.jsonl
```

或递归发现全部完整、未评分结果：

```bash
bash scripts/cv_bench/score_results.sh --list
bash scripts/cv_bench/score_results.sh
```

评分完成后检查 `summary.json` 的 2638 条、三个来源、四个任务、2D/3D/Overall 和全部 publication
gates，再生成全局表：

```bash
bash scripts/cv_bench/build_results_report.sh --check
bash scripts/cv_bench/build_results_report.sh
```

报告固定写到 `$CVBENCH_OUTPUT_ROOT/cv-bench-result.md`，并明确列出缺失 profile；只有 `23/23` 时
才宣称目标矩阵完整。
