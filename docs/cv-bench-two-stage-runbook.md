# CV-Bench 两阶段推理、评分与汇总

本手册只保留操作者需要的命令。协议与产物定义见
[CV-Bench canonical protocol](benchmarks/cv_bench/protocol.md)。截至 2026-08-06，排除四卡
InternVL3-78B 后的 22 条目标轨均已通过 full-2638 validator、当前 scorer 评分和 publication gates；
全局报告为 22/23。只需复制命令时直接看 [CV-Bench 简明运行指令](cv-bench-commands.md)，实时状态以
服务器 validator、metadata、summary 和 publication gates 为准。

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

OpenRouter API key 通过交互式隐藏输入写入未跟踪的共享环境文件；脚本原子替换旧值、设置 mode 600，
不会把 key 放入 shell history、命令参数或日志：

```bash
ssh -t msmu-a800 'cd /media/datasets/lihaoran/latent_reasoning/spatial-vlm-eval && bash scripts/cv_bench/set_openrouter_key.sh'
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

最终 prompt 在 profile 层生成：普通轨追加 direct-letter 后缀；`3dthinker_mental3d` 和
`spatialladder3b_thinking` 只使用各自官方 `<think>/<answer>` 模板。两条 reasoning 轨的 v1 gate
包含冲突指令，不能迁移，部署当前代码后必须重新运行 test stage；当前服务器已于 2026-08-04 完成
两条 v2 gate。其他轨的最终 prompt 不变，若仅 adapter digest 变化可走现有无模型调用的审计迁移。

HiSpatial 额外锁定 `Ruicheng/moge-2-vitl-normal@b135031bae30b5ac2ae141a0e68717795ce38340`
和 MoGe 上游 `925b8ed835a7a9cdb7578ba15c658a0afc969030`；runner 会同时验证 HiSpatial、
MoGe-2 checkpoint、两个模型上游 checkout，以及 MoGe requirements 锁定的
`EasternJournalist/utils3d@3fab839f0be9931dac7c8488eb0e1600c236e183`，任一 revision 或已安装
`utils3d` 内容不符即停止。`MOGE2_UTILS3D_ROOT` 必须指向该 checkout。
MoGe-2 的锁定 snapshot 内必须存在 `model.pt`；HiSpatial runner 将该文件而不是 snapshot 目录传给
上游 `MoGeModel.from_pretrained`，避免目录被误当作 torch checkpoint。

SpatialLadder checkpoint 在嵌套 `text_config` 中声明 tied output embeddings，但外层兼容字段为
false；runner 会把该锁定声明传播到模型外层 config，并在加载后使用同一共享输出权重，避免随机初始化
缺失 `lm_head`。runner 锁定 PyTorch SDPA；嵌套声明缺失或不是 true 时 fail closed。

SpatialBot 还必须设置 `SPATIALBOT_SIGLIP_MODEL`，指向
`google/siglip-so400m-patch14-384@9fdffc58afc957d1a03a25b10dba0329ab15c2a3` 的本地快照。
runner 会验证 revision，并仅在内存中把 checkpoint config 的同名 vision tower 改绑到该只读路径；
不会修改 checkpoint，也不会在离线推理时隐式下载。
ZoeDepth 轨还必须把 `SPATIALBOT_MIDAS_ROOT` 指向
`isl-org/MiDaS@454597711a62eabcbf7d1e89f3fb9f569051ac9b` 的本地 checkout；runner 验证 commit 后，
只把上游精确的 `intel-isl/MiDaS` torch.hub 请求改绑到该 checkout，禁止网络回退。

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
全部轨统一使用纯红、纯蓝两张 RGB 图的颜色识别最低门禁，不考察形状、方位或空间描述能力；两次
都必须证明模型边界恰好接收一张图。已通过旧版红圆/蓝方块严格 canary 的轨可经 artifact 审计迁移
到当前 gate，无需重新加载模型；迁移产物必须保留旧协议、答案和源 gate 路径。
当前 protocol gate 若仅组合 adapter source digest 改变、其他 binding 逐字段完全相同，也可自动生成
`adapter_digest_only` 迁移记录；任何模型 revision、decoding、input track、backend 或 sharding 变化仍须重测。

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

本地 vLLM 默认允许单请求最多 600 秒，并在首轮结束后只补一次 journal 缺失 index；不要把
`CVBENCH_INFERENCE_RETRIES` 用作本地 vLLM 的即时重试。确需现场覆盖时分别使用
`CVBENCH_VLLM_API_TIMEOUT`、`CVBENCH_VLLM_INFERENCE_RETRIES` 和
`CVBENCH_VLLM_RETRY_MISSING_PASSES`。重新执行同一 profile 的 full 会复用同签名 journal 中已经成功的
index，只补缺失项。

自动轮换每条通用模型的 vLLM 服务并明确跳过当前无法运行的 InternVL3-78B：

```bash
bash scripts/cv_bench/run_full_serial.sh --without-internvl78 --skip-completed
```

控制器只管理自己启动的服务，遇到占用端口、忙碌 GPU、过期 gate 或任一轨失败即停止；API 代理和
GPU burn 仍由操作者按对应手册在控制器外显式开关。`--skip-completed` 只在现有 prediction 重新通过
锁定数据的完整 validator 后跳过对应轨，并在 `status.tsv` 记录 `SKIP_COMPLETE`。控制器不评分。

InternVL3-78B 不进入上述两卡串行批次。四张 80GB GPU 可用后，用一个独立入口串起该轨的当前 test
gate、full-2638、独立 validator、精确单轨评分和原有全局报告：

```bash
bash scripts/cv_bench/run_internvl3_78b_evaluation.sh --check
bash scripts/cv_bench/run_internvl3_78b_evaluation.sh
```

入口固定 profile `internvl3_78b`、BF16、TP=4 和 registry decoding；只拥有自己启动的 vLLM 进程组，
在评分前停止服务并确认四卡释放。prediction、score 和报告继续使用原有 canonical 路径；控制日志单独
保存在 `$CVBENCH_OUTPUT_ROOT/_single_model_evaluation/logs/`。完整操作说明见
[InternVL3-78B 一键完整评测](cv-bench-internvl3-78b-evaluation.md)。

另开终端可用只读 watcher 自动跟随当前及后续模型的正式 journal，并逐条打印精简后的 prediction；
默认只显示启动监听后追加的事件，`--from-start` 可重放当前模型已有事件：

```bash
bash scripts/cv_bench/watch_live_predictions.sh
```

双 GPU 独立 lane 的状态分别位于 `_dual_lane/gpu0/status.tsv` 与 `_dual_lane/gpu1/status.tsv` 时，使用
两个终端分别跟随；脚本识别各 lane 的 `PASS`、`FAIL`、`BLOCKED`、`COMPLETE`，只读取对应当前
profile 的正式 journal：

```bash
bash scripts/cv_bench/watch_live_predictions.sh --lane gpu0
bash scripts/cv_bench/watch_live_predictions.sh --lane gpu1
```

按 `Ctrl-C` 仅停止 watcher，不会向控制器、模型服务或推理进程发送信号。

## 5. 校验、评分与报告

正式推理后可先独立执行完整 validator；评分入口还会强制重复校验，不能绕过：

```bash
bash scripts/cv_bench/validate_predictions.sh --predictions /absolute/path/to/predictions.jsonl
```

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
