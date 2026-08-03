# MSMU 手工测试入口

三个阶段分别有一个统一脚本。脚本会自动加载 `.env.server`、切到仓库根目录并选择固定输出目录；
不再需要逐行复制各模型的环境变量和 pipeline 命令。

1. [阶段一：接口与图像链路检查](msmu-stage1-canary.md)
2. [阶段二：八类 8 条小量测试](msmu-stage2-smoke8.md)
3. [阶段三：完整 987 条推理与评分](msmu-stage3-full-eval.md)

## 第一次测试前准备

只在服务器仓库第一次使用且 `.env.server` 不存在时运行：

```bash
cd /media/datasets/lihaoran/latent_reasoning/spatial-vlm-eval
test -f .env.server || cp configs/msmu-server.env.example .env.server
```

填好 `.env.server` 后，三个阶段脚本会自行加载它。只有在手工执行底层命令时，才需要单独运行：

```bash
source scripts/msmu/prepare_manual_test.sh
```

## 三个统一入口

```bash
bash scripts/msmu/run_manual_stage1.sh --list
bash scripts/msmu/run_manual_stage2.sh --list
bash scripts/msmu/run_manual_stage3.sh --list
```

查看说明但不启动模型：

```bash
bash scripts/msmu/run_manual_stage1.sh --help
MANUAL_DRY_RUN=1 bash scripts/msmu/run_manual_stage1.sh qwen25_vl_base
MANUAL_DRY_RUN=1 bash scripts/msmu/run_manual_stage1.sh qwen25_vl_32b
MANUAL_DRY_RUN=1 bash scripts/msmu/run_manual_stage1.sh qwen25_vl_72b
MANUAL_DRY_RUN=1 bash scripts/msmu/run_manual_stage1.sh qwen3_vl_2b
MANUAL_DRY_RUN=1 bash scripts/msmu/run_manual_stage1.sh internvl3_78b serve
```

`MANUAL_DRY_RUN=1` 只打印将要执行的底层命令，不占 GPU、不调用 API。

## 固定输出目录

`.env.server` 中的 `MANUAL_TEST_OUTPUT_ROOT` 默认是：

```text
/media/datasets/lihaoran/latent_reasoning/msmu-outputs/manual-three-stage-v1
```

该配置同时把所有新下载与 cache 根设置在 `/media/datasets/lihaoran/`；现有 dataset、model、
interpreter 与 upstream 仍通过精确 legacy 变量读取原位置。目录分工见
[推理手册](msmu-inference.md#1-运行前边界)。

```text
manual-three-stage-v1/
├── 01_canary/
├── 02_smoke8/
└── 03_full987/
```

pipeline 会继续追加模型 revision、inference protocol 和 scorer protocol；三个阶段不共用 journal。
阶段三推理调度状态另存于 `03_full987/_serial_inference/`，评分调度状态另存于
`03_full987/_serial_scoring/<scorer-protocol>/`，都不会混入任何模型的正式结果目录。

## 共同约定

- 一次只选择一个模型；模型通过当前阶段后才进入下一阶段。
- `qwen25_vl_base` 是 7B；另外两个入口是 `qwen25_vl_32b` 和 `qwen25_vl_72b`。32B 默认单卡，
  72B 默认双卡 `0,1` balanced 加载，三者输出目录互不复用。
- 当前补测使用 `qwen3_vl_2b`、`qwen3_vl_4b`、`qwen3_vl_8b`、`qwen3_vl_32b`。四者默认单卡，
  32B 固定 batch size 1；每个参数量使用独立 revision、protocol 和输出目录。
- Qwen stage 1 会先用一张“左上红圆、右下蓝方块”的非 MSMU 组合图检查颜色、形状和位置，再生成
  1 条 MSMU canary；两者都通过才进入 stage 2。
- GPT-5/Gemini stage 1 同样先运行上述组合视觉 canary（每模型 1 次 generation、无 inference retry），
  再生成 2 条 MSMU canary；`vision_canary.json`、provider/model/media audit 和 subset validator 都通过
  才能进入 stage 2。
- API key 只在当前终端导出，不能写入 Git。
- `qwen25_vl_peft` 从 `.env.server` 读取 `QWEN_PEFT_CHECKPOINT`；脚本会把 checkpoint 所在目录和
  basename 加入 run slug，避免与其他 PEFT checkpoint 共用输出。
- 如需自定义唯一输出名，可设置 `MANUAL_RUN_SLUG=name`；只允许字母、数字、点、下划线和连字符。
- `internvl3_78b` 是独立四卡手工补测轨，固定 BF16、TP=4，默认 GPU `0,1,2,3`；stage 1/2/3
  的 serve 都会在加载前确认选中和物理 GPU 均不少于四张，并逐卡执行空闲/显存 preflight。
- 已完成的历史阶段三批次未测试两个 API、Qwen PEFT 和两个 70B+ 模型；`qwen25_vl_72b` 仍由
  stage 3 手工入口拒绝，InternVL3-78B 则只通过独立四卡手工入口补测，不加入历史默认名单。
- 原 13 轨仍是串行脚本默认计划；四款 Qwen3-VL 使用同一脚本的 `--qwen3` 计划。

阶段三固定获准本地轨使用串行入口；实际名单和排除项以 `--list` 及阶段三 runbook 为准：

```bash
bash scripts/msmu/run_stage3_serial_inference.sh --list
MANUAL_DRY_RUN=1 bash scripts/msmu/run_stage3_serial_inference.sh
bash scripts/msmu/run_stage3_serial_inference.sh --check
bash scripts/msmu/run_stage3_serial_inference.sh
bash scripts/msmu/run_stage3_serial_inference.sh --status
```

四款 Qwen3-VL 补测：

```bash
bash scripts/msmu/run_stage3_serial_inference.sh --qwen3 --list
MANUAL_DRY_RUN=1 bash scripts/msmu/run_stage3_serial_inference.sh --qwen3
bash scripts/msmu/run_stage3_serial_inference.sh --qwen3 --check
bash scripts/msmu/run_stage3_serial_inference.sh --qwen3
bash scripts/msmu/run_stage3_serial_inference.sh --qwen3 --status
```

三个阶段共用 tmux session `msmu`。Qwen 窗口建议命名为：

```text
12-qwen-base
21-qwen32b
22-qwen72b
23-qwen3-2b
24-qwen3-4b
25-qwen3-8b
26-qwen3-32b
```
