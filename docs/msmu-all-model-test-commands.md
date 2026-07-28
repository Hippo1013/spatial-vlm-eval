# MSMU 手工测试入口

三个阶段分别有一个统一脚本。脚本会自动加载 `.env.server`、切到仓库根目录并选择固定输出目录；
不再需要逐行复制各模型的环境变量和 pipeline 命令。

1. [阶段一：接口与图像链路检查](msmu-stage1-canary.md)
2. [阶段二：八类 8 条小量测试](msmu-stage2-smoke8.md)
3. [阶段三：完整 987 条推理与评分](msmu-stage3-full-eval.md)

## 第一次测试前准备

只在服务器仓库第一次使用且 `.env.server` 不存在时运行：

```bash
cd /media/datasets/tangzecong/latent_reasoning/spatial-vlm-eval
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
```

`MANUAL_DRY_RUN=1` 只打印将要执行的底层命令，不占 GPU、不调用 API。

## 固定输出目录

`.env.server` 中的 `MANUAL_TEST_OUTPUT_ROOT` 默认是：

```text
/media/datasets/tangzecong/latent_reasoning/msmu-outputs/manual-three-stage-v1
```

```text
manual-three-stage-v1/
├── 01_canary/
├── 02_smoke8/
└── 03_full987/
```

pipeline 会继续追加模型 revision、inference protocol 和 scorer protocol；三个阶段不共用 journal。

## 共同约定

- 一次只选择一个模型；模型通过当前阶段后才进入下一阶段。
- API key 只在当前终端导出，不能写入 Git。
- `qwen25_vl_peft` 从 `.env.server` 读取 `QWEN_PEFT_CHECKPOINT`；脚本会把 checkpoint 所在目录和
  basename 加入 run slug，避免与其他 PEFT checkpoint 共用输出。
- 如需自定义唯一输出名，可设置 `MANUAL_RUN_SLUG=name`；只允许字母、数字、点、下划线和连字符。
- `internvl3_78b` 只允许阶段一静态检查，阶段二和阶段三由脚本强制拒绝。
