# MSMU 手工测试入口

三个阶段已经拆成独立文档。请严格按顺序执行，一个模型通过当前阶段后再进入下一阶段：

1. [阶段一：接口与图像链路检查](msmu-stage1-canary.md)
2. [阶段二：八类 8 条小量测试](msmu-stage2-smoke8.md)
3. [阶段三：完整 987 条推理与评分](msmu-stage3-full-eval.md)

## 固定输出目录

三份文档统一使用 `.env.server` 中的 `MANUAL_TEST_OUTPUT_ROOT`。模板默认是：

```text
/media/datasets/tangzecong/latent_reasoning/msmu-outputs/manual-three-stage-v1
```

三个阶段分别写入：

```text
manual-three-stage-v1/
├── 01_canary/   # 阶段一
├── 02_smoke8/   # 阶段二
└── 03_full987/  # 阶段三
```

pipeline 会在阶段目录下继续追加模型 revision、inference protocol 和 scorer protocol。small 与 full
绝不共用 journal。

## 第一次测试前准备

第一次使用时，复制配置模板并在 `.env.server` 中填好实际 dataset、模型 snapshot、上游源码和解释器。
此文件不进入 Git；API key 继续由单独的未跟踪环境变量提供：

```bash
cd /media/datasets/tangzecong/latent_reasoning/spatial-vlm-eval
test -f .env.server || cp configs/msmu-server.env.example .env.server
```

以后每个新终端只执行这一行：

```bash
source /media/datasets/tangzecong/latent_reasoning/spatial-vlm-eval/scripts/msmu/prepare_manual_test.sh
```

脚本会自动加载仓库旁的 `.env.server`、切到仓库根目录、设置固定 `OUTPUT_ROOT`、创建三个阶段目录并
打印 resolved dataset/output。不要用 `bash scripts/...` 执行它，否则导出的变量无法留在当前终端。

## 当前不能做全量的模型

`internvl3_78b` 目前只允许 processor preflight 和 vLLM `DRY_RUN=1`。在获得合适硬件与明确批准前，
不得绕过门禁运行 canary、8 条或 987 条。
