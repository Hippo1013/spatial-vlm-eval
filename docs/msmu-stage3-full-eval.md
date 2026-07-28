# MSMU 阶段三：完整 987 条推理与评分

## 目标

运行 official `test` split 全部 987 条，通过完整 validator 后用独立 local judge 评分。只有本阶段通过
publication gates 的 summary 才能进入结果表。固定写入：

```text
/media/datasets/tangzecong/latent_reasoning/msmu-outputs/manual-three-stage-v1/03_full987/
```

只有阶段二通过的模型才能进入本阶段。脚本会自动清除 `LIMIT`、`INDICES` 和
`MSMU_SMOKE_INDICES`，不会把 subset 参数带入正式运行。

## 第一步：完整推理

LLaVA/InternVL 先在终端 A 启动被测服务：

```bash
bash scripts/msmu/run_manual_stage3.sh MODEL serve
```

然后在终端 B 生成并校验 987 条：

```bash
bash scripts/msmu/run_manual_stage3.sh MODEL infer
```

API、Qwen 和空间专用模型不需要单独的 `serve`，直接执行 `infer`：

```bash
bash scripts/msmu/run_manual_stage3.sh MODEL infer
```

例如：

```bash
bash scripts/msmu/run_manual_stage3.sh qwen25_vl_base infer
```

`infer` 固定使用 `RUN_SCORE=0`，不会提前调用 judge。失败后用完全相同的命令重跑即可 resume。

## 第二步：启动独立 judge

确认完整推理和 validator 已通过，然后停止不再需要的被测模型服务。在一个独立终端启动 judge：

```bash
bash scripts/msmu/run_manual_stage3.sh judge serve
```

它从 `.env.server` 读取 `JUDGE_MODEL`，固定提供
`http://127.0.0.1:18080/v1`，并在启动前运行 GPU preflight。指定其他已协调 GPU：

```bash
MANUAL_JUDGE_CUDA_VISIBLE_DEVICES=1 \
  bash scripts/msmu/run_manual_stage3.sh judge serve
```

被测 vLLM 固定使用 `18081`；judge 与被测服务不能使用同一个 endpoint。

## 第三步：只评分已有结果

judge ready 后，在另一个终端执行：

```bash
bash scripts/msmu/run_manual_stage3.sh MODEL score
```

`score` 会解析同一 `RUN_NAME` 的既有 `predictions.jsonl`，先做正式完整校验，再调用 judge。它不会
重新加载待测模型，也不会再次请求 GPT/Gemini API。

例如：

```bash
bash scripts/msmu/run_manual_stage3.sh qwen25_vl_base score
```

可用模型名：

```bash
bash scripts/msmu/run_manual_stage3.sh --list
```

`internvl3_78b` 会被脚本拒绝。

## 推荐 tmux 名称

session 使用 `msmu-s3`，窗口建议：

```text
MODEL-srv     # 仅 vLLM 被测模型需要
MODEL-full    # 987 条推理
judge-srv     # 本地 judge
MODEL-score   # 正式评分
```

## 正式通过标准

- `predictions.jsonl` 恰好 987 行，index 精确覆盖 `0..986`；
- metadata 中 `num_predictions: 987`、dataset `num_targets: 987`、
  `publishable_inference: true`；
- `prediction_validation.json` 中 `passed: true`、`allow_subset: false`、
  `num_prediction_rows: 987`、`num_unique_indices: 987`；
- `scores/<scorer-protocol>/summary.json` 中 `num_samples: 987`、八类齐全、
  `num_judge_failures: 0`、`publishable: true`。

查看正式产物：

```bash
find /media/datasets/tangzecong/latent_reasoning/msmu-outputs/manual-three-stage-v1/03_full987 \
  -type f \
  \( -name 'prediction_validation.json' -o -name 'predictions.jsonl.metadata.json' \
     -o -name 'summary.json' \) \
  -print | sort
```

需要检查结果时只需提供 SSH 连接方式，不要发送 API key。
