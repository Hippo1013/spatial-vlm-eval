# MSMU 阶段二：八类 8 条小量测试

## 目标

从八个 official type 中各选择一条，共运行 8 条，验证真实数据端到端流水线。本阶段禁止评分，结果
不可发布。固定写入：

```text
/media/datasets/tangzecong/latent_reasoning/msmu-outputs/manual-three-stage-v1/02_smoke8/
```

只有阶段一通过的模型才能进入本阶段。脚本会自动加载 `.env.server`、重新选择 benchmark-owned 的固定
8 个 index，并写出 `selected_indices.json`；不要手工提供 `INDICES`。

## LLaVA-NeXT 与 InternVL3

如果阶段一服务仍在运行，直接在测试终端执行：

```bash
bash scripts/msmu/run_manual_stage2.sh MODEL
```

如果服务已经停止，使用两个终端：

```bash
# 终端 A：重新启动被测服务
bash scripts/msmu/run_manual_stage2.sh MODEL serve

# 终端 B：运行 8 条
bash scripts/msmu/run_manual_stage2.sh MODEL
```

例如：

```bash
bash scripts/msmu/run_manual_stage2.sh internvl3_8b serve
bash scripts/msmu/run_manual_stage2.sh internvl3_8b
```

`internvl3_78b` 会被脚本拒绝。

## API、Qwen 与空间专用模型

每个模型只需一条命令：

```bash
bash scripts/msmu/run_manual_stage2.sh MODEL
```

例如：

```bash
bash scripts/msmu/run_manual_stage2.sh gpt5
bash scripts/msmu/run_manual_stage2.sh qwen25_vl_base
bash scripts/msmu/run_manual_stage2.sh qwen25_vl_32b
bash scripts/msmu/run_manual_stage2.sh qwen25_vl_72b
bash scripts/msmu/run_manual_stage2.sh ssr_native
bash scripts/msmu/run_manual_stage2.sh spatialbot_native
```

`qwen25_vl_base` 是 7B；32B/72B 的单卡/双卡设置、API backend、key、PEFT checkpoint 和 GPU
覆盖方式与阶段一相同。可用模型名：

```bash
bash scripts/msmu/run_manual_stage2.sh --list
```

## 推荐 tmux 名称

继续使用 session `msmu` 和阶段一同一模型窗口，不另建 stage2 session。阶段二在该模型阶段一通过
并卸载后，直接在同一窗口执行阶段二命令。

## 通过标准

在当前模型的深层目录中检查：

- `predictions.jsonl` 恰好 8 行；
- `prediction_validation.json` 中 `passed: true`、`allow_subset: true`、
  `num_prediction_rows: 8`、`num_unique_indices: 8`；
- metadata 中 `publishable_inference: false`；
- 没有 `summary.json`。

查看本阶段产物：

```bash
find /media/datasets/tangzecong/latent_reasoning/msmu-outputs/manual-three-stage-v1/02_smoke8 \
  -type f \
  \( -name 'prediction_validation.json' -o -name 'predictions.jsonl.metadata.json' \) \
  -print | sort
```

完成后继续：[阶段三：完整 987 条推理与评分](msmu-stage3-full-eval.md)。
