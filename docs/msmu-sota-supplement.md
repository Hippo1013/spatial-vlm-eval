# MSMU SOTA 双 Lane 补测 Runbook

本入口只服务 MSMU official `test` 987 条，语义以
[MSMU canonical protocol](benchmarks/msmu/protocol.md)为唯一事实源。RoboBrain、HiSpatial 与
SpatialLadder 的其他 benchmark runner 只能作为官方 processor/模型加载技术证据，不能向本入口提供
题型、答案格式、reference、validator、scorer 或聚合语义。

## 冻结范围

控制器固定运行五条互不兼容的 inference profile：

| Lane | GPU | 串行顺序 |
|---|---|---|
| `gpu0` | `SOTA_SUPPLEMENT_GPU0`，默认 0 | RoboBrain2.5 NV → HiSpatial + same-RGB MoGe-2 XYZ → SpatialLadder direct |
| `gpu1` | `SOTA_SUPPLEMENT_GPU1`，默认 1 | RoboBrain2.5 MT → SpatialLadder generic thinking |

两个 SpatialLadder 轨故意分卡。lane 内严格串行，lane 间并发；canary 和 smoke8 都不评分。只有两条
lane 均 COMPLETE，五份 full prediction 精确覆盖 `0..986` 且 validator/provenance 通过后，才释放推理
模型并启动一次 GPU0 local judge。评分顺序固定为 NV、MT、HiSpatial、direct、thinking。

四条 direct/main 轨只有在全部现场门禁完成后才能加入 MSMU 主矩阵；thinking 永久只作补充行。

## 环境与只读预检

先检查已有 Conda 环境和资产；默认/系统 Python 缺包不是安装理由：

```bash
conda env list
cd /media/datasets/lihaoran/latent_reasoning/spatial-vlm-eval
bash scripts/msmu/run_sota_supplement.sh --list
MANUAL_DRY_RUN=1 bash scripts/msmu/run_sota_supplement.sh
bash scripts/msmu/run_sota_supplement.sh --check
```

`--check` 验证：

- official 987-row dataset fingerprint；
- 两张不同、空闲、至少 80GB 的选中 GPU；只检查选中卡，不接管或停止任何现有进程；
- 五个精确 HF snapshot revision；
- RoboBrain、HiSpatial、SpatialLadder、MoGe-2 与 utils3d 的锁定 checkout；
- 三个 family-specific interpreter 的模型依赖和 MSMU `datasets` loader import；
- supplement lock、既有 MSMU 全量推理锁与 judge `127.0.0.1:18080`；
- 已存在 frozen plan 是否与当前代码、dataset、profile 和 adapter digest 完全一致。

需要在 `.env.server` 或当前 shell 提供：

```text
ROBOBRAIN25_PYTHON
HISPATIAL_PYTHON
SPATIALLADDER_PYTHON
ROBOBRAIN25_UPSTREAM_ROOT
HISPATIAL_UPSTREAM_ROOT
SPATIALLADDER_UPSTREAM_ROOT
ROBOBRAIN25_8B_NV_MODEL
ROBOBRAIN25_8B_MT_MODEL
HISPATIAL_3B_MODEL
SPATIALLADDER_3B_MODEL
MOGE2_MODEL
MOGE2_UPSTREAM_ROOT
MOGE2_UTILS3D_ROOT
```

路径模板见 `configs/msmu-server.env.example`。新下载、checkout、env 和 cache 只能写入
`/media/datasets/lihaoran/`；优先复用已有 HiSpatial/SpatialLadder 环境。只有 import probe 失败且确认
没有合适环境时，才创建 family-specific 环境。当前 shell 显式提供的上述 supplement 解释器与资产
路径优先于通用 `.env.server`，并由 stage wrapper 原样保留。`/media/datasets/tangzecong/` 只读。

## 启动与观察

获得 GPU 推理和正式评分授权后执行：

```bash
bash scripts/msmu/run_sota_supplement.sh
bash scripts/msmu/run_sota_supplement.sh --status
```

正式控制产物只写到仓库外：

```text
$MANUAL_TEST_OUTPUT_ROOT/
├── 01_canary/<run>/<revision>/<inference>/<scorer>/
├── 02_smoke8/<run>/<revision>/<inference>/<scorer>/
└── 03_full987/
    ├── <run>/<revision>/<inference>/<scorer>/
    ├── msmu-result.md
    └── _sota_supplement/
        ├── frozen-plan.json
        ├── status.tsv
        └── logs/
```

每条 lane 只有一个纯事件驱动 watcher。控制器写状态后通过继承 pipe 唤醒 watcher；watcher只读取
`status.tsv` 并输出 PASS、FAIL、COMPLETE 或真实 FAULT，不轮询、不加载模型、不消耗 token。运行状态
以 `status.tsv`、validator、metadata 和 summary 为准，不以 tmux 是否安静判断。

## Artifact 与恢复门禁

每个 canary/smoke/full 目录独立绑定 model revision、inference protocol、scorer protocol、dataset
fingerprint 和 adapter digest。合法 finalized 产物直接复用；只有 journal 未完成或 prediction 尚未原子
生成时才续跑。若 `predictions.jsonl` 已存在但 metadata、validator、canary、indices 或 provenance 任一
不合法，控制器报告精确路径并停止，绝不覆盖。

任一 lane 失败时，控制器只向自己创建的进程组发送 TERM/KILL。另一 lane 已完成的 journal、prediction
和 validator 保留；同一命令重跑会从合法断点恢复。未知 GPU 进程、占用端口、不同 frozen plan 或锁
冲突都 fail closed。

## 统一评分与报告

推理双 lane 完成前绝不启动 judge。若某些 profile 已有当前 scorer protocol 的完整 publication-gated
summary，只评分缺失项；如果五条都已评分则不启动 judge。评分仍走目录驱动入口
`score_pending_results.sh --predictions <exact-path>`，不维护第二份模型名单，也不绕过完整 validator、
judge readiness 或 publication gates。评分故障保留 judge cache，不写报告。

五条评分均完成后，控制器先运行：

```bash
bash scripts/msmu/build_results_report.sh --check
```

该只读门禁要求既有 18 条 baseline 和新增 5 条在同一 canonical scorer protocol 下各有且只有一个
完整 summary。通过后才原子重建现有 `03_full987/msmu-result.md`，固定 23 行，并继续标为
`official-compatible internal score`。

## 现场验收与范围晋级

完成后逐条检查：

1. full prediction 为 987 行且 index 精确为 `0..986`；
2. `prediction_validation.json` passed，metadata 的 model/revision/inference/scorer/dataset/adapter 一致；
3. summary 的 8 类齐全、`num_samples=987`、judge failures 为 0、publication gates 全真；
4. thinking journal 的 raw-response hash/长度/标签抽取状态完整，fallback 有 warning；
5. `build_results_report.sh --check` 通过，报告恰有 23 个模型行；
6. 五条显示名分别标明 `RGB`、`RGB + MoGe-2 XYZ`、`RGB / direct` 和官方通用 thinking 提示词。

只有上述现场证据全部满足，才把四条 main profile 加入 `CURRENT_TARGET_PROFILE_KEYS`，将 MSMU 主矩阵
由 18 晋级为 22；thinking 不加入。晋级时同步更新模型矩阵、评测范围、README、CHANGELOG 和文档测试。
