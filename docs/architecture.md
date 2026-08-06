# Repository Architecture

## 分层原则

benchmark 定义合法输入、prediction provenance、validation 和 scoring；model adapter 只定义官方
processor/template、图像张量和 generation。shell 只负责编排，不复制 Python scorer 逻辑。

```text
dataset-owned test rows
  ├─ private provenance: answer/reference, task/source, conversation history
  └─ restricted input(index, exactly one RGB image, benchmark-owned prompt)
       → model-family adapter
       → GenerationResult(prediction + generation metadata)
       → fsync append-only journal
       → benchmark-owned prediction rows
       → atomic predictions.jsonl
       → mandatory validator
       → scorer / judge
```

## 数据所有权边界

`MSMUTestContract` 私有持有 source dataset。adapter 只能收到 frozen `MSMUModelInput`，字段严格是：

```text
index, image, question
```

它没有 reference、raw type、task family、完整 conversations 或其他同图历史。生成完成后，benchmark
层通过 index 从 official row 重新附着前五个 prediction 字段。新增 adapter 不得自行构造六字段
JSONL，也不得持有原始 dataset row。

`CVBenchTestContract` 使用相同三字段可见边界，数据层 prompt 只包含锁定数据集的题目和选项；profile
层再选择 direct-letter 后缀或锁定的 reasoning answer-tag 模板。两种输出指令不得同时出现。
prediction 只保存 `index, raw_prediction`，answer/task/source 仅在 scorer 中按 index 重新关联。MSMU
与 CV-Bench 的 schema、validator 和 scorer protocol 不复用。

`QSpatialTestContract` 使用两个显式数据根，私有持有数值答案、单位、split 与 type。adapter 只收到
`index, image, system_prompt, user_prompt`：一张 RGB、官方 Standard system prompt 和
`Question: {question}`。不支持 system role 的 runner 只允许按锁定分隔符折叠两段 prompt。prediction
同样是 `index, raw_prediction`，但 Q-Spatial validator/scorer 与 CV-Bench 完全独立；`1d_horizontal`
只在 scorer 派生分类中映射为 object width。

每次调用前创建输入审计：index、清洗后题干、RGB 数量、mode、尺寸、像素 SHA-256、profile、
inference protocol 和 chat template。审计禁止保存 base64/API key；它证明“送入哪张图”，但不会把
reference 泄漏给模型。

## 可恢复推理与原子输出

公共 runner 使用与 dataset fingerprint、目标 indices 和完整 adapter metadata 绑定的 run signature。
逐样本 journal 每次 append 后 `fsync`：

- 成功事件保存 prediction、warning 和 generation metadata；
- 失败事件保存脱敏错误，不生成伪空 prediction；
- resume 只跳过同一 signature 下唯一的成功 index；
- 重复成功 index、跨 run journal、越界 index 都是 hard error；
- API/模型调用可以重试，journal 持久化失败不重发付费请求；
- 真实空 completion 是成功事件，并自动带 empty warning。

CV-Bench 本地 vLLM 与付费 API 使用不同恢复策略：vLLM 长尾请求采用较长超时，并在一轮数据结束后
只补 journal 缺失项，避免超时连接背后的服务端生成尚未取消时立即制造重复请求；OpenRouter 保持
429/5xx 逐请求退避和付费调用安全边界。两类策略都写入最终 runtime metadata。

只有全部目标成功后，runner 才按 index 排序并原子替换 `predictions.jsonl`。metadata 同时记录 model、
revision、inference/scorer protocol、图像处理、decoding、upstream commit、runtime package、GPU、开始/
结束时间和 subset 状态。

## Python 包

### `spatial_vlm_eval.benchmarks`

每个 benchmark 子包拥有数据/split 合同、不可泄漏字段、prediction schema、validator、judge prompt、
cache identity、阈值、聚合与 scorer protocol。model 包不得定义指标。

CV-Bench 子包额外拥有 23 条目标轨的 benchmark-specific registry，因为同一模型在不同 benchmark 的
prompt、decoding 和合法 input track 并不相同。通用 journal/resume/原子写入来自 model-neutral
runtime；registry 不进入 scorer 的结果发现逻辑。

Q-Spatial 子包独立拥有两根数据合同、21 条 profile、system/user transport、LLaVA 两阶段格式修复、
numeric parser、split-macro 聚合和发布报告。它只复用 model-neutral runtime 与已锁定 family runner，
不复制 CV-Bench 的题目、parser 或聚合语义。

### `spatial_vlm_eval.models`

- `common/`：journal、resume、审计、原子写入、revision 检查和 CLI 公共参数；
- `profiles.py`：锁定 model/revision/input track/inference protocol；
- family 子包：上游 processor/chat template、图像输入、模型加载与 deterministic generation。

可识别的 Hugging Face `snapshots/<sha>` 路径会与锁定 hash 比较；Git checkout 存在 `.git` 时必须位于
锁定 HEAD。普通本地目录/源码 archive 无法自行证明 revision，metadata 会明确写
`*_revision_verified=false`，不得在报告中描述为已机器验证。

## shell 层

`scripts/cv_bench/` 提供统一两阶段入口。test stage 生成绑定 dataset/model/adapter/processor/decoding/
sharding/GPU selection 的 gate；full stage 不能绕过或复用过期 gate。TP=1 deterministic 通用模型使用
两个已经启动的 endpoint 做固定偶/奇分片，其他轨按 registry 保持单 endpoint 或上游明确支持的并行
方式。GPU preflight 只读 `nvidia-smi` inventory/process，从不终止任何进程。专用模型通过 persistent
JSONL bridge 接入锁定上游 runner；bridge 请求不包含 answer/task/source，runner 必须回传单图 tensor/
media count 和 template digest。

CV-Bench 评分器递归发现完整 prediction，不维护 profile 名单，评分前强制 full validator。报告器只
接受当前 robust scorer protocol 下通过 publication gates 且 registry provenance 完整的 summary；
一条 profile 出现多个发布候选时 fail closed。最终表固定使用模型名称区分 input track，一次只选择一个
scorer protocol，并保留完整 provenance 于 metadata/summary。scorer protocol 升级不要求重跑不变的
inference；兼容规则必须显式列出可消费的历史 inference-metadata scorer ID，并把原声明复制到新
summary，未知 ID 仍 fail closed。解析归一化只作用于 scorer 视图，逐行结果保留未改写的原始回答和
确定性 `parse_evidence`。

InternVL3-78B 的单模型控制器沿用同一边界：固定四卡 TP=4，必要时自动管理自有 vLLM，顺序调用既有
test/full、validator、`--predictions` 精确评分和全局报告入口。它不复制 benchmark/scorer 逻辑，也不
创建模型专属结果根；prediction、score 和 `cv-bench-result.md` 均由原有 canonical 路径与发现逻辑
生成。控制日志仅位于输出根的 `_single_model_evaluation/logs/`。

`scripts/msmu/` 包含 family inference、vLLM server、GPU preflight、validator/scorer 和 pipeline。
GPU preflight 只读取 `nvidia-smi`；显存不足、利用率超限或已有 compute process 时退出，绝不终止
其他进程。只有资源已经协调时才能显式放宽 utilization/process 门禁。

`INFERENCE_BASE_URL` 与 `JUDGE_BASE_URL` 是两个独立变量。`RUN_SCORE=1` 时 pipeline 强制要求
`JUDGE_BASE_URL`，避免把被测模型 endpoint 错当成 judge。`INDICES`/`LIMIT` 自动进入 subset validator，
且 pipeline 硬拒绝 subset scoring。

`scripts/q_spatial/` 提供 registry-driven test/full、严格 validator、目录评分和报告入口。test gate 绑定
两个数据根、Standard Prompt、profile/revision、processor/adapter digest、decoding/seed、GPU、capacity
和 sharding；TP=1 vLLM 用两个 endpoint 固定奇偶分片，其他 backend 单 endpoint/runner。LLaVA 两阶段
调用都传同一张图；specialized JSONL bridge 分别传 system/user prompt 且不含评分字段。评分只发现
271 条完整 prediction，报告按 comparison group 计算加粗且拒绝重复候选。

阶段三串行调度器只编排 inference 与完整 validator，不复制模型或 benchmark 逻辑。它在独立 session/
process group 中启动每个模型，使用 fsync journal 的文件活动做停滞 watchdog，只终止自己记录的
process group，并在进入下一条轨前等待相应 GPU 无 compute process。独占锁、同 commit 完成标记和
活动进程记录防止重复批次、跨代码版本误续跑或意外接管其他服务；judge/scoring 始终留在后续独立阶段。

阶段三评分调度器递归发现结果根中的 `predictions.jsonl`，从 scorer 模块读取当前 protocol，并按完整
路径稳定排序；脚本中不维护模型名单。只有 prediction 的直接父目录等于当前 scorer protocol 才能进入
评分。整轨状态为：

- `new`：尚无评分产物；
- `resume`：已有可复用的部分 judge cache，但没有成功 summary；
- `retry`：已有损坏、失败或不完整的 canonical 评分产物；
- `complete`：summary、完整 score validator、987 条 scored rows、空 judge failure 和全部
  publication gates 均合法；
- `excluded_protocol`：prediction 不属于当前 scorer protocol。

执行模式在结果根下持有进程级独占锁，锁内冻结候选快照，再逐个调用
`scripts/msmu/score_predictions.sh`。批次开始前和每轨开始前均核对 `/v1/models` 中的
`JUDGE_MODEL_NAME`；单轨非零退出或评分后仍非 `complete` 时立即停止。scorer 自身继续负责强制完整
validator、judge cache key、失败重试、阈值、逐样本得分与 macro-8 聚合，调度器不复制这些逻辑。
`Ctrl-C` 只中断当前评分批次，不管理另一个终端中的 judge。

批次控制文件为：

```text
03_full987/_serial_scoring/SCORER_PROTOCOL/
├── lock
├── status.tsv
└── runs/
    ├── UTC_TIMESTAMP.candidates.jsonl
    └── UTC_TIMESTAMP.log
```

候选在启动后冻结；运行期间新出现的 prediction 留到下一批。canonical `summary.json` 是唯一完成
依据，不创建额外完成标记，也不在日志中记录 API key。

单模型一键入口 `run_model_evaluation.sh` 仍保持上述阶段边界，只在一个受控进程中顺序编排它们：
从共享 manual-stage 注册表只读解析模型类型与输出路径，必要时启动并停止被测 vLLM，运行 full-987
与正式 validator，释放模型 GPU 后启动独立 judge，再用 `score_pending_results.sh --predictions` 精确
冻结本次一个结果，最后重建全局报告。一键入口不维护第二份模型名单，不复制 adapter/scorer/report
语义，也不把 inference 和 scorer protocol 合并。信号处理只终止入口自己创建的进程组；已有端口、
GPU 进程或批次锁均 fail closed 并保持原状。每次控制器运行日志写到
`03_full987/_single_model_evaluation/logs/UTC_TIMESTAMP-MODEL.log`；模型、validator、score 和报告产物
仍使用下述 canonical 目录，不复制到控制器状态目录。

结果报告生成器递归发现 `scores/<scorer-protocol>/summary.json`，不限定当前 scorer protocol，也不
维护模型名单。每条候选必须同时通过对应 protocol 的 canonical 完成检查、完整 publication gates、
推理 metadata 与目录 protocol 一致性，以及八类 accuracy 对 macro-8 的复算；不完整 summary 只在
`--list` 中报告，不能静默进入表格。报告可按 metadata profile 精确筛选；精简表不显示 protocol
列，因此一次只允许一个 scorer protocol，未指定时固定使用当前 canonical protocol，拒绝把历史和
当前评分混入同一张表。诊断 summary 会在终端告警并跳过，筛选后没有合法评分时 fail closed。
最终 Markdown 固定为 `# MSMU-Bench评测结果`、一行输入/提示配置说明和一张中文表；列为模型名称、
官方论文顺序的八项指标和平均值，每列（含平均）所有并列最高分加粗。未显式指定 profile 时，模型
默认按 API、通用开源、空间专项分组；同系列按参数量升序，专项同模型按纯 RGB 到额外先验排序。
显式 `--profile` 仍严格保留调用方给定顺序。当前专用模型使用固定的 profile 级展示规则：SSR 的两轨标为
`RGB` / `RGB + 深度估计`，SpatialRGPT 保持模型原名且不加注释，3DThinker 的两轨标为 `RGB` /
`RGB + Mental-3D 提示词`，SpatialBot 的两轨标为 `RGB` / `RGB + 深度估计`。未知双轨 profile 若没有
显式展示配置必须 fail closed，不能退回含混的“公平版/原生版”。完整 revision、inference protocol、
scorer protocol 与 result kind 仍以已经校验的 metadata/summary 和结果目录为 canonical provenance，
不复制到展示表。

正式评分前可用 `build_stage3_answer_audit.py` 只读加载获准批次的完整 validator 和 prediction，
对所有轨抽取同一组 index 并导出人工抽查文档。它不参与评分，产物写入仓库外
`MANUAL_TEST_OUTPUT_ROOT/_answer_audit/`。仓库根不得创建 `output/` 或 `outputs/`。

## 输出布局

CV-Bench 使用独立仓库外根：

```text
CVBENCH_OUTPUT_ROOT/
├── _single_model_evaluation/logs/        # 编排日志，不是正式结果
├── runs/PROFILE/MODEL_REVISION/INFERENCE_PROTOCOL/
│   ├── test_gate.json
│   ├── predictions.jsonl
│   ├── predictions.jsonl.journal.jsonl
│   ├── predictions.jsonl.metadata.json
│   ├── prediction_validation.json
│   └── scores/SCORER_PROTOCOL/
│       ├── scored_rows.jsonl
│       ├── summary.json
│       └── publication_gates.json
└── cv-bench-result.md
```

Q-Spatial 使用平行但独立的仓库外根：

```text
QSPATIAL_OUTPUT_ROOT/
├── runs/PROFILE/MODEL_REVISION/INFERENCE_PROTOCOL/
│   ├── test_artifacts/
│   │   ├── dataset_manifest.json
│   │   ├── vision_canary.json
│   │   ├── capacity_probe.json
│   │   └── test_gate.json
│   ├── predictions.jsonl
│   ├── predictions.jsonl.journal.jsonl
│   ├── predictions.jsonl.metadata.json
│   └── scores/SCORER_PROTOCOL/
│       ├── prediction_validation.json
│       ├── scored_rows.jsonl
│       ├── summary.json
│       └── publication_gates.json
└── q-spatial-result.md
```

以下是 MSMU 的既有布局：

未显式设置 `OUTPUT` 时，公共路径函数在 `OUTPUT_ROOT` 下生成：

```text
OUTPUT_ROOT/
└── RUN_NAME/
    └── MODEL_REVISION_TAG/
        └── INFERENCE_PROTOCOL/
            └── SCORER_PROTOCOL/
                ├── predictions.jsonl
                ├── predictions.jsonl.journal.jsonl
                ├── predictions.jsonl.metadata.json
                ├── prediction_validation.json
                ├── predictions.infer.log
                └── scores/SCORER_PROTOCOL/
```

不同模型 revision、decoding/input profile、inference protocol 或 scorer protocol 不共享目录。生成物
不进入 Git，可另行归档到对象存储或实验平台。
