# Repository Architecture

## 分层原则

benchmark 定义合法输入、prediction provenance、validation 和 scoring；model adapter 只定义官方
processor/template、图像张量和 generation。shell 只负责编排，不复制 Python scorer 逻辑。

```text
dataset-owned test rows
  ├─ private provenance: raw type, reference, conversation history
  └─ restricted MSMUModelInput(index, RGB image, clean first question)
       → model-family adapter
       → GenerationResult(prediction + generation metadata)
       → fsync append-only journal
       → benchmark-owned six-field rows
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

只有全部目标成功后，runner 才按 index 排序并原子替换 `predictions.jsonl`。metadata 同时记录 model、
revision、inference/scorer protocol、图像处理、decoding、upstream commit、runtime package、GPU、开始/
结束时间和 subset 状态。

## Python 包

### `spatial_vlm_eval.benchmarks`

每个 benchmark 子包拥有数据/split 合同、不可泄漏字段、prediction schema、validator、judge prompt、
cache identity、阈值、聚合与 scorer protocol。model 包不得定义指标。

### `spatial_vlm_eval.models`

- `common/`：journal、resume、审计、原子写入、revision 检查和 CLI 公共参数；
- `profiles.py`：锁定 model/revision/input track/inference protocol；
- family 子包：上游 processor/chat template、图像输入、模型加载与 deterministic generation。

可识别的 Hugging Face `snapshots/<sha>` 路径会与锁定 hash 比较；Git checkout 存在 `.git` 时必须位于
锁定 HEAD。普通本地目录/源码 archive 无法自行证明 revision，metadata 会明确写
`*_revision_verified=false`，不得在报告中描述为已机器验证。

## shell 层

`scripts/msmu/` 包含 family inference、vLLM server、GPU preflight、validator/scorer 和 pipeline。
GPU preflight 只读取 `nvidia-smi`；显存不足、利用率超限或已有 compute process 时退出，绝不终止
其他进程。只有资源已经协调时才能显式放宽 utilization/process 门禁。

`INFERENCE_BASE_URL` 与 `JUDGE_BASE_URL` 是两个独立变量。`RUN_SCORE=1` 时 pipeline 强制要求
`JUDGE_BASE_URL`，避免把被测模型 endpoint 错当成 judge。`INDICES`/`LIMIT` 自动进入 subset validator，
且 pipeline 硬拒绝 subset scoring。

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

正式评分前可用 `build_stage3_answer_audit.py` 只读加载获准批次的完整 validator 和 prediction，
对所有轨抽取同一组 index 并导出人工抽查文档。它不参与评分，产物写入 Git 忽略的 `outputs/`。

## 输出布局

未显式设置 `OUTPUT` 时，公共路径函数生成：

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
