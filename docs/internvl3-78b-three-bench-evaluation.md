# InternVL3-78B 三 Benchmark 单次 vLLM 补测

本入口在同一台四卡服务器上只加载一次 `OpenGVLab/InternVL3-78B-hf`，依次补齐 Q-Spatial、
SPBench-SI 和 CV-Bench 的 `internvl3_78b`。它只负责跨 benchmark 编排；每项仍调用自己的 inference、
validator、scorer 和 report 脚本，prediction schema、scorer protocol、publication gates 与 canonical
输出根均不改变。三个原有单 benchmark 一键入口继续可独立使用。

## 1. 固定身份与资源

- model revision：`3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356`；
- served model：`internvl3-78b-three-bench`；
- vLLM `0.19.0`、BF16、TP=4、`max_model_len=32768`、单图、seed 42；
- 默认 GPU：`0,1,2,3`，默认端口：`18103`；
- Q-Spatial 仍生成最多 512 token，SPBench-SI 仍生成最多 128 token，CV-Bench 仍使用自身 registry
  decoding。共享服务不会统一或改写三项 decoding；
- controller、service、阶段日志和 `status.tsv` 写入独立的仓库外 control root；正式结果只写回
  `QSPATIAL_OUTPUT_ROOT`、`SPBENCH_SI_OUTPUT_ROOT` 和 `CVBENCH_OUTPUT_ROOT`。

任一 benchmark profile 的 model、revision、served name、TP、processor family 或 seed 与上述身份不符
时，入口在启动服务前 fail closed。

## 2. 配置

将 [`configs/internvl3-78b-three-bench.env.example`](../configs/internvl3-78b-three-bench.env.example)
合并到四卡服务器上的一个未跟踪环境文件。三个 output root 必须已经含有各自的正式基线；control root
必须独立于仓库和三个 output root。

```bash
export INTERNVL3_78B_THREE_BENCH_ENV_FILE=/absolute/path/to/untracked-three-bench.env
```

关键公共变量是：

```text
LATENT_PYTHON
INTERNVL3_78B_MODEL
INTERNVL3_78B_THREE_BENCH_VLLM
INTERNVL3_78B_THREE_BENCH_GPU_IDS=0,1,2,3
INTERNVL3_78B_THREE_BENCH_PORT=18103
INTERNVL3_78B_THREE_BENCH_CONTROL_ROOT
```

此外必须提供三个 benchmark 原有的数据根和输出根。入口把 benchmark-specific `*_ENV_FILE` 固定为
`/dev/null`，因此所有需要的值必须已由这个公共环境文件提供，不能依赖三个脚本再次加载不同环境。

## 3. 只读检查

```bash
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --dry-run
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --status
bash scripts/internvl3_78b/run_three_bench_evaluation.sh --check
```

- `--dry-run` 不读取数据、结果或 GPU，不创建日志，不推理、不评分；输出中应只有一次 `vllm serve`，
  且顺序固定为 Q-Spatial → SPBench-SI → CV-Bench。
- `--status` 只读 control status、三条 78B full provenance 和报告发现结果。
- `--check` 检查路径、模型和 vLLM 版本、端口、四张空闲 80GB GPU、全局及 benchmark 锁、既有结果基线
  和三个 benchmark 自身的 `--check`。端口占用、GPU 不足/繁忙或锁冲突统一退出 `4`，不接管外部进程。

当前只有两张 A800 的服务器只能执行以上只读命令；`--check` 应在四卡资源检查处明确退出 `4`。

## 4. 正式流程

只有四张 80GB GPU 可用且推理、评分再次获得明确授权后，才执行无参数入口：

```bash
bash scripts/internvl3_78b/run_three_bench_evaluation.sh
```

启动前必须恰好满足以下之一：

- CV-Bench 22/23、Q-Spatial 20/21、SPBench-SI 20/21，且三者都只缺 `internvl3_78b`；
- 某项已经存在 provenance 完全匹配的 78B full，可恢复其后续 validator/评分/报告；
- 某项 78B 已完整发布，报告发现数已达到该 benchmark 的目标总数。

运行时序固定为：

```text
启动一次共享 vLLM
  Q-Spatial test -> full-271 -> validator -> 后台 score/report
  SPBench-SI test -> full-1009 -> validator -> 后台 score/report
  CV-Bench test -> full-2638 -> validator -> 后台 score/report
停止自有 vLLM -> 等待三个后台 publication worker -> 复核三个最终报告
```

每项 full validator 通过后，其 CPU 评分与报告任务立即在后台启动，并可与下一项 GPU 推理重叠。评分
失败只隔离该 publication lane，后续 GPU 推理继续，但最终整体非零退出。推理或 validator 失败会停止
后续推理、回收入口自己记录的服务/步骤进程组，并等待此前已经有效的 publication lane；入口不会使用
`pkill`、`killall` 或端口反查去清理外部进程。

## 5. 恢复与互斥

已有 full 只有同时满足完整样本数、当前 profile registry digest、model revision、inference/scorer
protocol、dataset fingerprint、test-gate binding、prediction SHA256 和已存 validator 时才可跳过推理；
跳过前还会再运行该 benchmark 的公开 validator。若三项 full 都有效，入口完全不启动 vLLM，只补齐
缺失的评分/报告。

入口同时持有一个全局锁和三个 benchmark 已有的批次/单模型锁。SPBench-SI 双卡 scheduler 也持有
`SPBENCH_SI_OUTPUT_ROOT/_scheduled_batch/lock`，因此两种控制器不能同时写同一结果根。冲突只返回
资源阻塞，不等待、不抢占。

## 6. 最终验收

整体成功必须同时满足：

- Q-Spatial 271/271、当前 validator/provenance/scorer protocol、`q-spatial-result.md` 21/21；
- SPBench-SI 1009/1009、当前 validator/provenance、主/audit scorer publication gates、
  `spbench-si-result.md` 21/21；
- CV-Bench 2638/2638、当前 validator/provenance/scorer protocol、`cv-bench-result.md` 23/23；
- control `status.tsv` 的 workflow final 为 `COMPLETE`。

没有 validator、metadata、summary 和 publication gate 的现场证据时，不得把“推理完成”写成“评分或
发布完成”。
