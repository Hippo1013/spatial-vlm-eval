# Changelog

本文件只记录会影响评测行为、结果解释、模型覆盖或操作方式的语义变化。逐文件差异和完整时间线以
Git 历史为准；临时调试过程和未定位问题不写入。

## Unreleased

### Added

- MSMU SOTA supplement 的 RoboBrain 环境预检从“Auto 类可导入”收紧为离线解析锁定 NV checkpoint，
  并强制 `model_type=qwen3_vl`；避免 Transformers 版本过旧时通过 `--check` 后才在双 lane canary 失败。
- MSMU SOTA supplement 的 smoke8 冻结身份统一使用 recoverable runtime 的 canonical 数值 index 顺序，
  避免八类选择顺序与落盘顺序不同而误拒绝已通过 validator 的 smoke 产物。

- 为 MSMU 新增 RoboBrain2.5 NV/MT、HiSpatial + same-RGB MoGe-2 XYZ、SpatialLadder direct 与官方
  generic thinking 共五条独立 inference profile。adapter 只接收 MSMU `index/image/question`，锁定官方
  processor/predictor、revision、decoding、XYZ digest、SpatialLadder tied embeddings/FA2/left-padded
  native batch 和 thinking 最后完整 answer-tag 抽取；MSMU scorer protocol、阈值、judge 与 macro-8 不变。
- 增加 MSMU SOTA 双 GPU lane 控制器：GPU0 固定 NV → HiSpatial → Ladder direct，GPU1 固定 MT →
  Ladder thinking；每条 lane 有 pipe-driven 只读 watcher，支持 canary/smoke/full provenance 恢复、只清理
  自有进程组和非法 finalized 产物 fail closed。两 lane 完成后只启动一次 judge，按五条冻结路径串行
  评分；报告 `--check` 要求 baseline18 + main4 + thinking1 唯一完整后才原子重建 23 行结果。
- 四条 main supplement 在现场 full-987、validator、summary、judge failures、publication gates 与报告
  验收前保持在 `CURRENT_TARGET_PROFILE_KEYS` 之外；完成后主矩阵由 18 晋级为 22，thinking 永久作为
  第 23 条补充行。该范围门禁由 ADR-0005 和独立 runbook 固化。

- SPBench-SI 结果表对齐其他 benchmark 的精简展示：模型名与实际输入形式合并为
  `模型（输入）` 单元格，每个指标列的并列最高分全部加粗，Markdown 只展示主协议结果表；
  上游兼容审计仍作为独立评分与
  publication provenance 产物保留，但不进入汇总文档。该变更只影响报告展示，不改 scorer protocol
  或分数。
- SPBench-SI 报告集合不再强制 21/21 或唯一的 20/21 形态：任意非空、逐轨通过完整
  publication gates 的子集都可汇总；可重复使用 `--exclude-profile` 明确排除注册轨，
  Markdown 分开声明排除项与其余未完成项。这只改变报告集合门禁，不改 scorer protocol
  或单轨 validator/provenance/双协议/publication 门禁。
- SPBench-SI Gemini 3.1 Pro 增加 PackyAPI `Gemini-slb` 企业池作为同一模型轨的补充额度来源：隐藏输入
  key 工具只写未跟踪 `.env.server`；专用入口不重跑 test 或 OpenRouter 已成功题，逐条验证旧 journal
  后用新 signature 只补缺失 index。authenticated `/models`、首个缺失题串行 request/response、返回
  model id、reasoning/temperature/token 参数与单图证据均 fail closed；最终 metadata 单列两个 API
  source 的计数和 index digest，报告模型身份与 scorer protocol 不变。
- 调整 InternVL3-78B 三 benchmark 一键入口的完成边界：其他 profile 的报告源不完整或发现失败不再
  阻塞 78B 推理与评分；三个 benchmark 均强制完成目标 validator、精确单轨评分和 publication gates，
  仅当对应基线恰好只缺 78B（或恢复时已完整）才重建全局报告，否则明确记录 `report=skipped`。
- 修复 SPBench-SI SpatialLadder native batch 与锁定官方 runner 不一致的问题：processor 现在强制并
  fail-closed 验证 tokenizer left padding，capacity probe 改用两种长度的 red/blue prompt，generation、
  processor audit 与 gate 均保存 padding 证据。该轨 inference protocol 升为 v2；服务器旧 right-padded
  v1 gate/full 即使通过结构 validator 也已作废，不能评分或恢复，等待获得 GPU test/full 授权后重跑。
- SPBench-SI 主 scorer 升级为真实输出驱动的 v2 parser：自由文本 `a/an` 不再误提为 1，仅在强答案区域
  剥离 `A-D.` 数值标签，受控识别最后的 distance/longest-dimension 声明，并按题型期望单位选择模型
  显式写出的同单位数值而不做换算。v1 inference metadata 明确兼容、v1 score 不再是当前结果；锁定
  SpatialLadder upstream compatibility audit 的提取与 inclusive MRA 字节语义保持不变。
- 增加 InternVL3-78B 三 benchmark 单次 vLLM 补测入口：三个 profile 统一 served name，固定 vLLM
  0.19.0/BF16/TP=4/四卡/32768 上下文，按 Q-Spatial 271 → SPBench-SI 1009 → CV-Bench 2638 串行
  推理；每项 validator 通过后并行调用其现有 scorer/report，支持严格 provenance 恢复、全局与 benchmark
  锁、评分故障隔离、推理 fail-fast 和仅清理自有进程组。SPBench-SI scheduler 同步补齐输出根互斥锁。
- 增加 Q-Spatial InternVL3-78B 独立四卡补测入口：固定 BF16/TP=4，按当前 registry/binding/scorer
  protocol 顺序执行 test/full-271、validator、精确单轨评分，并在原 `QSPATIAL_OUTPUT_ROOT` 中把既有
  `q-spatial-result.md` 原地重建为 21/21；提供 check/status/dry-run、严格恢复门禁和内置迁移 FAQ。
- 增加独立 SPBench-SI 单图全链路：锁定 SpatialLadder commit `7a0d2ee` 与数据 revision
  `03611025`，直接从锁定 ZIP 验证/解码 524 张 JPEG，提供 1,009 条防泄漏输入合同、21 条目标 profile、
  default/direct prompt、red/blue + smoke8 绑定 gate、fsync 恢复、双卡 20 轨失败隔离调度和只读 watcher。
- 增加 SPBench-SI 原始十阈值严格 MRA 主 scorer：唯一 final/tag 答案、冲突 fail-closed、四题型宏平均，
  并在独立目录精确保留当前 SpatialLadder direct-mode 提取与 inclusive 边界 compatibility audit；入表轨
  必须逐条通过完整 publication gates。
- 修复 SPBench-SI 双卡 test 调度的端口可用性探针与清理失败记账，并确保共享 server env 不覆盖逐轨
  GPU 分配和 LLaVA-NeXT 4096 上下文；服务器 20 条非 78B 轨已通过当前 binding 的完整 test gate。
- 增加 SPBench-SI InternVL3-78B 四卡独立全链路：固定 BF16/TP=4，自有 vLLM 顺序执行 test/full-1009、
  validator、目标双协议评分与原报告 21/21 重建，并提供只读 check/status、dry-run、恢复和内置 FAQ。

- 2026-08-07 服务器现场复核：Q-Spatial 除 TP=4 blocked 的 InternVL3-78B 外，20 条计划轨均通过
  red/blue canary + smoke8 当前 test gate、full-271、正式 validator、完整 provenance、当前 v2 scorer
  评分与 publication gates；全局报告为 20/21。

- 2026-08-08 服务器现场复核：SPBench-SI 报告发现器逐轨验证出 20 条当前可发布候选，包含
  InternVL3-78B 与 left-padded v2 SpatialLadder；Gemini 续跑仍未形成可发布 summary。按操作者本次选择，
  结果文档明确排除 Gemini 与 InternVL3-78B，汇总其余 19 条可发布轨。

- 修复 Q-Spatial 目录评分把 `test_artifacts/` 与 `test_artifacts.stale-*` 中的 smoke8 prediction 误纳入
  正式候选的问题；当前发现器只冻结 20 条 full 结果，旧 `test_runs/`、shards 与 score 子树仍被排除。

- Q-Spatial 双卡/API 调度器新增显式 `--stage test`：复用同一 20 轨冻结分队与失败隔离，只建立或复用
  当前绑定的 test gate，绝不进入 full/正式 validator/评分；`--skip-completed` 仍仅服务 full 模式。
- 修复 Q-Spatial 调度器回收自有 vLLM 时未先 reap 已退出 group leader、导致每次换模误等完整 stop
  timeout 的问题。
- 修复 Q-Spatial LLaVA 两阶段第二次 vLLM 请求同时启用 `continue_final_message` 与默认 generation
  prompt 而被拒绝的问题；assistant prefill 现显式设置 `add_generation_prompt=false`。
- Q-Spatial test gate 因 binding 更新失效时，自动把旧 test artifacts/gate 无损轮换为带旧 digest 的
  `stale-*` 归档，避免跨 resume signature 混用或阻塞合法重测。
- Q-Spatial 调度器停止自有 vLLM 后有限等待监听端口实际释放，消除 GPU 已清空但 socket 尚未解绑的
  换模竞态；超时仍拒绝接管端口。
- Q-Spatial vLLM 默认上下文固定为 32768 并纳入 test binding，覆盖 Qwen3-VL 合法图像 token 输入与
  1024 输出预算；LLaVA-NeXT Yi/Mistral 由调度器逐 profile 保持 checkpoint 合法的 4096 上限。
- Q-Spatial 3DThinker 为同一 RGB 的 checkpoint processor 绑定 `12544..401408` pixels 并记录
  provenance，避免大图视觉 attention OOM；prompt、单图边界和 decoding 不变。
- 增加独立 Q-Spatial Bench 全链路：锁定官方代码 `ebe8137` 与数据 revision `17b92e4`、两个显式
  数据根、170+101 行及 99-frame ScanNet manifest，不可泄漏的 system/user 单图输入、两字段
  prediction validator、21 条目标 profile、test/full 绑定 gate、目录评分和 publication-gated 报告。
- 增加 Q-Spatial tag-first robust numeric scorer：确定性单位换算、论文 inclusive `δ≤2` / `δ≤1.25`、
  ScanNet/Q-Spatial++ 等权 Overall、ScanNet 五类分项，以及同批次旧 notebook strict-threshold 审计；
  malformed/冲突答案保守记零且逐行保留差异。
- 扩展 OpenAI-compatible client 以按 benchmark 输入可选发送 system + single-image user messages，并
  支持 vLLM `top_k`、presence/repetition penalty、seed 与自定义 token 上限；MSMU/CV-Bench 原有
  user-only payload 保持不变。增加 LLaVA 两阶段格式修复、纯红/纯蓝 canary、smoke8、单 endpoint
  并发和独立 Q-Spatial specialized JSONL bridge。
- 增加 Q-Spatial 双卡/API 分阶段控制器：冻结 20 轨计划，阶段 A 双卡与串行 API lane 并行，双卡成功
  后阶段 B 才启动 GPU 0/1 独立 lane；每 job 复用合法 gate 或 test 后运行 full/validator，严格复核完整
  skip provenance，隔离 lane 失败并只清理 owned process group。增加 `_scheduled_batch` 状态/日志与
  `tmux wait-for` 只读 health watcher；控制器不评分。
- 增加独立 CV-Bench 全链路：锁定 revision `bc284db50d036958861cb60cdd7b77612052ce0d`
  的 2D/3D 两个 Parquet（2638 条）、不可泄漏的单图输入合同、两字段 prediction validator、23 条目标
  profile registry、test/full 两阶段绑定 gate、目录驱动评分与 publication-gated Markdown 报告。
- 增加 CV-Bench answer-tag-aware robust multiple-choice scorer v2：只接受唯一合法字母、唯一完整
  选项文本或唯一 `<answer>...</answer>` 最终答案，冲突/多答案/越界/空值保守记零；主指标固定为
  ADE/COCO 等权 2D、Omni3D 3D 及二者等权 Overall，micro accuracy 仅作审计。
- 增加 benchmark-neutral 可恢复推理 runner，以及 CV-Bench 通用 vLLM/OpenRouter adapter、官方
  Transformers processor/template 审计、组合视觉 canary、固定 smoke8、vLLM 容量探测、双 endpoint
  确定性分片和专用上游 persistent JSONL bridge。专用 runner 缺实现 SHA 或 generation manifest 时
  fail closed。
- 增加 CV-Bench 通用开源轨的 registry-driven vLLM 0.19 单 endpoint 启动器；启动前检查所选 GPU
  空闲状态，按 profile 锁定 revision、served name、TP、BF16、单图上限与 seed 42。
- 增加 CV-Bench OpenRouter key 的交互式隐藏输入工具：只写入未跟踪的 `.env.server`，原子替换旧值并
  固定 mode 600，避免 key 出现在 shell history、命令参数或运行日志中。
- 增加 CV-Bench registry-driven full 串行控制器：可明确排除 InternVL3-78B，自动为通用开源轨启动、
  验证并停止其拥有的单/双 endpoint vLLM 服务，再按顺序运行 API 与专用轨；任一失败立即停止且不评分。
- 增加 CV-Bench 只读逐条结果 watcher：自动跟随 full 串行批次的当前 profile，只读取正式 append-only
  journal 并打印精简 success/failure；默认从启动时刻继续，支持 `--from-start` 重放当前模型。
- 增加 CV-Bench InternVL3-78B 四卡单模型一键入口：固定 TP=4/BF16 和 registry decoding，自动复用
  test gate/journal/完整结果，顺序执行 full-2638、独立 validator、精确单轨评分和原有全局报告重建；
  正式 prediction、score 与 `cv-bench-result.md` 继续使用既有 canonical 路径。
- 将 CV-Bench 独立完整 prediction validator 明确为推理与评分之间的公开阶段；CLI 默认读取
  `CVBENCH_DATASET_ROOT`，评分入口仍强制重复校验且不提供绕过参数。
- 增加 12 条空间专用轨共用的 dataset-blind persistent runner、五份从锁定上游/checkpoint 解析的
  generation manifest，并把 HiSpatial 的 MoGe-2 checkpoint revision 与上游 commit 纳入 profile
  binding；既有 MSMU 专用 adapter 仅新增显式 generation/token-cap 注入点，默认 MSMU 行为不变。
- 为 GPT-5 与 Gemini 3.1 Pro 增加用户明确授权的 OpenRouter non-ZDR 独立 profile、inference protocol、
  run slug 和三阶段入口；仍锁定首方 provider、禁止 fallback、要求完整参数并设置
  `data_collection=deny`，不改写原 ZDR 轨或 scorer protocol。
- 增加注册模型通用的 MSMU 单模型一键正式评测入口：默认只运行 stage 3，按共享 manual-stage 注册
  信息自动管理被测 vLLM 与独立 judge，精确评分本次 `predictions.jsonl`，通过 publication gates 后
  重建全局结果报告；支持 API、vLLM、Qwen 和空间专用 adapter，不维护第二份模型名单。
- 为目录驱动评分增加绝对 `--predictions` 单结果选择器；仍在全局评分锁、judge readiness、完整
  validator 和 publication gates 下执行，不改变 scorer/cache protocol 或批量默认发现行为。
- 增加跨 scorer protocol 发现的 MSMU Markdown 结果表生成器，支持 publication-gated 全量汇总和
  metadata profile/单 scorer protocol 精确筛选；输出固定为 `msmu-result.md` 中文精简表，专用模型
  按 profile 直接标注 `RGB`、`RGB + 深度估计` 或 `RGB + Mental-3D 提示词`，SpatialRGPT 不加展示
  注释，未知双轨 profile 无显式配置时 fail closed；默认按 API、通用开源、空间专项及参数量/输入先验
  排序，并加粗各指标列和平均列的并列最高分；精确 provenance 保留在已校验的 metadata、summary
  与结果目录。
- 将当前 Qwen 横评计划从 Qwen2.5-VL 7B/32B/72B 更新为 Qwen3-VL-Instruct
  2B/4B/8B/32B，锁定四个独立 revision、inference protocol、原生无 system chat template、
  greedy/192-token decoding 和等视觉 token 的 `16384..147456` pixel 范围。
- 增加共享 Qwen-VL 推理核心和 Qwen3-VL adapter，并扩展现有 Qwen pipeline 与三阶段 MODEL
  参数；Qwen2.5-VL/PEFT adapter 与历史结果继续保留。
- 为现有阶段三串行脚本增加 `--qwen3` 计划，仅依次运行四条 Qwen3-VL 补测轨，并与原 13 轨状态隔离。
- 将 stage 1 视觉语义 canary 统一为一张 512×512 抗锯齿白底组合图（左上红圆、右下蓝方块），由
  4× 超采样后 LANCZOS 缩小确定性生成，并将 canary protocol 升为 v2；要求颜色、形状和位置关联
  全部正确。Qwen 通过同模型/processor 执行，OpenAI-compatible API/vLLM 通过同 adapter
  执行。GPT-5/Gemini 各只增加 1 次无 inference retry 的 generation，并继续强制 OpenRouter
  provider/model/media audit，避免只凭请求结构或非空图像张量判定模型已看图。

### Changed

- 2026-08-11 只读复核 canonical 产物后，项目进度文档更新为：MSMU 当前目标 18/18、CV-Bench
  23/23、Q-Spatial 21/21；SPBench-SI 有 20/21 条可发布候选，仅 Gemini 缺失，现有结果表按操作者
  选择纳入 19/21 并额外排除已完成的 InternVL3-78B。进度快照统一收口到
  `docs/evaluation-scope.md`，README、模型矩阵和 runbook 不再复制易漂移的阶段状态。本次只读复核与
  文档整理没有启动推理、GPU、评分或 API。

- 明确所有 benchmark 的闭源 API 轨只作补充参照，不是项目核心比较对象或阶段收尾的强制完成条件；
  未完成轨可以显式搁置，不能为了形式完整度自动发起付费调用。已经运行并准备入表的闭源结果仍须
  通过原有 validator、provenance、scorer protocol 与 publication gates。SPBench-SI Gemini 轨按此
  共识搁置，保留恢复入口但只在新的明确需求和付费授权后使用。

- Q-Spatial Markdown 汇总移除旧 notebook 解析分数、主/旧差异条数及相关文字，只展示当前 v2 scorer
  的 `δ≤2` 主结果、ScanNet 五类明细与 `δ≤1.25` 严格阈值；主表同时沿用 MSMU 命名方式，把实际派生
  输入配置写入模型名括号并移除独立 input/comparison 列。底层兼容性审计与分组比较规则继续保留。

- Q-Spatial numeric scorer 升级为 declared-final v2：接受等价重复标签、唯一 final 标签、unit-only 标签、
  紧凑单位、LaTeX boxed/distance 与 diameter/unit wrapper、简单分数，并排除 `PS4` / `Region [0]`
  标识数字；范围、冲突、多候选和缺单位继续 fail closed。零或未知单位保留模型声明但计零。v2 只读兼容
  声明 v1/v2 scorer 的既有完整 inference metadata，无需重跑或改写 prediction。

- 修复 Q-Spatial `spatialbot_zoedepth` 的模型边界门禁：继续强制一张源 RGB，同时严格接受一个 RGB
  tensor 加一个由同图派生的 depth tensor；其他专用轨仍要求恰好一个 model image tensor。

- Q-Spatial TP=1 vLLM 从双 GPU 双 endpoint 奇偶分片改为单 GPU 单 endpoint 内请求并发；TP=2/4 保持
  单 tensor-parallel endpoint。endpoint/GPU/sharding 仍进入 binding，旧双 endpoint test gate 自动失效；
  vLLM capacity 保持 `32→16→8→4→2→1`，API 独立使用 `8→4→2→1`。

- 将 CV-Bench robust multiple-choice scorer 升级为 v3 declared-answer parser：仅在解析视图剥离已知
  末尾生成 token，支持首/末行独立字母和字母/完整选项文本一致的紧凑格式，为每条结果记录
  `parse_evidence`；竞争字母、文本冲突、多 tag、越界和截断继续判零。修复 `options and` 的复数
  单词边界误识别，并显式允许 v3 scorer 消费未改写的 v2/v3 inference metadata，不引入模型提取器。
- CV-Bench 只读逐条结果 watcher 增加双 lane 选择：`--lane gpu0|gpu1` 分别跟随独立 status 与当前正式
  journal，识别 PASS/FAIL/BLOCKED/COMPLETE 并自动随该 lane 换模；原单串行默认入口保持兼容。
- CV-Bench full 串行控制器增加显式 `--skip-completed`：在启动模型服务前重新以锁定数据完整校验现有
  2638 条 prediction，仅验证通过的轨记录 `SKIP_COMPLETE` 并跳过，避免恢复批次重新加载已完成模型。
- 按目标测试策略将 CV-Bench 全部 23 条轨统一改为纯红、纯蓝 RGB 图颜色识别最低视觉 canary，取消
  形状、方位和空间描述能力门槛；smoke8、单图边界审计及其他 provenance gate 保持不变。已通过旧版
  组合空间语义 canary 的轨可在严格 artifact 审计后迁移当前 gate，避免重复模型调用；颜色回答只需
  明确包含目标色，允许 `blue-purple`、`red-orange` 等近色措辞。当前 protocol 的 gate 若仅组合
  adapter source digest 改变、其余 binding 完全相同，也可审计迁移而不重复调用模型。
- CV-Bench 服务器 test stage 已现场完成 22/23 条轨的红/蓝视觉接收、smoke8 和单图审计 gate；
  InternVL3-78B 因当前服务器仅有 2×A800、协议要求 4×80GB GPU 而保持阻塞。排除该轨后的 22 条目标轨
  已于 2026-08-06 全部通过 full-2638 validator、scorer v3 评分和 publication gates；全局报告为
  22/23 并明确只缺 `internvl3_78b`。
- CV-Bench prompt 冲突修复后的 `3dthinker_mental3d` 与 `spatialladder3b_thinking` 已现场重跑并通过
  v2 test gate；逐条 journal 审计确认 smoke8 均为单图、包含 reasoning answer tags，且不再含
  direct-answer 后缀。
- OpenAI-compatible 可恢复 runner 仅对 429/5xx 执行指数退避；非重试型 HTTP 错误不重复请求，成功
  journal 继续保证 resume 不重复付费。CV-Bench 本地模型每次 test/full 另保存只读 GPU inventory 与
  compute-process 审计；InternVL3-78B 强制显式枚举四张 80GB GPU。
- 将项目级待测范围扩展为 MSMU-Bench、CV-Bench、Q-Spatial Bench、SPBench-SI 四个 benchmark；MSMU
  既有 18 条 profile 已完成，下一实施对象为 CV-Bench。目标模型范围在原有 15 个模型身份上新增
  RoboBrain2.5-8B-NV、RoboBrain2.5-8B-MT、HiSpatial-3B 和 SpatialLadder-3B，共 19 个模型身份；
  新增模型尚在下载且未注册 profile，不追溯计入 MSMU 已完成结果。
- 将当前 MSMU 目标测试矩阵固化为 18 条已经通过 full-987 validator 与 publication gates 的结果轨；
  目标范围由 `CURRENT_TARGET_PROFILE_KEYS` 统一维护，历史 adapter、注册 profile 与结果继续保留用于
  复现，但不再混入当前目标矩阵。
- 将服务器项目与正式输出根迁移到 `/media/datasets/lihaoran/`；服务器配置将未来 Hugging Face
  data/model、Conda env/package、pip/uv/PyTorch cache、upstream 与 checkpoint 下载统一路由到新
  namespace，同时让当前已验证的 dataset、模型、解释器和 upstream 继续显式引用未改动的
  `tangzecong` 资产。shell 编排仍只读取环境变量，不硬编码任一服务器 namespace。
- OpenAI-compatible API inference 在首轮遍历结束后固定对仍缺失的 index 再执行一轮；已成功 journal
  项不会重复请求，补跑后仍不完整则继续拒绝 finalization，并允许相同命令从 journal 续跑。
- OpenRouter generation metadata 的默认查询窗口扩为 10 次 metadata-only 重试，并让 canary 与
  inference wrapper 共用 `OPENROUTER_METADATA_RETRIES`；处理 completion 成功后 metadata 短暂 404
  的最终一致性延迟，不会重发付费 completion，也不改变推理协议。

- OpenRouter HTTP/API 错误现在在 canary 失败报告中保留脱敏后的 status、typed error、router metadata、
  request/generation ID 和耗时；API key、cookie 与响应正文不进入产物，不改变成功响应或推理协议。

- 将 OpenRouter GPT-5 与 Gemini 3.1 Pro 的 generation metadata 校验从请求别名改为精确锁定 catalog
  canonical revision（分别为 `openai/gpt-5-2025-08-07` 与
  `google/gemini-3.1-pro-preview-20260219`）；仍拒绝任意其他返回 revision，并在 non-ZDR 输出路径与
  run metadata 中记录该 revision。
- 将 GPT-5 与 Gemini 3.1 Pro 两条 OpenRouter non-ZDR 能力轨从 low/512 v2 升级为 medium reasoning、
  16384 total completion tokens 的 v3，并分别更换 run slug；原因是 v2 正式运行仍出现 hidden
  reasoning 耗尽预算的空 prediction 与可见回答截断，且 EASI 对相同 `gpt-5-2025-08-07` revision
  的正式空间评测采用 medium/16384。既有 v2 journal/结果仅作历史诊断，不恢复或混入 v3。
- 将 InternVL3-78B 从仅静态检查改为独立四卡手工补测轨：固定 BF16、TP=4 和默认 GPU `0,1,2,3`，
  serve 前同时校验选中 GPU 与物理 GPU 均不少于四张，并放开 stage 1/2/3 手工入口；已完成的阶段三
  历史 13 轨默认名单和完成标记保持不变。

### Fixed

- CV-Bench 与 Q-Spatial 公共 shell 入口在未显式设置 benchmark-specific `PYTHON` 时，会在系统
  `python` 前复用 `.env.server` 的 `LATENT_PYTHON`；避免服务器系统解释器缺少项目依赖时连
  `--list` / `--dry-run` 都失败，显式 `CVBENCH_PYTHON`、`QSPATIAL_PYTHON` 或 `PYTHON` 仍优先。
- 修复 CV-Bench 本地 vLLM full 的长尾恢复：将本地请求超时与 OpenRouter 超时解耦，默认延长为 600
  秒，并把即时重复请求改为首轮结束后仅补 journal 缺失 index，避免官方 512-token 配置下极少数长
  输出连续超时导致完整 2638 条无法原子落盘。
- 将 CV-Bench 最终回答格式从全局数据层移到 profile 层：普通轨继续追加 direct-letter 后缀，
  `3dthinker_mental3d` 与 `spatialladder3b_thinking` 只保留各自官方 `<think>/<answer>` prompt，避免
  “直接回答”与“先推理再回答”同时出现。两条 reasoning inference protocol 升为 v2，旧 test gate
  自动失效；scorer 的 answer-tag 解析未变，因此 scorer protocol 保持不变。
- SpatialLadder runner 现在把 checkpoint 嵌套 `text_config` 中的 tied-output 声明传播到模型外层
  config，并锁定 PyTorch SDPA；避免缺失 `lm_head` 被随机初始化后继续生成的不可信状态。
- HiSpatial/MoGe-2 runner 现在锁定并核验 MoGe requirements 指定的 `utils3d` commit，同时比对环境中
  实际导入的关键文件与 checkout，避免新版包移除 `utils3d.pt` 兼容别名后在推理中途失败。
- HiSpatial runner 现在把锁定 MoGe-2 snapshot 内的 `model.pt` 文件传给上游 loader，不再把目录误作
  torch checkpoint；revision、upstream commit 和文件名均 fail-closed。
- CV-Bench SpatialBot runner 现在验证并显式绑定锁定的本地 SigLIP vision tower，避免上游 checkpoint
  config 在离线测试中按仓库名隐式联网；ZoeDepth 的 MiDaS torch.hub 请求同样改绑到锁定 commit 的
  本地 checkout；checkpoint 与 legacy 只读快照均不修改。
- 3DThinker Mental-3D adapter 在保留 index `-1` 的视觉 canary 中验证完整 raw response，不再先抽取
  `<answer>` 而丢失颜色/形状/位置证据；真实 benchmark index 继续只保留最后完整 answer tag，正式
  prediction 语义不变。
- 将组合视觉 canary 升级为 bbox-aware v4：用不泄露具体答案的 quadrant/corner 问句要求逐对象描述，
  在继续严格要求红圆/蓝方块与左上/右下正确关联的前提下，接受模型输出的合法归一化 bbox 作为位置
  证据，并拒绝交换、越界或与方位词冲突的框；CV-Bench test binding 现在显式包含 canary protocol，
  旧 gate 自动失效并必须重测。
- 移除 MSMU 单模型/串行 stage-3 控制器对系统 `curl` 的非必要硬依赖；本地 OpenAI-compatible 服务
  readiness 现在由配置的 `LATENT_PYTHON` 标准库探针检查，并精确匹配 `/v1/models` 中的 model ID。

### Documentation

- 增加 Q-Spatial canonical protocol、两阶段 runbook、简明命令、21 轨模型矩阵、numeric scorer ADR
  和遗留小问题；同步 README、架构、评测范围、文档地图、来源记录、AGENTS 路由与论文目录索引。
- 增加 CV-Bench canonical protocol、两阶段 runbook、23 条目标 profile 矩阵与 robust parser/publication
  gate ADR，以及面向操作者的精简 test/full/评分/汇总命令页；同步架构、评测范围、文档地图、来源记录
  和文档一致性测试。
- 增加四 benchmark 评测范围文档，区分外部 SOTA 报告值与项目复现结果，记录三个待实现 benchmark
  的 legacy 数据位置、新模型下载位置、公平/原生输入边界和 CV-Bench 实现前门禁。
- 增加 `msmu-a800` Mihomo 显式出站代理手册，记录仓库外安装、tmux/PID 生命周期、按 shell 开关、
  本机监听与出口验证；订阅、完整节点身份和出口 IP 不进入仓库。
- 建立统一文档地图、维护触发规则、ADR 决策记录和 troubleshooting 知识库。
- 增加文档链接、profile 矩阵、阶段三名单与 scorer protocol 的一致性检查。
- 明确仓库外 `MANUAL_TEST_OUTPUT_ROOT` 才是正式推理/评分结果根；人工抽查与派生导出同样写在
  仓库外，仓库根禁止创建 `output/` 或 `outputs/`。
- 将 `msmu-a800` burn 恢复命令与服务器实际运行的 A800 monitor/watchdog 脚本保持一致。
- 明确 Agent 记忆、项目文档与未跟踪运行日志的职责和晋升路径；按需读取任务相关文档，移除推理手册
  中重复的阶段三命令，并为规则/文档尺寸和用户 `tmp/` 草稿区增加自动门禁。

## 2026-07-31

### Changed

- 将所有 judge 路径在重试后仍 malformed/schema-invalid、但已返回文本的响应缓存并保守记为零分；
  无响应的网络或 worker 故障仍阻断 publication。该语义将 scorer protocol 升级为 v4
  （`442dbb7`）。
- 定性 judge 响应允许确定性恢复首行裸 `"your_mark": 0/1`，同时保持 scorer/cache protocol
  不变并记录恢复日志（`5e30876`）。
- 记录阶段三 13 条获准本地轨的完整 validator 状态和固定样本答案抽查流程（`0aa99aa`）。

## 2026-07-30

### Added

- 增加目录驱动的阶段三串行评分入口，按结果目录发现待评分轨并执行完整 publication gates
  （`1a41600`）。
- 增加面向操作者的精简评分命令与只读检查流程（`d9e8f89`）。

## 2026-07-29

### Added

- 增加 Qwen2.5-VL 32B/72B 独立 profile、revision、协议与输出目录（`82cd58d`）。
- 增加可恢复的阶段三 13 轨串行推理、watchdog、批次锁和 GPU 释放门禁（`3660402`）。

### Fixed

- 完成 SpatialBot/ZoeDepth 在锁定 TIMM、Torch 和 MiDaS 组合下的兼容处理，并以回归测试锁定
  resize、derived buffer、legacy import 和 relative-position 行为（`1bc3874` 至 `2d0f959`）。

## 2026-07-28

### Added

- 增加 MSMU 三阶段人工测试文档与统一入口脚本（`109dd1a`、`05c0cea`）。

### Fixed

- 修复 SSR 上游 `autoroot` 导入锚点和 Transformers adapter 的离线加载参数传递
  （`b5e28d3`、`843c05d`）。

## 2026-07-26

### Added

- 建立 MSMU 数据合同、六字段 prediction schema、严格 validator、本地 judge scorer 和基础测试
  （`2182df9`、`c92d0e4`）。
- 增加多模型 inference profile、可恢复 journal、输入审计、revision/provenance 校验和服务器部署配置
  （`748020f`）。
