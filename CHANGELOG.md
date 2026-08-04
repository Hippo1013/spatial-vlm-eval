# Changelog

本文件只记录会影响评测行为、结果解释、模型覆盖或操作方式的语义变化。逐文件差异和完整时间线以
Git 历史为准；临时调试过程和未定位问题不写入。

## Unreleased

### Added

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

- CV-Bench full 串行控制器增加显式 `--skip-completed`：在启动模型服务前重新以锁定数据完整校验现有
  2638 条 prediction，仅验证通过的轨记录 `SKIP_COMPLETE` 并跳过，避免恢复批次重新加载已完成模型。
- 按目标测试策略将 CV-Bench 全部 23 条轨统一改为纯红、纯蓝 RGB 图颜色识别最低视觉 canary，取消
  形状、方位和空间描述能力门槛；smoke8、单图边界审计及其他 provenance gate 保持不变。已通过旧版
  组合空间语义 canary 的轨可在严格 artifact 审计后迁移当前 gate，避免重复模型调用；颜色回答只需
  明确包含目标色，允许 `blue-purple`、`red-orange` 等近色措辞。当前 protocol 的 gate 若仅组合
  adapter source digest 改变、其余 binding 完全相同，也可审计迁移而不重复调用模型。
- CV-Bench 服务器 test stage 已现场完成 22/23 条轨的红/蓝视觉接收、smoke8 和单图审计 gate；
  InternVL3-78B 因当前服务器仅有 2×A800、协议要求 4×80GB GPU 而保持阻塞。排除该轨后的 22 条目标轨
  已于 2026-08-04 启动 registry-driven full-2638 串行推理；已验证完成轨由 validator 复核后跳过，
  当前未启动评分。
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
