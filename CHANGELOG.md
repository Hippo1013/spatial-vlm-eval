# Changelog

本文件只记录会影响评测行为、结果解释、模型覆盖或操作方式的语义变化。逐文件差异和完整时间线以
Git 历史为准；临时调试过程和未定位问题不写入。

## Unreleased

### Added

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

- 移除 MSMU 单模型/串行 stage-3 控制器对系统 `curl` 的非必要硬依赖；本地 OpenAI-compatible 服务
  readiness 现在由配置的 `LATENT_PYTHON` 标准库探针检查，并精确匹配 `/v1/models` 中的 model ID。

### Documentation

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
