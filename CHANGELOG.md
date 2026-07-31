# Changelog

本文件只记录会影响评测行为、结果解释、模型覆盖或操作方式的语义变化。逐文件差异和完整时间线以
Git 历史为准；临时调试过程和未定位问题不写入。

## Unreleased

### Added

- 增加跨 scorer protocol 发现的 MSMU Markdown 结果表生成器，支持 publication-gated 全量汇总和
  metadata profile/单 scorer protocol 精确筛选；输出固定为 `msmu-result.md` 中文精简表，专用模型
  按 profile 直接标注 `RGB`、`RGB + 深度估计` 或 `RGB + Mental-3D 提示词`，SpatialRGPT 不加展示
  注释，未知双轨 profile 无显式配置时 fail closed；精确 provenance 保留在已校验的 metadata、
  summary 与结果目录。
- 将当前 Qwen 横评计划从 Qwen2.5-VL 7B/32B/72B 更新为 Qwen3-VL-Instruct
  2B/4B/8B/32B，锁定四个独立 revision、inference protocol、原生无 system chat template、
  greedy/192-token decoding 和等视觉 token 的 `16384..147456` pixel 范围。
- 增加共享 Qwen-VL 推理核心和 Qwen3-VL adapter，并扩展现有 Qwen pipeline 与三阶段 MODEL
  参数；Qwen2.5-VL/PEFT adapter 与历史结果继续保留。
- 为现有阶段三串行脚本增加 `--qwen3` 计划，仅依次运行四条 Qwen3-VL 补测轨，并与原 13 轨状态隔离。
- Qwen stage 1 增加同模型、同 processor 的红/蓝合成图语义 canary，避免只凭非空图像张量判定模型已看图。

### Documentation

- 建立统一文档地图、维护触发规则、ADR 决策记录和 troubleshooting 知识库。
- 增加文档链接、profile 矩阵、阶段三名单与 scorer protocol 的一致性检查。

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
