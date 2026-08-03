# 服务器问题与解决方法

仅记录 Ubuntu 服务器端开发、部署和评测实验中已经定位的报错及解决方法。通用规则与原始日志位置见
[Troubleshooting Knowledge Base](README.md)；本机环境、Git/GitHub、文档编辑和一般协作事项不得写入。

## 书写规则

- 新条目置顶；同类问题合并更新，不重复记录。
- 只记录服务器端问题，并注明相关运行环境或任务阶段。
- 每条最多 6 行，只保留关键报错，禁止粘贴完整 traceback 或日志。
- 必须写明原因、处理和验证结果；尚未定位的问题不写入。
- 不记录 token、密码、私有 endpoint 等敏感信息；大文件和完整日志只写路径。

## 条目模板

```markdown
### YYYY-MM-DD · [benchmark/model] 简短问题名
- 场景：服务器环境、运行阶段、模型/协议及关键配置。
- 报错：`最有辨识度的一行错误`。
- 原因：一句话说明根因。
- 处理：实际生效的修改或命令。
- 验证：smoke/full run 结果；相关脚本、commit 或日志路径。
```

## 已解决问题

<!-- 按模板在此处下方插入条目，最新条目在最上方。 -->

### 2026-08-01 · [MSMU/deployment] 单模型评测镜像缺少 curl
- 场景：远程任务运行 `run_model_evaluation.sh internvl3_78b`，准备启动 stage 3。
- 报错：`[msmu-eval] required command is unavailable: curl`。
- 原因：控制器只为本地 `/v1/models` readiness 查询而把外部 `curl` 错列为硬依赖。
- 处理：单模型和串行控制器统一改用 `LATENT_PYTHON` 标准库探针，并严格匹配返回的 model ID。
- 验证：探针覆盖 ready、相似 ID、malformed、HTTP error 和非本地 endpoint；shell 与相关 unittest 通过。

### 2026-08-01 · [MSMU/OpenRouter] completion 成功后 generation metadata 短暂 404
- 场景：Mac 端运行 Gemini 3.1 Pro OpenRouter non-ZDR stage 1 组合视觉 canary。
- 报错：completion 已生成 generation id，但紧随其后的 `/generation?id=...` 在原约 4 秒窗口内持续返回 404。
- 原因：OpenRouter generation metadata 最终一致；同一 generation id 稍后返回 200，并确认 Google AI Studio、锁定 revision 和 `num_media_prompt=1`。
- 处理：metadata-only 重试默认增至 10 次（累计约 16 秒），canary 与 inference wrapper 共用 `OPENROUTER_METADATA_RETRIES`；不重发付费 completion。
- 验证：eventual-consistency 回归测试确认只调用一次 completion，404 后仅重试 metadata GET。

### 2026-08-01 · [MSMU/API canary] 极小平面 PNG 触发空 502
- 场景：OpenRouter non-ZDR 首方 GPT-5/Gemini stage 1 组合视觉 canary。
- 报错：多模态请求约 3 秒返回 `HTTP 502 ... empty response body`，无 generation/router metadata；同路由纯文本 200。
- 原因：任意采用的 256×256、约 1.1 KB 硬边 PNG 触发 OpenRouter→首方 provider 图像适配异常。
- 处理：canary v2 改为 512×512、4× 超采样后 LANCZOS 缩小的确定性抗锯齿 PNG；保持首方 provider only、无 fallback。
- 验证：同一 Gemini 路由用新图返回 200，并正确识别左上红圆与右下蓝方；canonical 双模型报告见 stage 1 输出目录。

### 2026-07-26 · [MSMU/deployment] macOS archive caused dubious ownership
- 场景：从 macOS 同步工作树到 `msmu-a800` 后执行服务器 Git 检查。
- 报错：`fatal: detected dubious ownership in repository`，并出现 `._*` AppleDouble 文件。
- 原因：BSD tar 携带本机 numeric UID 与扩展属性，root 解包后保留了错误 owner。
- 处理：隔离 `._*`；目标 repo 恢复服务器既有 `root:root`；后续解包使用 `--no-same-owner`。
- 验证：服务器可直接执行 `git status`、`git diff --check`，repo 内无 `._*`。

### 2026-07-26 · [MSMU/environments] Conda launcher unavailable
- 场景：`msmu-a800` 创建四个依赖互斥的专用模型环境。
- 报错：一个 Conda base Python `Segmentation fault`，另一个 launcher 指向失效旧路径。
- 原因：launcher/base 损坏；既有 `latent`、`vllm019` 环境的绝对 Python 本身仍可运行。
- 处理：共享环境不写包；以可用解释器创建 `uv venv --system-site-packages` 隔离 overlay。
- 验证：各 overlay 保存 base/pip manifest；本仓库服务器 69 项 unittest 与 compileall 通过。
