# Server Development Log

仅用于记录 Ubuntu 服务器端开发、部署和评测实验中已经定位的报错及解决方法。这里只保留可复用
结论，不写逐日流水账；本机环境、Git/GitHub、文档编辑和一般协作事项不得写入。

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

## Entries

<!-- 按模板在此处下方插入条目，最新条目在最上方。 -->

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
