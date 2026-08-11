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

### 2026-08-11 · [MSMU/RoboBrain2.5] 通用 Auto 类可导入但不能解析 Qwen3-VL checkpoint
- 场景：SOTA supplement 双 lane 的 NV/MT stage-1 canary，固定 RoboBrain2.5 权重。
- 报错：`KeyError: 'qwen3_vl'`，随后 Transformers 报 checkpoint 架构不受支持。
- 原因：旧预检只验证 Auto 类可导入；复用的 4.55.2 环境不含 Qwen3-VL config mapping。
- 处理：RoboBrain 复用现有 `vlmeval_qwen3vl` 环境，并让 `--check` 离线解析锁定 NV config、强制
  `model_type=qwen3_vl`；失败尝试没有生成 prediction/journal。
- 验证：checkpoint 级环境探针与回归通过；双 lane canary/full 状态以本次重跑产物为准。

### 2026-08-07 · [Q-Spatial/scoring] smoke8 产物被误纳入正式候选
- 场景：对 20 条 full-271 结果运行目录驱动 v2 评分预检。
- 报错：`num_candidates=50, num_pending=20`，30 条 `test_artifacts*/smoke8` 被列为 invalid。
- 原因：发现器只排除了旧 `test_runs/` 命名，未排除当前 test gate 目录及其 stale 归档。
- 处理：同时排除 `test_artifacts/` 与 `test_artifacts.stale-*`，并增加真实目录命名回归。
- 验证：服务器预检为 20 candidates/0 invalid；评分后 20 complete、publication gates 与报告 20/21 通过。

### 2026-08-07 · [SPBench-SI/scheduler] server env 覆盖逐轨 GPU 与上下文
- 场景：双卡 test 调度切换专用模型与 LLaVA-NeXT profile。
- 报错：专用模型错误占用 GPU 0；LLaVA 仍收到全局 `max_model_len=32768`。
- 原因：子脚本再次 source server env，覆盖调度器传入的 `CUDA_VISIBLE_DEVICES` 与逐轨 4096 上限。
- 处理：`_env.sh` 在 source 前保存并在其后恢复已显式传入的逐 job 值。
- 验证：环境覆盖回归通过；20 条非 78B 轨的当前绑定 test gate 全部 PASS。

### 2026-08-07 · [SPBench-SI/SpatialLadder] 通用环境缺少 FlashAttention/qwen-vl-utils 组合
- 场景：SpatialLadder 官方 BF16/FlashAttention2 smoke8，单卡 A800。
- 报错：既有通用推理环境不能同时导入锁定 runner 所需依赖。
- 原因：服务器上没有满足该官方组合的完整现成环境。
- 处理：在 `/media/datasets/lihaoran/envs/` 建立复用 torch/flash-attn 的隔离 overlay，仅补 qwen-vl-utils。
- 验证：旧 v1 batch gate 因遗漏官方 left padding 作废；left-padded v2 已通过异长 canary、smoke8、
  full-1009 validator 与 publication gates，1,009 条 generation metadata 均记录 left padding。

### 2026-08-07 · [SPBench-SI/scheduler] 端口 connect 探针误判可复用
- 场景：vLLM 换模清理阶段等待监听端口释放。
- 报错：旧服务仍监听时清理被记为成功，下一条轨启动后冲突。
- 原因：availability 探针用 connect 失败代表可用，没有验证控制器能 bind。
- 处理：释放门禁改为实际 bind，readiness 单独使用 listener connect，并在返回前记录清理失败。
- 验证：端口、readiness、失败事件回归通过；20 条目标 test gate 全部通过独立重算校验。

### 2026-08-06 · [Q-Spatial/3DThinker] Q-Spatial++ 大图导致视觉 attention OOM
- 场景：3DThinker RGB smoke8 的 Q-Spatial++ 索引 `205/247/250`，单卡 A800 80GB。
- 报错：Qwen2.5-VL SDPA 尝试分配约 230.66 GiB，重试后三条仍缺失。
- 原因：checkpoint processor 未绑定像素上限，原始大图产生不可承受的视觉 token attention。
- 处理：Q-Spatial 私有配置显式绑定 `min_pixels=12544`、`max_pixels=401408`；仍只传同一张 RGB。
- 验证：配置解析/metadata provenance 回归已通过；服务器需重跑 3DThinker red/blue 与 smoke8。

### 2026-08-06 · [Q-Spatial/LLaVA] 全局 32768 超出 checkpoint 上下文上限
- 场景：修复 Qwen3-VL 长图 smoke 后，调度器切换到 LLaVA-NeXT-Yi-34B。
- 报错：vLLM 拒绝 `max_model_len=32768`，checkpoint `max_position_embeddings=4096`。
- 原因：Qwen3-VL 所需的服务上下文被作为所有 vLLM profile 的统一值。
- 处理：调度器对 Yi/Mistral 两条 LLaVA-NeXT 轨显式覆盖 4096；该值继续进入 test binding。
- 验证：回归测试证明两条 LLaVA 使用 4096、Qwen3-VL-32B 保留 32768；服务器需重跑对应 test gate。

### 2026-08-06 · [Q-Spatial/Qwen3-VL] 4096 服务上下文装不下图像 token
- 场景：Qwen3-VL-32B Q-Spatial smoke8，锁定图像处理与 1024 输出预算。
- 报错：`Input length (11954) exceeds model's maximum context length (4096)`。
- 原因：私有服务器配置把 vLLM `max_model_len` 降到 4096，短于图像 token 化后的合法输入。
- 处理：恢复 32768 服务上下文，并把该值加入 test binding，旧 4096 gate 自动失效。
- 验证：配置/绑定回归锁定正整数 32768；服务器需重新通过 Qwen3-VL red/blue 与 smoke8。

### 2026-08-06 · [Q-Spatial/scheduler] vLLM 退出后端口短暂晚于 GPU 释放
- 场景：test-only 调度回收 InternVL vLLM 后立即在同一 lane/端口换下一个模型。
- 报错：GPU/进程组已释放，但下一轨瞬时得到 `port 18101 is occupied`。
- 原因：vLLM 监听 socket 的释放短暂晚于自有进程组退出与 GPU 清空。
- 处理：自有服务停止后有限等待端口可 bind；超时仍 fail closed，绝不接管未知 listener。
- 验证：回归覆盖端口从 occupied 到 available 的等待，启动器原有非接管检查保持不变。

### 2026-08-06 · [Q-Spatial/test gate] binding 更新后旧 smoke journal 阻止重测
- 场景：adapter digest 更新使已完成 test gate 失效，调度器在同一 profile 目录自动重跑 test。
- 报错：`Journal run signature mismatch ... use a separate output directory`。
- 原因：旧 gate 正确失效，但旧 `test_artifacts/smoke8` journal 仍占用固定 test 路径。
- 处理：重测前把旧 artifacts/gate 无损轮换为含旧 binding digest 的 `stale-*` 归档，再建立新 signature。
- 验证：回归证明旧 journal/gate 均保留、新 test 路径为空且不跨 signature 恢复。

### 2026-08-06 · [Q-Spatial/LLaVA] vLLM 拒绝 assistant-prefill 第二阶段
- 场景：LLaVA-NeXT 官方两阶段格式修复的第二次请求，vLLM 0.19 assistant continuation。
- 报错：`continue_final_message and add_generation_prompt are not compatible`。
- 原因：请求启用 continuation 时未显式关闭 vLLM 默认的 generation prompt。
- 处理：第二阶段 payload 固定 `continue_final_message=true` 且 `add_generation_prompt=false`。
- 验证：payload 回归锁定两个互补字段；服务器需重新通过 LLaVA red/blue 与 smoke8 gate。

### 2026-08-06 · [Q-Spatial/scheduler] vLLM 换模等待已退出的 zombie group leader
- 场景：双卡/API test-only 调度完成一个 vLLM profile 后回收自有服务并准备换模。
- 报错：GPU 已释放且 vLLM 为 `<defunct>`，lane 仍等待完整的 stop timeout。
- 原因：进程组存活探针早于 `Popen.wait/poll`，未先回收已退出的直属子进程。
- 处理：owned-group 等待循环主动 `poll()` 回收 group leader，再判断进程组是否仍存活。
- 验证：回归测试证明只向记录的进程组发信号，并在等待循环调用 `poll()`。

### 2026-08-04 · [CV-Bench/vLLM] Yi-34B 长尾请求连续超时导致 full 缺两条
- 场景：Yi-34B TP=2、并发 32、官方 greedy/512 配置下运行 full-2638。
- 报错：`Inference incomplete after retries; missing 2 indices: [491, 501]`。
- 原因：两个长输出超过旧 180 秒客户端超时，服务端生成未取消时即时重试又叠加重复请求。
- 处理：本地 vLLM 独立使用 600 秒超时、零即时重试，并在首轮结束后仅补一次 journal 缺失项。
- 验证：两条首轮超时后在 missing pass 成功；2638 条唯一覆盖、validator 和 publishable inference 通过。

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
