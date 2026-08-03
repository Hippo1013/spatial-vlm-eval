# msmu-a800 网络代理手册

本手册记录服务器上的显式出站代理。它用于下载依赖、访问外部 API 等需要代理的命令，不改变 MSMU
输入、推理或评分协议。代理工具安装在仓库外，由操作者独立维护：

```text
/media/datasets/lihaoran/tools/mihomo/
```

## 安全与运行边界

- 内核为官方 Mihomo Linux AMD64，当前安装基线是 `v1.19.28`。
- 只监听 `127.0.0.1:7890`，`allow-lan` 关闭；不得改为公网或局域网监听。
- 采用显式 HTTP/SOCKS 代理，不启用 TUN，不修改服务器全局路由。
- 内核运行在独立 tmux session `mihomo-proxy`。它可以跨 SSH 断连保持，但服务器重启后必须重新启动。
- 服务器当前没有用 systemd 托管这个进程；不要把“tmux 存在”单独当作代理可用的证据。
- 订阅 URL 只保存在权限为 `600` 的 `config.yaml` 中；不得写入 Git、命令行、shell history、日志或
  文档。provider 缓存同样保持私有权限。
- 当前 `PROXY` 组用唯一的 `美国02` 标签过滤目标节点。节点清单可能随订阅变化，实际选择与出口国家
  必须现场验证，不能从本文推断。

## 首次配置

在本地终端运行：

```bash
ssh -t msmu-a800 /media/datasets/lihaoran/tools/mihomo/configure.sh
```

在隐藏输入提示中粘贴 HTTPS 订阅 URL 并回车。脚本会原子写入 `config.yaml`、设为 `600`、执行配置
校验、启动 tmux，并输出状态。已经配置时重复执行不会覆盖原订阅。

## 启动、查看与关闭

登录服务器后设置便于复制的路径：

```bash
MIHOMO_ROOT=/media/datasets/lihaoran/tools/mihomo
```

启动或查看状态：

```bash
"$MIHOMO_ROOT/start.sh"
"$MIHOMO_ROOT/status.sh"
```

查看实时 pane；退出查看时按 `Ctrl-b d`，不要按 `Ctrl-c`：

```bash
tmux attach -t mihomo-proxy
```

关闭内核：

```bash
"$MIHOMO_ROOT/stop.sh"
```

必须用 `stop.sh` 关闭，不能直接运行 `tmux kill-session -t mihomo-proxy`。关闭脚本会核对 pid、向内核
转发 `SIGTERM` 并清理 pidfile；直接杀 tmux 可能留下仍占用 `7890` 的孤儿进程。

## 在当前 shell 开关代理

启动内核并不自动代理现有命令。需要代理的每个 shell 单独启用：

```bash
source /media/datasets/lihaoran/tools/mihomo/proxy-on.sh
```

它设置小写和大写的 `http_proxy`、`https_proxy`、`all_proxy`，并让 `127.0.0.1`、`localhost`、`::1`
绕过代理，避免影响本机 vLLM/judge 服务。

只关闭当前 shell 的代理环境变量，不停止后台内核：

```bash
source /media/datasets/lihaoran/tools/mihomo/proxy-off.sh
```

新开的 shell 默认不继承这组设置；不要把订阅 URL 或代理变量写入仓库的 `.env.server`。

## 验证

先确认 tmux、内核和端口三项都为 running/listening：

```bash
/media/datasets/lihaoran/tools/mihomo/status.sh
```

分别验证 HTTP 和 SOCKS 出口：

```bash
curl -fsS --max-time 20 --proxy http://127.0.0.1:7890 \
  -o /dev/null -w '%{http_code}\n' https://api.github.com/rate_limit

curl -fsS --max-time 20 --proxy socks5h://127.0.0.1:7890 \
  -o /dev/null -w '%{http_code}\n' https://api.github.com/rate_limit
```

当前节点策略要求美国出口；只打印国家代码，不把出口 IP 复制到文档或日志：

```bash
curl -fsS --max-time 20 --proxy http://127.0.0.1:7890 \
  https://www.cloudflare.com/cdn-cgi/trace | sed -n 's/^loc=/loc=/p'
```

预期为 `loc=US`。如状态、HTTP/SOCKS 探针或国家代码任一不符，先不要启动依赖代理的正式任务。

## 常见恢复

- `config.yaml` 缺失或仍含占位符：重新运行首次配置命令。
- tmux 已停止或服务器刚重启：运行 `start.sh`，再执行完整验证。
- `7890` 被占用、tmux 状态和内核状态不一致：先运行 `status.sh`，再运行 `stop.sh`；不要按模糊进程名
  批量杀进程。确认三项均停止后再 `start.sh`。
- 启动失败：查看 `/media/datasets/lihaoran/tools/mihomo/logs/mihomo.log`，分享日志前删除订阅地址、节点
  地址、出口 IP 和其他 provider 信息。

## 已验证安装基线

2026-08-03 现场核验：Mihomo `v1.19.28`、Linux AMD64，正式内核文件 SHA-256 为：

```text
08df1464bde7d16936ad086a29b12c435fc6b1cf6554d3b7669433fc13f6fc68
```

升级内核、改变监听地址、端口、节点选择策略或生命周期管理方式时，必须重新完成上述验证并同步本
手册与 `CHANGELOG.md`。运行是否健康始终以服务器现场的 `status.sh` 和出口探针为准。
