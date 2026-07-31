# msmu-a800 GPU burn 启停手册

本手册管理 `msmu-a800` 上 GPU 0/1 的 LLaMA-Factory burn。始终保留现有 tmux session，只暂停或
恢复其中的命令，不执行 `tmux kill-session`。

固定 pane：

| 用途 | tmux target |
|---|---|
| GPU 0 burn | `0:0.0` |
| GPU 1 burn | `1:0.0` |
| burn 自动监控 | `monitor:0.0` |
| monitor watchdog | `watch_dog:0.0` |

## 关闭 burn，释放两张 GPU

先连接服务器：

```bash
ssh msmu-a800
```

必须按“watchdog → monitor → 两个 burn”的顺序停止，避免任务被自动重新拉起：

```bash
tmux send-keys -t watch_dog:0.0 C-c
tmux send-keys -t monitor:0.0 C-c
tmux send-keys -t 0:0.0 C-c
tmux send-keys -t 1:0.0 C-c
```

等待 20～30 秒，让训练进程清理并释放显存，然后检查：

```bash
pgrep -af 'llamafactory_burn|llamafactory_monitor|llamafactory-cli train|run_watchdog'
nvidia-smi
```

预期结果：没有 burn/train/monitor/watchdog 进程，两张卡显存接近空闲、GPU utilization 很低。tmux
pane 应回到 shell prompt；不要删除 session。若进程仍在，先检查对应 pane，不要直接使用
`kill -9`。

## 重新开启 burn

先确认没有重复任务：

```bash
pgrep -af 'llamafactory_burn|llamafactory_monitor|llamafactory-cli train|run_watchdog'
```

立即启动 GPU 0/1 的 burn：

```bash
tmux send-keys -t 0:0.0 \
  "cd /media/damoxing/tangzecong && BURN_STEPS=2000 BURN_BATCH=8 BURN_CUTOFF_LEN=4096 BURN_IMAGE_PIXELS=1572864 BURN_LORA_RANK=128 bash ./llamafactory_burn.sh 0" C-m

tmux send-keys -t 1:0.0 \
  "cd /media/damoxing/tangzecong && BURN_STEPS=2000 BURN_BATCH=8 BURN_CUTOFF_LEN=4096 BURN_IMAGE_PIXELS=1572864 BURN_LORA_RANK=128 bash ./llamafactory_burn.sh 1" C-m
```

恢复 monitor：

```bash
tmux send-keys -t monitor:0.0 \
  "cd /media/damoxing/tangzecong && GPU_LIST='0 1' CHECK_INTERVAL=1800 bash ./llamafactory_monitor_H200_2.sh" C-m
```

恢复最外层 watchdog：

```bash
tmux send-keys -t watch_dog:0.0 \
  "cd /media/damoxing/tangzecong && source ./monitor_watch_dog.sh && run_watchdog llamafactory_monitor_H200_2.sh" C-m
```

最后检查进程和 GPU：

```bash
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} command=#{pane_current_command}'
pgrep -af 'llamafactory_burn|llamafactory_monitor|llamafactory-cli train|run_watchdog'
nvidia-smi
```

预期结果：GPU 0/1 各有一个 burn loop，monitor 和 watchdog 均存在；模型加载完成后两张卡会出现
高显存占用和高利用率。

## 注意事项

- MSMU 推理前保持 burn、monitor、watchdog 全部停止；评测结束后再按上述顺序恢复。
- burn 是循环压力测试，一轮结束会立即开始下一轮；`current` 输出会覆盖，不作为正式训练结果。
- 不要重复发送启动命令。若 pane 中已有 burn，再启动一次会形成重复任务。
- 如果某个 tmux session 意外不存在，应先确认没有遗留进程，再单独恢复该 session；日常启停不要
  删除 session。
