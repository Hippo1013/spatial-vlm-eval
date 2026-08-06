# Q-Spatial 遗留小问题

以下事项不改变当前 canonical protocol；在新实验设计获批前不得静默调整。

1. 当前每条 profile 只执行一次。Qwen3-VL 及专用 sampling runner 使用逐请求固定 base seed，API 标记
   provider nondeterministic。若以后改为论文式多 seed 均值，必须新增独立 protocol、输出目录和聚合
   语义，不能覆盖单次轨。
2. 论文叙述曾把 Q-Spatial++ 概括为 101 images；锁定公开 revision 实际是 101 个 QA、87 个不同
   `image_path`/图像。当前合同按发布资产记录两者，不把 QA 数误写为 distinct-image 数。
