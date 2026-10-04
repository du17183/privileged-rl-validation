# Panda Door 专家数据报告

`door_dataset/planner.py` 根据仿真工装读到的铰链和把手位姿规划末端轨迹：静止、接近、对齐、闭合双指、沿铰链圆弧超前 0.15 rad 拉门。每步由 Isaac Lab 差分 IK 转成 Panda 关节控制，门角由物理接触自然推进；没有直接写门 joint 来伪造专家轨迹。每个 arm action 加独立标准差 0.01 的高斯扰动，规划器每步纠偏。GT 只用于离线专家生成和训练 Critic；A/B/D Actor 不读取 GT。

`door_dataset/door_expert_1000.h5` 经 `door_dataset/validate.py` 审计：

| 指标 | 结果 |
|:--|--:|
| 成功轨迹 / 尝试 | 1000 / 1000 |
| 不重复的 action 轨迹 | 1000 |
| 保存 transition | 271562 |
| 平均长度 / 范围 | 271.562 / 268–276 步 |
| Robot observation / privileged / action | 26 / 11 / 7 维 |
| 回合中曾有左右双指同时接触 | 100% |
| 平均最终门角 | 1.0030 rad |
| 采集仿真总交互（含并行超额步） | 277184 |

每个 `traj_XXXXX` group 包含 `observation`、`robot_state`、`state`、`privileged_state`、`action`、`reward`、`next_observation`、`next_state`、`next_privileged_state`、`done`、`terminated`、`truncated`；同义字段使用 HDF5 hard link，不重复存储。终止步的 `next_state` 取自动 reset 前的 terminal snapshot，全部门角 >1.0 rad，终止标记只在最后一步。根属性记录 seed、环境数量、专家方法、状态字段、接触与成功阈值。逐项统计见 `results/door/dataset_summary.json`。

这 1000 条示教由同一闭环规划器产生，动作扰动使轨迹不完全相同，但几何和初始工位固定。它们适合该固定场景的 BC 初始化与共享离线 replay，不能代表跨门型、把手形状和工位的泛化数据。

`results/door/expert_trajectory.png` 给出首条轨迹的门角、TCP-把手距离和双指接触随时间变化：约 1 s 形成接触，随后持续抓握并在约 4.5 s 达到 1.0 rad 成功门槛。
