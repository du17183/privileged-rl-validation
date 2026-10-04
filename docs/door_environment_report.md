# Panda Door 环境报告

## 平台与任务

- 服务器入口：`b300-2`；实际系统报告主机名 `discover2`，Ubuntu 24.04.2，内核 6.9.12。
- 8 × NVIDIA B300 SXM6 AC（每卡 275040 MiB），驱动 580.82.07。
- Isaac Sim 5.1.0.0；Isaac Lab 源码版本 v2.3.0，commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`（安装包元数据 0.47.2）；Python 3.11.15、PyTorch 2.7.0+cu128、NumPy 1.26.0、h5py 3.16.0、TensorBoard 2.21.0。
- 每个仿真进程先 `source configs/runtime_env.sh`，加载项目本地 NVRTC 12.9 / libGLU 运行时适配。`experiments/door/launch_full.sh` 把同一波的 8 个训练进程各分配到一张 GPU。

本阶段使用 Franka Panda、6 维相对末端差分 IK 加 1 维二值夹爪动作，60 Hz。独立的 `door_env/door.py` 以 Isaac Lab 的 Panda/Sektion 场景为基础，仅用于 Door，不更改任何 Drawer 文件或结果。右门关节 `door_right_joint` 为 hinge articulation，位置范围 0–1.57 rad。单回合最长 10 s / 600 步，门角 >1.0 rad 为成功。

## 把手与物理状态

Sektion 原有小旋钮在 Panda 当前工位的夹爪姿态下难以保持接触。`door_env/author_asset.py` 生成独立 `assets/panda_door_cabinet.usd`：引用原 Sektion articulation，在右门旋钮刚体下增加尺寸 0.12 × 0.065 × 0.025 m 的可视、可碰撞抓握条。原始铰链和关节限位不变。门把手中心用旋钮刚体位姿与局部偏置 `(0.13, -0.35, 0.185)` m 计算，因此开门时随铰链运动。Panda 双指对 `door_right_nob_link` 的过滤接触传感器提供接触位。资产与门任务独立于抽屉资产；向真实门迁移时应按真实把手形状重新标定抓握条几何与接触阈值。

`PandaDoorEnv.get_tool_state()` 是仿真工装接口：门角 1、角速度 1、把手位置 3、把手四元数 4、左右接触位 2，共 11 维。接触阈值 0.5 N；目前没有把连续接触力作为 GT，未来可扩展。

可部署 Actor 的 `robot_observation()` 为 26 维：Panda 9 个关节位置（包括两指）、9 个关节速度、机器人 TCP 位置 3 和四元数 4、自动 reset 控制器可提供的回合进度 1。A/B/D Actor 的推理函数只调用这个接口；B/D Critic 在训练时追加 11 维工装 GT。C 的 Actor/ Critic 都读取 GT，仅用作理论上限。当前未加入 RGB；机械臂本体观测及固定工位构成第一版可复现对照。

奖励为末端接近把手、双指接触、正向门角速度和开门完成奖励之和。训练与评估的四组均使用完全相同的物理任务和奖励。`PandaDoorEnv` 在自动 reset 前保存 terminal robot/GT/门角，避免 replay 把重置后的初始态错写成 transition 的 next state；reset 后刷新仿真 frame。`results/door/reset_validation.json` 记录强制成功终止后 20/20、超时终止后 20/20 个并行环境均恢复闭门、Panda home 并清除接触位。

四组的标量 reward 与成功标签均由工装 GT 计算，这是实际系统中自动评分所需的共享接口；A 的 Actor/Critic 输入仍不包含 GT，B/D 仅在训练 Critic 追加 GT。标量 reward 的使用与把原始门角/把手位姿送入策略是不同的信息路径。

## 可迁移接口

真实工装替换 `get_tool_state()` 的数据源即可映射角度编码器、角速度估计、把手跟踪及接触检测。Actor 侧只需要机器人关节、TCP 和自动 reset 时钟；真实部署前仍需验证控制时延、关节限位、接触校准、门/把手几何差异与 reset 可靠性。仿真中的 0.5 N 接触阈值和 1.0 rad 成功门槛不可直接当作真实工装的安全阈值。
