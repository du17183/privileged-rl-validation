# b300-2 Panda Drawer privileged RL：阶段记录

记录日期：2026-09-28。**该实验尚未产生专家轨迹或强化学习结果，不能回答 privileged information 是否显著提高 sample efficiency。**

## 服务器实测

| 项目 | 实测值 |
|---|---|
| SSH 别名 | `b300-2-xiaolong`；实际主机名 `discover2` |
| 工作目录 | `/home/xiaolong/privileged_rl_validation`（`/home/xiaolong` 指向 `/DATA/disk1/home/xiaolong`） |
| OS | Ubuntu 24.04，glibc 2.39 |
| GPU | 8 × NVIDIA B300 SXM6 AC，每卡 275040 MiB |
| NVIDIA 驱动 | 580.82.07 |
| Python | 系统 Python 3.12.3；可用 Python 3.11.15 |
| 空间 | 工作分区剩余约 676 GiB，内存可用约 2.6 TiB（预检时） |
| 仿真平台 | 未安装 Isaac Sim；Isaac Lab v2.3.0 源码已固定到 commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77` |

完整机器记录位于服务器 `results/environment.json`。

## 方案与已上传代码

Isaac Lab v2.3.0 使用 Isaac Sim 5.1.0，已有 Franka Panda 抽屉任务、prismatic drawer joint、相对 IK 动作和开抽屉状态机。任务配置位于服务器 `third_party/IsaacLab`；实验代码位于 `envs/`、`planners/`、`algorithms/`、`train/` 与 `evaluation/`。[Isaac Lab 2.3.0 发布说明](https://github.com/isaac-sim/IsaacLab/releases)、[官方任务目录](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments)、[官方状态机示例](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/run_state_machines.html)。

自定义适配层明确分离机器人观察和工装 GT；抽屉 GT 包含关节位移/速度、把手位姿、左右手指对把手的接触标志。基线 A 的 Actor/Critic 均用机器人观察；B 的 Actor 仅用机器人观察，Critic 加 GT；C 的 Actor/Critic 均加 GT。三组共用任务、奖励、示教、BC 和 SAC 配置。默认计划为 8 个配对 seed、每组每 seed 200k 训练交互步、每 10k 步评估 64 回合。

HDF5 采集器已实现 `observation`、`state`、`action`、`reward`、`next_observation`、`next_state`、`done`、`terminated`、`truncated`；自动 reset 前保存终止状态，避免将下一回合初始状态误作末步的 `next_state`。采集目标是 500 条**成功**示教；实际成功率需在物理仿真中验证。规划器是笛卡尔航点加差分 IK，尚未验证碰撞及物理成功率。

## 已运行的验证

- 服务器现有 PyTorch 环境：A/B/C 的输入路由、BC 梯度步、SAC 更新通过；B Actor 改变 GT 后输出不变。
- 在线/专家混合经验池的环形写入和抽样通过。
- 机器人/GT 张量拆分、接触标志、奖励和成功判定在合成张量上通过。
- 所有 Python 文件与 Bash 启动脚本通过语法检查；统计程序仅用**合成 CSV** 验证，合成数据不在服务器 `results/` 内作为实验结果。

## 尚需完成

Isaac Sim 安装及首次运行要求用户接受 NVIDIA Omniverse 许可协议；确认前未执行安装。之后仍需运行真实环境 smoke test、检查接触力和 reset、调通规划器并生成至少 500 条成功示教，再运行 24 个训练配置并汇总成功率曲线、回报曲线、50/80/90% 阈值和配对统计检验。当前没有可支持“显著提高”的测量数据。[Isaac Lab 安装说明中的许可提示](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)。
