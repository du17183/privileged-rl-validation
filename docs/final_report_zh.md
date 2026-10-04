# Panda 抽屉任务：工装 GT 对在线 SAC 样本效率的验证

## 实验范围与环境

本阶段只完成 Drawer。服务器 b300-2 使用 8 张 NVIDIA B300、Isaac Sim 5.1 和 Isaac Lab v2.3.0 的 Franka Panda 与 Sektion 抽屉 articulation。任务控制为 6 维相对末端 IK 加 1 维夹爪，60 Hz；抽屉位移超过 0.30 m 判为成功。A/B 部署 Actor 只读取 26 维机器人侧观察（关节位置与速度、TCP 位姿、reset 控制器时钟）；B 的训练 Critic 另外读取 11 维工装 GT（抽屉位置与速度、把手位姿、左右接触）。C 的 Actor 和 Critic 都读取 GT，仅作上限参照。

环境通过 20/20 次扰动复位检查。成功终止后的自动 reset 恢复了关闭的抽屉和 Panda home，TCP 与把手 frame 相对显式 reset 的误差均为 0。无动作回合在 480 步超时后也自动恢复。A/B Actor 的接口检查确认提供或不提供工装 GT 时动作逐元素相同；C Actor 则要求 GT。详见 `environment_setup.md`、`drawer_task_report.md` 和 `results/actor_gt_isolation.json`。

## 专家数据与训练预算

笛卡尔航点规划和差分 IK 生成 500 条普通示教及 500 条带动作扰动、闭环纠偏的成功示教。共尝试 1,012 回合、消耗 308,256 次仿真交互；合并 HDF5 保存 1,000 条成功轨迹、300,645 个 transition。字段包含机器人 observation、工装 state、action、reward、next observation/state、done、terminated、truncated。完整审计见 `dataset_report_v2.md`。

主实验使用 8 个配对 seed（0–7），A/B/C 各 200,000 次在线训练交互、32 个并行环境。每个运行先进行 3,000 次 BC 和 1,000 次离线 Critic 更新；在线 SAC 每个向量步更新 4 次，batch 256，25% 专家 replay，Actor 的 BC 正则权重恒为 10。每 10,000 训练步评估 64 回合。A/B 每个配对 seed 的 step-0 Actor 权重与评估结果完全一致。24 次运行均通过最终 checkpoint 和日志验收。训练共消耗 4,800,000 次交互；单独计数的周期性评估消耗 14,542,464 次；专家数据采集预算由所有运行共享。配置在 `configs/experiment_primary.json`。

## 结果

| 组别 | 200k 终点成功率，均值 ± SD | 终点接触辅助成功率 | 每 seed 峰值成功率均值 | 成功率 AUC，均值 ± SD |
|:--|--:|--:|--:|--:|
| A：普通 SAC | 0.242 ± 0.449 | 0.242 | 0.754 | 0.233 ± 0.246 |
| B：Privileged Critic SAC | 0.254 ± 0.461 | 0.254 | 0.961 | 0.213 ± 0.149 |
| C：Privileged Policy SAC | 0.484 ± 0.519 | 0.484 | 0.961 | 0.291 ± 0.208 |

首次达到成功率门槛的在线训练步数中位数如下；括号表示达到该门槛的 seed 数。首次达标可能随后退化。

| 组别 | 50% | 80% | 90% |
|:--|--:|--:|--:|
| A | 50,000（6/8） | 50,000（6/8） | 50,000（6/8） |
| B | 50,016（8/8） | 50,016（7/8） | 50,016（7/8） |
| C | 25,008（8/8） | 60,000（7/8） | 60,000（7/8） |

要求从某次评估起直到 200k 终点均保持门槛、且至少跨越两次评估时，50% 门槛仅 A 1/8、B 1/8、C 2/8；80% 同样为 1/8、1/8、2/8。90% 为 A 0/8、B 1/8、C 2/8。具体首次及保持步数，以及计入专家采集和评估后的总交互次数，见 `results/thresholds.csv`。成功率和奖励曲线见 `results/success.png`、`results/reward.png`。

主比较 B−A 的配对成功率 AUC 平均差为 **−0.0208**；配对 bootstrap 95% 区间 **[−0.1926, 0.1459]**，8 对 seed 的精确双侧符号翻转检验 **p=0.7969**。B 虽在 8/8 个 seed 中至少一次达到 50%（A 为 6/8），但没有更高的整体 AUC，终点均值也几乎相同。三个算法均存在明显的阶段性策略退化。

另用不同的 reset seed 901 对全部 24 个 200k 最终 checkpoint 各独立回放 64 回合，额外消耗 670,144 次评估交互。A/B/C 成功率均值分别为 0.246、0.270、0.488，且接触辅助成功率相同；B−A 的配对终点成功率差为 +0.0234，精确双侧检验 p=0.75。此复测没有推翻主实验的“不显著”结论。逐 seed 数据见 `results/heldout_final.csv` 和 `results/heldout_final_summary.json`。

## 结论与边界

**在本次固定抽屉、这套 BC＋SAC 配方和 200k 在线步预算下，没有证据表明训练 Critic 增加工装 GT 能显著提高机器人在线强化学习的样本效率。** 这不是“GT 在所有任务上无效”的证明。C 的 AUC 与终点成功率均值较高，但 C−A 的配对 AUC 差 +0.0577 同样不显著（p=0.6484），且 C 不能作为无 GT 部署方案。策略跨 seed 与训练阶段波动很大，需要先改善在线微调稳定性，再用相同配对协议检验 GT 是否带来可靠收益。

本实验只在仿真中验证，工装与把手位置固定，A/B Actor 没有 RGB。真实系统还需相机或其他定位、时延与接触阈值校准、控制安全限制及 reset 可靠性验证。真实工装接口映射见 `real_fixture_port.md`。Door 与 DIVL 尚未实现，作为后续阶段。

首轮失败协议完整保留在 `results/v1/`，先导和额外 100k 次 DAgger 纠偏交互保留在 `results/pilots/` 与 `datasets/drawer_dagger_v1.h5`，均未混入上述主实验统计。版本、依赖、GPU、数据和代码指纹见 `results/environment_report.md`、`results/dependency_versions.txt`、`results/gpu_inventory.csv`、`results/primary_manifest_sha256.txt`。
