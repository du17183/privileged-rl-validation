# Privileged RL Validation

Franka Panda Drawer / Door Opening实验框架：验证工装提供的ground truth、任务进度和恢复数据如何用于机器人在线学习。

覆盖 **Phase1–14.3**：SAC对照、Privileged Critic、GT辅助表示、轨迹质量、稳定在线更新、Progress Reward、随机工位BC、自动恢复专家、恢复BC与state-only π0.5。

## 新服务器重建

```bash
git clone https://github.com/du17183/privileged-rl-validation.git
cd privileged-rl-validation
python3 scripts/restore_release.py --groups all
python3 scripts/restore_weights.py --groups all
```

完整步骤：[重建与复现](docs/REPRODUCE.md)及[权重背景与加载说明](docs/WEIGHTS.md)。精简权重Release包含80个任务checkpoint和一份共享π0.5基础模型；无需重新训练即可复测所选模型。不要直接在恢复的历史完成标记目录启动新训练，使用`prepare_run.py`创建干净目录。

## 数据与结果在哪里

- Git：所有项目源码、配置、报告、关键CSV/JSON摘要、图表、双环境版本快照、OpenPI实际源码及Transformers模型补丁。
- [Release快照](https://github.com/du17183/privileged-rl-validation/releases/tag/repro-phase14-3-20261004)：所有阶段非权重数据、全部原始指标与日志、逐episode记录、checkpoint元数据、历史源码包和Isaac Lab源码。
- [精简权重Release](https://github.com/du17183/privileged-rl-validation/releases/tag/selected-weights-phase1-14-3-20261004)：81个模型，未压缩14.71GiB；包括π0.5微调增量与同SHA基础模型。每个文件的任务、方案、seed、训练步数和指标来源见[权重索引](reproducibility/weight_catalog.csv)。
- 每卷≤512MiB，下载后核对SHA256再恢复；每个原文件的hash在`reproducibility/original_files.json`。
- 无效实验和历史试跑保留原有命名及报告解释，不能混入正式统计。

## 代码结构

| 目录 | 内容 |
| --- | --- |
| `envs/`, `door_env/`, `randomized_env/`, `environment_state/` | Isaac Lab任务、随机工位、GT接口 |
| `planners/`, `door_dataset/`, `recovery_expert/` | 运动规划、IK、自动专家及恢复数据生成 |
| `algorithms/`, `train/`, `replay/`, `auxiliary_learning/` | SAC、BC、辅助表示、经验利用 |
| `stability/`, `safe_online/`, `progress_rl/` | 动作分布控制、Anchor、进度信号 |
| `bc/`, `pi05/` | 参数条件BC及π0.5 state-only LoRA适配 |
| `experiments/` | 各阶段训练、对照、统计和报告入口 |
| `evaluation/`, `diagnostics/` | 独立评估、恢复压力、动作与value诊断 |
| `configs/`, `scripts/`, `reproducibility/` | 配置、恢复/安装、版本和完整性记录 |
| `docs/`, `results/` | 完整报告及可浏览的结果摘要；全量结果见Release |

## 最新结果：Phase14.3

五seed独立评测的成功率均值（%）；S为随机成功专家，R为恢复专家。四组同为Robot+GT、无RGB。

| 方法 | 随机工位 | 固定工位 | 人工恢复宏平均 | 自然严重偏离 |
| --- | ---: | ---: | ---: | ---: |
| MSE BC + S | 84.22 | 100.00 | 65.23 | 6.25 |
| MSE BC + S+R | 70.31 | 85.94 | 46.25 | 15.00 |
| π0.5 + S | 89.69 | 100.00 | 52.89 | 5.31 |
| π0.5 + S+R | 81.25 | 94.06 | 85.55 | 6.88 |

π0.5+恢复数据改善人工恢复，但随机模型增益未达到校正后的统计显著，自然严重偏离恢复不足，未通过可靠Anchor门槛。没有进入本阶段RL/LWD/DIVL。

[Phase14.3完整报告](docs/phase14_3_pi05_bc_report.md)包含SD、95%CI、配对检验、GT遮挡、动作诊断和成本。π0.5 state-only配方有多个同时变化的因素，不能单独归因为多模态建模；实测单机器人推理延迟也未满足原控制时限。

## 历史报告

- [Drawer](docs/final_report_zh.md)、[Door](docs/door_rl_report.md)
- [GT诊断](docs/privileged_analysis_report.md)、[稳定表示](docs/stable_privileged_report.md)
- [稳定在线RL](docs/phase6_stable_rl_report.md)、[稳定策略](docs/phase7_stable_policy_report.md)
- [Progress Reward](docs/phase8_progress_rl_report.md)、[安全在线闭环](docs/phase9_safe_online_report.md)
- [经验质量](docs/phase10_quality_lwd_report.md)、[随机化](docs/phase11_parameter_generalization_report.md)、[参数反馈](docs/phase12_environment_state_feedback_report.md)
- [随机专家BC](docs/phase13_random_expert_bc_report.md)、[恢复诊断](docs/phase14_recovery_bc_report.md)
- [恢复专家生成](docs/phase14_1_recovery_generation_report.md)、[恢复BC](docs/phase14_2_recovery_bc_report.md)

所有阶段原始证据保留。不同选模、数据和评估协议下的数字不能直接相减当作方法提升。

## 依赖与许可

原平台为Isaac Sim5.1、固定Isaac Lab源码、8×B300。主仿真与π0.5使用两个独立Python环境，版本快照见`reproducibility/`。第三方源码保留上游LICENSE，包括OpenPI和Gemma许可说明；运行Isaac Sim需遵守其许可。
