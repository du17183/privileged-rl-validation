# Phase 11：测得的环境参数是否提高 Door 泛化

## 研究问题与比较

固定 Phase9 C：Progress Reward、持续 BC（λ=10）、Expert Replay（50%）、冻结 Phase8 B anchor（有效分布 KL λ=1）、全路径 pre-tanh std≤0.01、完整优化器/温度 checkpoint 回退。每个 seed 从对应 Phase9 C `step_300000.pt` 续训；新增参数列和 Adam moments 为零，复制所有旧权重/状态，初始策略和 Q 函数保持不变。

| 组 | Actor 和 Critic 的相同状态向量 | 维度（Critic另加7维action） |
|---|---|---:|
| A | 原机器人观测 | 26 |
| B | A + door angle + target angle + progress | 29 |
| C | B + 左右手指接触二值状态 | 31 |
| D | C + 把手workspace XYZ position | 34 |

姿态朝向与角速度在状态接口中提供，但不默认加入核心实验。真机朝向可靠性未验证。参数来自未来编码器、校准工装位姿与接触传感器的假设接口；本阶段不声称已完成真机传感验证。

## 固定预算

4组 ×5 seeds ×300k 新交互 = 600万；num_envs=32、batch=256、每向量步4次更新、lr=3e-4。每个run 37,500次更新，固定50/50 SAC minibatch，另有相同的expert BC minibatch。专家1000条原始文件只读，Progress Reward重标记与原实现一致。训练曲线起点是继承的成功策略，不能解读为从零学习的样本效率。

## 随机化

Level0 标称；Level1 角度0–2.5°、工装XYZ各±5mm；Level2 角度0–5°、XYZ各±10mm；Level3另加柜体静/动摩擦×U(0.9,1.1)。平移整个柜体，不重新建模把手，不更改机器人控制/原reward/任务目标/episode长度。0°闭门下限不允许负角。每个reset核对真实关节/根位姿/材质读回。每个clone独立随机流，相同seed生成相同顺序的episode参数；策略导致episode长度不同，不保证同一global step的参数相同。

## 条件课程与控制变量

仅用D seed0的10,016步直接Level2 pilot决定是否启用课程：Level2随机成功率<80%或被拒绝则启用。决定在20个正式run启动前冻结。同一个控制器用于所有组：Level0→1→2→3，仅当当前level连续两组不重叠64-episode随机策略评价均≥80%才晋级。晋级只作用于后续episode，不截断正在进行的轨迹。

自适应课程的实际level占比可能随组而变，是状态输入影响学习过程的一条中介路径。必须报告level停留时间/实际episode暴露，固定Level2的独立测试作为主比较；不将其包装为固定训练数据分布的纯网络消融。

## 策略保护与评价

每约10k（32整除后的实际步数）候选在标称和当前课程level下用固定32episodes分别评价deterministic/policy。沿用Phase9的success/progress/max/final angle/regression容差，拒绝即恢复actor/critic/targets/alpha/优化器，保留已发生的交互和replay。Collector在一个更新区间中冻结为已接受策略。

正式曲线每10k在固定Level2和标称分别用64随机episodes，Best依据Level2成功率、标称成功率、进度选取。最终独立Best/Final：新seed、每条件64episodes，另记录确定性执行；不会用独立测试重新选择checkpoint。

条件包括联合Level1/2/3，单独角度0–1/1–2.5/2.5–5°、固定+2.5/+5°，XYZ±5/10mm，历史y±1cm，摩擦×0.9/1.1。联合Level2使用更多独立episodes提供条件切片，按初角与工装L∞位移分箱并报告分母。切片和单独扰动测试分开。

## 统计与诊断

seed为统计单位：均值、样本SD/方差、t95%CI（不裁剪区间），同seed配对差/CI、两侧精确sign-flip检验、预设对比B−A/C−B/D−C/D−A的Holm校正。n=5下精确检验分辨率有限，不以大量episode替代独立seed。

记录逐episode实际初角、初/末把手位姿、摩擦、最终角、success、return、接触稳定性、质量分数；逐10k critic/actor/BC/KL loss、参数列权重、std、拒绝次数。所有经验uniform；质量仅用于诊断异质性。

固定robot observation的角度/handle输入干预，保存action mean；动作变化本身不能证明合理控制，需结合条件成功率。D若相对A有明确增益且标称≥90%，再进行evaluation-only参数遮蔽与±0.5°/±2mm/2%接触误判及一控制步延迟测试。条件不满足时明确暂缓，不编造结果。

## 完成门槛

标称≥90%；+2.5°优于旧51.9%、+5°非零；y−1cm显著改善；多seed稳定、持续在线成功、Best−Final小。历史Phase10只作背景，新同seed冻结Phase9 C复测用于更公平参照。未满足即不进入完整LWD/DIVL，不立即扩展Drawer。
