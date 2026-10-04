# 在线经验筛选接口与后续扩展

参考 [Learning While Deploying（arXiv:2605.00416）](https://arxiv.org/abs/2605.00416) 的 offline-to-online 框架与 DIVL/QAM 方向。当前阶段只实现可检验的 trajectory value weighted replay；D 是本项目的启发式消融，不是论文方法的复现。`replay/value_weighted_replay.py` 复用 SAC 的七字段 transition，并提供未来扩展的 `sample_experience()`：`{state, action, reward, next_state, value, success}`。其中 state/next_state 明确区分 robot 与 privileged 两部分，保证部署 Actor 接口不混入 GT。

离线源是 `door_dataset/door_expert_1000.h5` 的成功规划示教；在线源是并行 Isaac Lab rollout。A/B/C 从在线 replay 均匀采样，并固定 25% 的离线专家采样份额。D 与 B 使用同一 Actor、Critic、BC、SAC 更新数和专家份额，只改变在线 replay：在每条在线轨迹结束时，B 型 Critic 对该轨迹 transition 的双 Q 取较小值并求均值；按最近 256 条轨迹的 Q 分数做标准化、截断在 ±2，温度 1 的指数权重按轨迹长度分摊给 transition。采样分布为 80% 加权 + 20% 均匀，因此失败和新轨迹仍有机会进入更新。成功标签只用于日志分析，不参与权重计算。

`results/door/replay_D_seed*.csv` 和 TensorBoard 记录有效样本量、最高 10% transition 的采样概率质量、成功 transition 在加权/均匀采样下的概率质量及已完成轨迹数。D−B 的配对比较因此检验这一种基于当前 Critic 价值估计的经验重放策略，不代表对整篇论文或完整 DIVL 的复现。价值在轨迹完成时冻结，Critic 随训练变化后不会重估旧轨迹；这是未来改进点，也避免当前实现引入额外 critic 计算预算。

D 在轨迹之间按价值选择，再在轨迹内部均匀取 transition；因此相同价值的短、长轨迹获得相同轨迹级概率，与 B 的逐 transition 均匀采样并不完全等价。D−B 衡量的是这套**完整重放策略**的效应，不能把差异只归因于 Q 分数。后续应增加「轨迹均匀但不按价值加权」的长度匹配对照，并测试随 Critic 更新而重估旧轨迹价值的影响。

未来接入 DIVL/QAM 可在 `sample_experience()` 后增加独立的数据选择器与质量估计模块，保留 offline/online 来源、轨迹 ID、价值时间戳和选择概率，并进行 importance 校正、重估频率及成功标签泄漏消融。需要同时固定 A/B/D 的总环境交互和网络更新预算，报告额外离线数据成本和额外计算成本。所有新实验写入独立目录，避免与现有 Drawer 和本次 Door 主实验混算。
