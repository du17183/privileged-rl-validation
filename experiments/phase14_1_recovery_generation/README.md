# Phase14.1 自动恢复数据

仅数据生成和专家验证，无BC、RL或LWD入口。历史Phase14专家文件保持不变，新专家在`recovery_expert/phase14_1/`。

在服务器项目根目录先执行`source configs/runtime_env.sh`，然后：

```bash
.venv/bin/python -u -m experiments.phase14_1_recovery_generation.run --stage pilot --run-id unique_pilot --gpus 2,3
.venv/bin/python -u -m experiments.phase14_1_recovery_generation.run --stage formal --run-id unique_formal --gpus 2,3
```

每次新run-id，源代码冻结，拒绝混用中途改变的recipe。预检84次有效扰动；正式500次，末端偏离200、其余各100。仅在物理扰动达到条件后计入恢复分母，未触达扰动阶段/未达目标的前缀回合单独保存。策略前缀由相同冻结BC+GT生成，不以规划器替代。

EEF XYZ±5/10/20mm由实际动作反馈控制产生，无机器人瞬移。门角回退显式写门关节位置与速度（用户要求的30°→20°注入）；仅为注入事件，不混入恢复动作标签。接管保持物理状态、episode时钟和原reward。类型/clone配额固定，EEF18桶覆盖，不按专家成功筛掉困难负例。

成功与失败HDF5、attempts.csv、progress.json、summary.json位于`datasets/recovery_expert/phase14_1/<run-id>/<case>/`。输入26维robot、13维environment、11维GT与next字段、action7、reward/done、实际reset参数、规划阶段均保存。策略与注入前缀在独立group，不冒充专家动作。

失败分类为诊断假设，不能把局部IK误差当成全局不可达；原传感器未覆盖全部机器人碰撞，不猜测确认碰撞原因。初角在继承门驱动下会迅速回到关闭，已有泛化结论的限制保持。

正式总体与EEF恢复点估计都>90%、数据审计通过才标记Phase14.2资格；小样本pilot不授权下一阶段。本阶段即便通过也不自动训练BC。

正式运行完成后，`python -m experiments.phase14_1_recovery_generation.deliver`重建正式报告、自然交叉验证诊断和历史保存审计。该命令只有离线分析，无rollout或训练。正式集500有效尝试与独立自然64次接管分别统计；注入数据资格不能冒充自然全状态恢复资格。
