# Phase14：恢复专家 → 配对 BC → 条件 RL

当前正式验证为 **138/160 = 86.25%**，未通过用户要求的 >90% 专家门槛。BC 与 RL 均未运行。

## 已保存证据

- `results/phase14_recovery_bc/recovery_validation/`：全新种子 14101 的未筛选接管验证，成功和失败 HDF、491 episode 记录。
- `datasets/recovery_expert/validation_v1/`：138 成功、22 失败恢复及策略前缀，附 SHA256 manifest。它是验证数据，不是已获资格的 BC 训练集。
- `results/phase14_recovery_bc/baseline_*.json`：冻结 Phase13 候选的三种物理恢复测试，每种 64 个有效扰动。
- `experiments/phase14_recovery_bc/source_archives/`：各开发版本和正式版本源代码，负结果保留。
- `docs/phase14_recovery_bc_report.md`：当前阶段报告；其中尚未执行的 BC/RL 标为待验证。

## 复核现有结果

在服务器项目根目录，加载 `configs/runtime_env.sh` 后执行：

```bash
.venv/bin/python -m experiments.phase14_recovery_bc.gate \
  --summary results/phase14_recovery_bc/recovery_validation/summary.json
.venv/bin/python -m experiments.phase14_recovery_bc.expert_analysis
```

第一条只检查门槛，第二条只审计/汇总，不会启动训练。当前 `prepare` 和 BC 入口都会拒绝未通过的专家验证。

## 专家修复后的新实验

修复自然末端偏离的恢复控制后，使用独立的 run-id，保留本轮证据：

```bash
source configs/runtime_env.sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u -m experiments.phase14_recovery_bc.pipeline \
  --run-id recovery_v2 --stage all
```

流水线会依次做 160 次独立接管验证、人工恢复测试、200 成功恢复与 200 普通成功控制数据、五 seed 六臂 BC 和闭环评估。专家验证或采集分布的恢复率 ≤90% 会停止后续阶段。已有输出不覆盖；运行中源代码发生变化会停止，避免不同专家版本混入同一实验。

各 run-id 使用独立的 results/logs/datasets/checkpoints 子目录。BC 评估后使用：

```bash
.venv/bin/python -m experiments.phase14_recovery_bc.analyze --run-id recovery_v2
```

BC 公平比较：A/B 原数据，C/D 加恢复数据，E/F 加等标签预算普通成功数据；每对只改变 GT 掩码。所有 seed、网络容量、Adam、512 batch、20k 更新、归一化和验证选择规则一致。新普通成功数据使用相同修复专家，控制规划器差异。

流水线不自动启动 RL。只有 D 的收益和独立 Anchor 随机/恢复成功率达到所需 80–90% 后，才能另行执行 3 seed、50k–100k 的 Frozen / KL / Residual 微调。当前没有合格的新 Anchor。
