# Phase 15.3：严重自然偏离恢复诊断

正式历史基线为 Phase 15.2 最终交付（300 项 / 30,500 回合 / 五模型 seed）。本目录保存新的单模型 seed 探索性诊断与受控候选验证。是否修复严重恢复、是否形成 Anchor，与实验文件是否闭合是不同验收项。

## 目录与证据

- `diagnostics/`：历史输入 SHA 核验、只读正式漏斗、机制与时间分层、逐步 Handoff、初始状态及动作轨迹配对审计。
- `reacquire/`：独立命名空间中的规划候选，保持原动作接口、关节和碰撞检查、严格 60 mm 接管、真实 15 tick 预测。候选构造 gate 不表示闭环有效。
- `experiments/`：单 seed 原 E 探针、真实新自然开发与未见确认源、四臂开发、600/900/1200 tick 新协议、预声明候选选择与锁定记录。
- `evaluation/`：冻结模型接入与物理重放，只写本阶段输出。恢复动作的前状态和物理执行后的状态分别记录。
- `results/`：真实 CSV/JSON、图表、输入身份快照、互斥失败主类、版本和工程接入归档、审计、交付 SHA。
- 项目根目录 `docs/phase15_3_reacquire_report.md`：十项问题的证据回答与局限。

## 如何解释结果

原正式五 seed 统计、原 25 paired / 补充 45 / 阶段 7 的独立 Holm 族保持。新的状态配对 McNemar 与 Wilson 区间仅为探索性统计；一个模型 seed 无法估计五 seed 方差或泛化 t 区间。

原自然测试仅有实际 10–24 cm 状态；人工 grade 的低距状态不能替代低距自然证据。Phase 15.1 与 15.2 的自然源不同，不能直接相减得到算法增益。

Raw 里程碑可跨尝试、允许阶段绕过，不是连续因果链。Ordered 交集是覆盖描述。F1–F15 是末次尝试互斥主类；预算、模型拒绝与关节证据为可重叠次级标记。几何模型拒绝不代表实测机器人碰撞，有限 IK 未解不代表数学不可达。

600/900/1200 使用相同旧 validation 状态，仅诊断总预算。它不改变历史协议，也不计为同 600 tick 算法增益。共享 batch32 墙钟不除以 32 伪作每回合实测；仿真 tick 与 CPU 墙钟分别报告。

## 选择与学习条件

新自然开发 16 状态用于候选选择；独立 reset 的新自然确认 16 状态不参与调参。确认源 reset 在候选结果出现之前锁定。原正式 test 仅作只读历史诊断，不参与候选选择。

候选按预声明 Normal 交还 / 有效抓取 / Pre-grasp 到达 / 安全位置数量的字典序选取，平局优先现有 GT 直接路由。探索性专家前提为两集各至少 13/16 严格交还且无观察到的关节越界。该门槛是本轮设计判断，不是 Anchor 门槛。含 Recovery 策略桥接与直接 GT 过渡分别记录，不能将桥接动作作为纯 GT 监督。

专家前提未过时不采专家训练标签、不训练 BC 或 π0.5。即使前提通过，也还需要单 seed 专家标签、数据和动作链路验收，再按用户要求先 3 seed 小训练、稳定后才 5 seed；同数据与预算的机器人状态/工装 GT 对照必须实际训练与闭环测量。

本阶段不启动在线 RL、SAC、Residual SAC、LWD、DIVL 或 Phase 16。

## 复核入口

读取 `results/phase15_2_baseline_audit.json` 的历史 SHA、配额、checkpoint 与 validation/test 身份隔离；读取 `results/input_identity_*.json` 的本轮进程完整 argv/PPID 与冻结输入 SHA。运行任何监测脚本前先核验其已记录 SHA。

`experiments/execution_manifest_v1.json` 在开发运行中、候选选择及确认前保存，覆盖执行源码与新来源文件。从该时点起字节冻结；不能冒称启动前独立验签。

全部物理 case 和 CPU 子进程自然退出后，独立分析、Handoff、轨迹审计、统计与报告需基于最终闭合文件刷新；历史 SHA 核验再执行一次，并运行 `diagnostics/final_audit.py`。最终 `results/delivery/` manifest / tar 仍需本地逐项 SHA 验证。任何中途 review 快照都不表示最终交付或算法目标达成。

不要重复启动已有 supervisor；当前状态以对应 `lifecycle.json`、`complete.json`、`failed.json` 与 `/proc` 完整身份为准。资源排队及串行几何计算慢是正常状态，不构成重启、终止或放宽安全/科学门槛的理由。
