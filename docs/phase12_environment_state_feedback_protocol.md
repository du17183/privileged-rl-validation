# Phase 12 预注册执行协议

## 对照与预算

使用同seed Phase9 C final完整学习状态继续训练，原Phase8 B best冻结Anchor。
Progress Reward、BC=10、uniform expert/online各50%、std≤0.01、学习率与32环境/4update不变。
所有episode从首次reset起采用完整0–5°、XYZ各±1cm随机分布；不启用自适应课程，摩擦不变。

核心A/B/C/D各5seed×300k。B/D额外Anchor系数0.1、0.01、0各5seed×300k；强Anchor=1复用核心。
共50个唯一run，15M新增训练交互。保持每run更新/交互比例，扩大吞吐通过并行run。

## 输入

A保持原26D机器人观测；B新增angle/velocity/target/progress/remaining五个标量；C加双指contact；D加handle XYZ。
Actor和Critic使用完全相同向量，Critic另加action；不存在Critic专用状态。
初始新增列及Adam moments置零，保留旧权重、动作列、target、alpha与优化器状态；初始策略/Q等价已由物理预检确认。

## 评测与统计

每10k在固定完整随机分布与nominal执行validation；Best只据validation选择。
独立Best/Final随机各128episode/seed，条件分桶各64episode/seed；所有反馈组无条件执行各单通道及关联遮罩测试。
动作敏感性使用预设1024个transition索引、固定robot state，比较0°/5°门角动作L2，记录angle-only与关联reset标量干预。
单通道遮罩存在冗余；关联遮罩用于诊断，不改变物理环境、reward，也不等于重训消融。

推断单位为5匹配seed。配对t95 CI、exact sign-flip及预设比较族Holm校正均报告。
n=5双侧exact最小p=0.0625，不能用episode伪重复获得显著性。

## 条件扩展

仅当B/D相对A的Final增益配对t95下界>0、8个候选Holm t-p<0.05、至少4/5 seed正向、均值增益≥5pp、
|Best−Final均值|≤5pp且拒绝更新≤20%时，继续该组与匹配A至500k。
这是模型假设下探索性门槛，不能代替exact检验确认。未满足则保留300k结果，不盲目扩展。

## 文件和资源保护

所有新代码、checkpoint、轨迹和结果写入Phase12独立路径，历史文件已登记size/mtime与重要文件hash。
GPU共享启动曾被自动审批拒绝；当前队列检查实际GPU compute PID，只在无外部计算进程的GPU上新增本项目作业。
用户若明确批准共享，才会调整此资源策略。其他项目的作业不被停止或修改。
