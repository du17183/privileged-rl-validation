# Phase15.1 Re-acquire 与推理时延报告

## 设计与冻结输入

冻结Phase15五个Normal与五个Recovery模型，仅修改重新接近控制器与推理工程/调度。没有BC、RL、SAC、Residual RL、LWD/DIVL或新训练数据。任务、reward、reset、600tick预算与原控制接口保持不变。
先复现旧方法64个自然严重偏离状态；随后只用独立validation选择候选。正式test使用未见reset seed以及独立评测cohort。自然cohort由冻结Normal+Recovery生成，只有源episode实际失败后才保存，不能用人工扰动替代。
人工四级距离每级至少100个独立起点，每种控制器再以五个冻结模型seed匹配评测。graded测试在相同物理handoff强制进入Re-acquire，隔离重新接近能力；自然及完整闭环测试保留实际Router决策。
锁定候选：`{"controller": "B", "schedule": {"normal": 15, "recovery": 15}, "backend": "graph", "selected_by": "validation_only", "numerical_candidate_physical_retention": true, "controller_scores": {"A": [0.2475, 0.86875, 0.90625], "B": [0.30874999999999997, 0.90625, 0.921875], "C": [0.22999999999999998, 0.90625, 0.90625], "D": [0.18125, 0.890625, 0.896875]}, "schedules": {"uniform_5": {"config": {"normal": 5, "recovery": 5}, "validation_rates": [0.859375, 0.7109375, 0.8125]}, "uniform_10": {"config": {"normal": 10, "recovery": 10}, "validation_rates": [0.875, 0.75, 0.921875]}, "uniform_15": {"config": {"normal": 15, "recovery": 15}, "validation_rates": [0.921875, 0.7109375, 0.875]}, "uniform_20": {"config": {"normal": 20, "recovery": 20}, "validation_rates": [0.734375, 0.6484375, 0.65625]}, "uniform_30": {"config": {"normal": 30, "recovery": 30}, "validation_rates": [0.78125, 0.6640625, 0.828125]}, "normal15_recovery5": {"config": {"normal": 15, "recovery": 5}, "validation_rates": [0.921875, 0.78125, 0.875]}, "normal20_recovery5": {"config": {"normal": 20, "recovery": 5}, "validation_rates": [0.734375, 0.765625, 0.65625]}, "normal20_recovery10": {"config": {"normal": 20, "recovery": 10}, "validation_rates": [0.65625, 0.7734375, 0.703125]}}, "rl_updates": 0}`。

## 五seed正式结果

| 方法 | 指标 | 平均成功率 | 95% CI | seed SD |
|---|---|---:|---:|---:|
| A | fixed | 98.75% | [97.13, 100.37]% | 1.31pp |
| A | random | 89.69% | [84.82, 94.56]% | 3.92pp |
| A | ee_offset | 58.20% | [49.53, 66.87]% | 6.98pp |
| A | contact_loss | 90.80% | [88.41, 93.19]% | 1.92pp |
| A | door_regression | 83.20% | [78.20, 88.20]% | 4.02pp |
| A | stagnation | 95.40% | [93.52, 97.28]% | 1.52pp |
| A | natural_severe | 30.87% | [27.25, 34.50]% | 2.92pp |
| A | full_loop | 90.62% | [88.81, 92.44]% | 1.46pp |
| A | grade_05 | 41.80% | [32.79, 50.81]% | 7.26pp |
| A | grade_10 | 55.80% | [40.43, 71.17]% | 12.38pp |
| A | grade_15 | 0.00% | [0.00, 0.00]% | 0.00pp |
| A | grade_20 | 0.00% | [0.00, 0.00]% | 0.00pp |
| A | artificial_macro | 81.90% | [79.46, 84.34]% | 1.97pp |
| B | fixed | 98.75% | [97.13, 100.37]% | 1.31pp |
| B | random | 92.34% | [88.04, 96.65]% | 3.47pp |
| B | ee_offset | 51.80% | [40.37, 63.23]% | 9.20pp |
| B | contact_loss | 93.60% | [92.92, 94.28]% | 0.55pp |
| B | door_regression | 42.80% | [34.13, 51.47]% | 6.98pp |
| B | stagnation | 95.60% | [93.72, 97.48]% | 1.52pp |
| B | natural_severe | 31.00% | [29.39, 32.61]% | 1.30pp |
| B | full_loop | 95.62% | [94.32, 96.93]% | 1.05pp |
| B | grade_05 | 60.40% | [59.72, 61.08]% | 0.55pp |
| B | grade_10 | 7.40% | [6.29, 8.51]% | 0.89pp |
| B | grade_15 | 0.00% | [0.00, 0.00]% | 0.00pp |
| B | grade_20 | 0.00% | [0.00, 0.00]% | 0.00pp |
| B | artificial_macro | 70.95% | [67.01, 74.89]% | 3.17pp |
| C | fixed | 98.75% | [97.13, 100.37]% | 1.31pp |
| C | random | 90.62% | [85.63, 95.62]% | 4.02pp |
| C | ee_offset | 49.40% | [38.38, 60.42]% | 8.88pp |
| C | contact_loss | 93.40% | [92.72, 94.08]% | 0.55pp |
| C | door_regression | 0.00% | [0.00, 0.00]% | 0.00pp |
| C | stagnation | 95.40% | [93.52, 97.28]% | 1.52pp |
| C | natural_severe | 21.25% | [19.51, 22.99]% | 1.40pp |
| C | full_loop | 91.56% | [87.60, 95.53]% | 3.19pp |
| C | grade_05 | 59.20% | [58.64, 59.76]% | 0.45pp |
| C | grade_10 | 1.20% | [0.64, 1.76]% | 0.45pp |
| C | grade_15 | 0.00% | [0.00, 0.00]% | 0.00pp |
| C | grade_20 | 0.00% | [0.00, 0.00]% | 0.00pp |
| C | artificial_macro | 59.55% | [56.68, 62.42]% | 2.31pp |
| D | fixed | 98.75% | [97.13, 100.37]% | 1.31pp |
| D | random | 91.09% | [86.12, 96.07]% | 4.01pp |
| D | ee_offset | 47.60% | [35.23, 59.97]% | 9.96pp |
| D | contact_loss | 92.40% | [89.54, 95.26]% | 2.30pp |
| D | door_regression | 33.40% | [3.62, 63.18]% | 23.99pp |
| D | stagnation | 95.80% | [94.18, 97.42]% | 1.30pp |
| D | natural_severe | 13.00% | [11.15, 14.85]% | 1.49pp |
| D | full_loop | 91.09% | [88.06, 94.13]% | 2.45pp |
| D | grade_05 | 59.20% | [58.64, 59.76]% | 0.45pp |
| D | grade_10 | 1.20% | [0.64, 1.76]% | 0.45pp |
| D | grade_15 | 0.00% | [0.00, 0.00]% | 0.00pp |
| D | grade_20 | 0.00% | [0.00, 0.00]% | 0.00pp |
| D | artificial_macro | 67.30% | [58.58, 76.02]% | 7.03pp |
| E_final | fixed | 99.38% | [98.31, 100.44]% | 0.86pp |
| E_final | random | 88.75% | [82.99, 94.51]% | 4.64pp |
| E_final | ee_offset | 79.00% | [72.55, 85.45]% | 5.20pp |
| E_final | contact_loss | 97.60% | [96.92, 98.28]% | 0.55pp |
| E_final | door_regression | 16.00% | [-16.69, 48.69]% | 26.32pp |
| E_final | stagnation | 97.00% | [97.00, 97.00]% | 0.00pp |
| E_final | natural_severe | 35.25% | [33.46, 37.04]% | 1.44pp |
| E_final | full_loop | 90.62% | [83.60, 97.65]% | 5.66pp |
| E_final | grade_05 | 60.80% | [59.76, 61.84]% | 0.84pp |
| E_final | grade_10 | 8.40% | [7.72, 9.08]% | 0.55pp |
| E_final | grade_15 | 0.00% | [0.00, 0.00]% | 0.00pp |
| E_final | grade_20 | 0.00% | [0.00, 0.00]% | 0.00pp |
| E_final | artificial_macro | 72.40% | [63.56, 81.24]% | 7.12pp |

A=旧GT Re-acquire；B=受检查的撤离/六维pregrasp/有效抓取直线路径；C=B+碰撞感知路径搜索；D=C+迟滞；E_final=validation锁定的控制器与优化推理调度。B对A同时改变多个控制环节，不能单独归因于Safe Retreat。

CI为五冻结模型seed的Student-t区间，表中不截断区间；它不是把全部episode当独立训练seed的区间。配对检验和Holm校正在paired_tests.csv。五seed双侧精确sign-flip最小p为0.0625，不能据此声称p<0.05。

## 阶段拆解

stage_metrics.csv同时记录episode分母、planner尝试次数、撤离/预抓取/重抓取/交还数量及交还后最终成功数量。每次触发可能与每个episode不同，不能混用分母。A没有显式Safe Retreat，其该项为不适用。碰撞模型拒绝与实际物理碰撞必须区分；未安装的臂/腕接触观测不能解释为零碰撞。

## Action chunk与实时性

5/10/15/20/30及不同Normal/Recovery调度均在validation评估。模型训练预测长度为10；15–30是实际更长suffix推理的外推，绝不重复末动作伪造预测。BF16和persistent worker是原始基线已有能力；state-only模型无图像流水线。
精确state-bin文本缓存仅用于相同输入；每次更新state都重新计算prefix KV。CUDA Graph失败记录和编译数值差异记录保留。compile_approx为明确标记的近似BF16融合后端，<=0.01实际动作单位的误差仍必须通过匹配闭环成功率门槛，不能称为位一致推理。

| Seed | 模块 | 预测/执行长度 | Forward ms | RPC ms | 重规划预算 ms | 预算通过 |
|---:|---|---|---:|---:|---:|---|
| 0 | normal | 15/15 | 49.33 | 50.17 | 250.00 | True |
| 0 | recovery | 15/15 | 57.76 | 58.63 | 250.00 | True |
| 1 | normal | 15/15 | 49.40 | 50.28 | 250.00 | True |
| 1 | recovery | 15/15 | 57.82 | 58.69 | 250.00 | True |
| 2 | normal | 15/15 | 49.94 | 50.85 | 250.00 | True |
| 2 | recovery | 15/15 | 58.35 | 59.28 | 250.00 | True |
| 3 | normal | 15/15 | 49.40 | 50.26 | 250.00 | True |
| 3 | recovery | 15/15 | 57.78 | 58.71 | 250.00 | True |
| 4 | normal | 15/15 | 50.39 | 51.43 | 250.00 | True |
| 4 | recovery | 15/15 | 50.50 | 51.53 | 250.00 | True |

吞吐调度允许每GPU四个独立32环境评测进程组，8卡配置上限1024环境；它是配置容量，不是实测最大FPS。原始同卡并发worker时延包含争用，不用于单机器人部署结论。Graph缓存预算每worker48GiB，超预算的新非batch1形状使用等价unroll并记录fallback。
所有并发评测结束后，单机器人时延按batch=1、每GPU一个独占模型worker重新测量，不用8卡吞吐推算单机器人延迟。正式原始结果还保存每episode调用次数、状态阶段ticks、RPC/service时延、显存、deadline misses、端到端控制耗时；平均有效周期=实际活动控制ticks/60÷该episode模型调用次数，Planner阶段不调用π0.5。
仿真等待推理且不推进物理时钟；满足某个chunk nominal预算仍不能等同真实机器人闭环实时通过。路径搜索与碰撞检查的CPU耗时、通信与实时GT采集也需要包含在部署预算中。

## 十三个问题

1. 旧严重偏离主要在起点拒绝；新方法的具体堵点见阶段计数及原始失败分类，不能用IK solver失败证明数学不可达。
2. 撤离提供原方法没有的起点逃逸能力；是否足够由B/C/D撤离计数与最终成功证明。复合消融不能证明单一组件的独立因果效应。
3. 六维pregrasp的效果见到达率、姿态误差与C−B路径搜索比较，未到达不能称问题解决。
4. 交还后是否继续打开由final_success_after_handback/handbacks单独回答，抓住不等于任务成功。
5. 新旧提升值及95%配对CI见paired_tests.csv；负结果完整保留。
6. 自然严重偏离五seed均值=35.25%；>50%：False，≥60%：False。
7. 完整闭环均值=90.62%；≥90%：True。
8. validation选择执行长度={'normal': 15, 'recovery': 15}；不能根据test再挑最佳。
9. 各状态理论周期为执行ticks/60；实际episode平均周期与调用次数需从phase_ticks及pi_calls计算，包含Planner空调用阶段。
10. π0.5 nominal重规划预算通过=True；真实闭环仍需验证，Planner/CPU与输入采集可能继续构成瓶颈。
11. 任务成功门槛全部通过=False。未通过就不能称可靠Randomized Anchor System。
12. 本阶段不自动进入Phase16。先确认阶段门槛及真实端到端安全控制时间预算，未满足条件继续定位恢复/控制问题。
13. 完整LWD/DIVL仍后置；冻结推理/重新抓取评估没有证明在线RL增量收益。

## 复现与证据

结果：phase15_1/results/metrics.csv、per_seed_metrics.csv、stage_metrics.csv、paired_tests.csv、timing.csv、summary.json；逐episode CSV、轨迹输入cohort、推理请求、控制器阶段和故障日志均保留。frozen_verification.json验证历史源码与权重未改。所有性能均来自真实Isaac Lab物理执行，不使用训练loss代替成功率。

## Throughput scheduling revision v9 and sampling provenance

The initial 8 unroll collectors achieved approximately 75-90 environment transitions/s per 32-environment group; the two Graph groups achieved approximately 150-160 transitions/s under the observed concurrent GPU load. These are collection measurements, not single-robot latency benchmarks.

To reduce idle GPU time and serial case collection, the remaining collection was split into 10 natural-source tasks (five frozen model seeds in each split) and 16 artificial/graded case tasks. Each task retains 32 simulator environments; the 26-task configuration capacity is 832 environments, within the 1024-environment scheduler limit. Neither count establishes an experimentally measured maximum FPS.

Eight verified unroll evaluators were interrupted with SIGINT for an engineering migration. Their HDF5 contexts and inference workers closed gracefully. Original files, logs, planned-interruption markers and metadata were retained. Two existing Graph evaluations were not interrupted; one completed naturally before the new supervisor launch. The 227 saved snapshots from the interrupted jobs passed required-field and natural failed-source checks. The resume utility passed an exact-copy fixture and rejected falsely labelled natural examples. The system-Python/native-h5py incompatibility in the migration helper was repaired by using the project Python 3.11 environment and recorded separately.

Natural cohorts use the first 32 qualifying failed-source snapshots per frozen source model. Original snapshots are used first, and missing examples are appended from separately allocated reset streams. Extra original samples remain archived. Artificial and graded quotas remain 32 per validation case and 100 per test case. Changes to source reset/noise streams and case scheduling are recorded in collection_plan_v9.json and cohort manifests; the collection protocol is therefore revised, rather than claimed identical to the initial unroll collection. All controller comparisons use the same finalized validation/test inputs, and selection remains validation-only.

No model was trained, no recovery training dataset was created, and no task/reward/reset/controller action interface or 600-tick episode budget was changed. Historical Phase 1-15 inputs remain covered by frozen verification. Graph numerical and physical retention gates, and exclusive GPU batch-one timing after throughput jobs, still apply.

Revision v10 overlaps the unchanged 20 controller-validation jobs with the remaining independent graded-test collection. All validation inputs were complete and SHA-verified before overlap; the live graded collector was adopted without interruption. Selection still waits for the validation jobs, and no test metric is used for candidate selection. This changes execution order only.


## v11 endpoint/clock repair
The v10 A_seed4 sequential evaluator failed with 31/32 pressure endpoints. Matched sequential replay reproduced it: clone3 reset once during its physical prefix; handoff age39 instead of the charged144 ticks. The Phase15.1-only wrapper restores prefix+one neutral tick age, retaining the original600 tick budget. A fresh32 matched comparison is unchanged;160 fresh and160 sequential corrected starts pass endpoint, geometry and budget audits. All old pressure validation artifacts are archived and re-evaluated; fresh fixed/random/full-loop outputs are SHA-preserved. The interrupted grade20 source retains78 valid starts and appends22 from an independent reset stream. Negative controller results remain results, not technical gate failures. No model was trained.

## 独立审计后的最终结论

本阶段全部正式评测已完成，并通过独立原始结果审计；任务门槛未全部达成。审核包含300组方法×模型seed×测试条件、32,000次物理评测执行，65行五seed聚合指标、25项配对检验及Holm校正、10行独占GPU时延。这里的32,000是评测执行次数；不同模型重复使用共同cohort，并非32,000个独立训练seed或全部独立初态。
40个冻结源码/权重文件重新计算SHA一致；18个共同validation/test cohort、26个源cohort、锁定selection、逐episode CSV/telemetry与统计已重算核验。自然validation/test各160个不同源episode，各由五源模型提供32个；自然源episode失败标签成立，分割身份无交集。全部压力测试终点完整，所有计费prefix+恢复预算≤600tick。审计细节见`phase15_1/results/final_audit.json`。

### 主要方法与正式结果

| 方法 | 固定 | 随机 | 四类人工失败宏均值 | 自然严重偏离 | 完整闭环 |
|---|---:|---:|---:|---:|---:|
| A | 98.75% | 89.69% | 81.90% | 30.87% | 90.62% |
| B | 98.75% | 92.34% | 70.95% | 31.00% | 95.62% |
| C | 98.75% | 90.62% | 59.55% | 21.25% | 91.56% |
| D | 98.75% | 91.09% | 67.30% | 13.00% | 91.09% |
| E_final | 99.38% | 88.75% | 72.40% | 35.25% | 90.62% |

E_final是事前validation锁定的B控制器、15tick执行/15步真实预测、graph后端，正式test后没有重新选择。A/B/C/D对照使用原5tick执行、10步预测；B与E_final的差异同时涉及执行和预测长度，不能分离为单独的chunk因果效应。Graph数值及匹配物理保真检查通过；这里的成功率差异不能归因于“GPU加速改善学习”。
E_final自然恢复35.25%（五模型seed 95% CI 33.46%–37.04%），未达到>50%或≥60%门槛。固定99.38%通过95%点估计门槛；随机88.75%未达到90%；人工四类宏均值72.40%未达到85%；完整闭环90.63%通过90%点估计门槛，但95% CI为83.60%–97.65%，不能据此声称真实成功率已被证明≥90%。总体task gate为False。
B自然恢复31.00%，A为30.875%，配对变化仅+0.125pp，95% CI −2.47至+2.72pp，Holm p=1，不能声称受检查撤离整体显著提高自然恢复。E_final相对A为+4.375pp，配对95% CI +1.27至+7.48pp，原始t p=0.01735，25项Holm p=0.32961，精确sign-flip p=0.0625；不能声称校正后显著。相对同控制器B，E_final自然恢复+4.25pp（CI +3.60至+4.90pp，原始配对t p=0.00005391，Holm p=0.001294，精确p=0.0625）；参数t检验有证据，五seed精确检验不达0.05。两种检验分开报告，不能选择性只展示一个。
E_final相对B随机成功率低3.59pp，完整闭环低5.00pp，两个差异的配对CI包含0、Holm p=1。这是方向性负结果。B随机92.34%、完整闭环95.63%的点估计较高；E_final在自然恢复和独占时延方面有优势，本阶段没有一个方案通过全部门槛。
历史原seed0严重偏离2/64=3.125%仅是旧cohort诊断复现。它与新的独立自然cohort不是匹配对照，不能把3.125%→35.25%写成同测试集的提升。科学方法比较使用共同新test的A/B/C/D/E_final。

### 严重偏离与交还瓶颈

E_final在共同160个自然起点×五模型上执行800次，成功282、失败518。795次进入Re-acquire，撤离544/795=68.43%，到达pregrasp385/795=48.43%，有效重新抓取并交还370/795=46.54%，交还后最终成功278/370=75.14%。另有4次成功未经过Re-acquire交还。

| 失败终点分类 | 次数 | 占518次失败 |
|---|---:|---:|
| 找不到通过原安全检查的单调撤离 | 245 | 47.30% |
| 剩余600tick预算耗尽 | 123 | 23.75% |
| 交还Normal后任务失败 | 92 | 17.76% |
| 找不到通过检查的直线路径 | 57 | 11.00% |
| 没有Re-acquire接管 | 1 | 0.19% |

这些分类按每次失败episode最后一次planner尝试归类，互斥且总数518；它们不是五个独立样本集。没有安全撤离是算法/模型检查拒绝，不能改称实测物理碰撞或证明真实几何不可恢复。时间耗尽可能覆盖多个前序问题，不能单独证明需要放宽600tick。

| 人工偏离 | 到达pregrasp/500 | 有效抓取交还/500 | 最终成功/500 | 交还后成功率 |
|---|---:|---:|---:|---:|
| 5cm | 495 | 420 | 304 | 72.38% |
| 10cm | 500 | 495 | 42 | 8.48% |
| 15cm | 500 | 475 | 0 | 0.00% |
| 20cm | 190 | 0 | 0 | 不适用 |

10cm与15cm档位已经大多重新抓住，但交还后分别仅42/495与0/475完成开门；20cm主要卡在重新抓取。因而不能只看pregrasp或contact称恢复问题解决。交还状态、门角、接触姿态、剩余预算与Normal后续动作需要逐步区分；现有证据不能唯一判定是分布外交还、控制接口、姿态或预算原因。相应阶段条件率与五seed CI另存`stage_conditional_metrics.csv`；原stage_metrics.csv保留每episode/每尝试分母。

### 时延、吞吐与下一步边界

全部仿真进程结束后，五张独占GPU分别执行batch=1测试，每卡一个推理worker。Normal RPC均值50.17–51.43ms，Recovery 51.53–59.28ms；各组P95最高59.60ms，20个热身后计量样本中均没有超过250ms名义预算。每模块每seed先运行3次不计入热时延，冷启动不纳入这些均值。执行15tick=250ms，4Hz重规划；低于名义预算并未证明实时机器人部署成立。
Planner start路径计算的原记录均值约1.95ms、P95约4.85ms；完整planner.action组调用墙钟耗时在每个当时活跃clone累计，含并发争用，自然恢复该累计字段每episode均值约37.95s。它不是独占单机器人planner时延，也不是单次start路径计算时延。RPC、Planner、GT采集、控制周期和硬件安全停机需要独占端到端测量，不能仅凭forward/RPC宣称部署可用。
1024是调度配置容量。主评测完成后的空卡属任务收尾，不能从利用率推断仍需增大吞吐；本阶段没有测得硬件绝对最大FPS。严格torch.compile与compile_approx均有数值拒绝证据，未进入正式候选；使用的是验证通过的graph，保留回退等价unroll证据。
建议下一阶段优先做两类小规模诊断：①对自然起点无法单调撤离的状态检查安全模型与物理接触，并在保持碰撞规则下验证可行出口；②针对10–15cm抓取交还后的失败，固定相同交还状态和原预算，记录Normal动作、接触保持与门角增量，区分交还兼容性和时间预算。随机初态仍未达90%门槛，保留B原调度与E_final分别作为成功率和时延对照。先修复恢复/交还闭环，再考虑新增训练或在线学习；本阶段不自动进入Phase16，更没有进入LWD/DIVL。

本次没有训练任何模型。实验完成标记表示评测与报告齐备，不表示恢复或真实部署目标已达成。
