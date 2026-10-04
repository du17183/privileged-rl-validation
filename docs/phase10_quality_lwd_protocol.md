# Phase 10 执行协议与交付口径

本文件记录已启动实验的协议；最终五 seed 结果将由真实数据生成 `phase10_quality_lwd_report.md`，不会用短运行替代正式结论。

## 固定基础

- 使用同 seed 的 Phase9 C 最终模型，恢复完整 SAC weights、target、optimizers、alpha。
- 保持原 Phase8 B frozen anchor，KL=1、持续 BC λ=10、lr=3e-4。
- pre-tanh std≤0.01 覆盖采集、actor loss、target-Q。
- Actor/Critic：26 维机器人观察；GT只用于原进度奖励、安全评估、完成轨迹质量打分。
- Door、Panda、reward、reset、1000 条专家数据保持。
- Phase9 checkpoint没有replay快照，所有组重新建立在线buffer；历史5001条成功不计入本阶段新增数量。

## 实验矩阵

|组|经验利用|
|---|---|
|REFC|原Phase9 C：50% expert +50% fresh online，允许pending transitions|
|A|仅fresh online SAC replay，保持相同expert BC loss|
|B|expert +已完成online success/failure，transition uniform|
|C诊断|Progress Quality scoring，供B/D/E共用，不另训网络|
|D05|B的相同数据池，quality weighting，T=0.5|
|D10|B的相同数据池，quality weighting，T=1.0，预定主要温度|
|D20|B的相同数据池，quality weighting，T=2.0|
|E|相同池，top-quality quintile selection +uniform支持，边界并列全保留|

D/E对B隔离质量干预。A/REFC对B也改变专家比例和pending资格，单独解释。

## 训练与统计

每组5 seeds、追加300k交互、96环境、batch256、每向量步12 updates，共37500 updates/seed。35组总10.5M训练交互，等于Phase9 aggregate预算。共享八卡，从每卡2组提高到最多3组并发，单组参数保持；其他项目任务继续。

每10k使用独立评估进程进行32 episode/模式保护与64 episode/模式正式曲线评估。Actor采样策略和确定性策略均测。best和final换新process/seed复测64 episodes/模式。

以seed为统计单位，t95 CI、paired exact sign-flip；三个温度比较附Holm校正。五配对seed最小双侧exact p=.0625，不声称p<.05。初始化成功率很高，本阶段比较追加学习AUC/稳定性，不解释为首次学会开门效率。

## 质量定义

`score=.55*success+.20*final_progress+.15*net_progress+.10*contact_stability`。

contact仅在门移动超过0.02rad后计双指接触。只打分完成轨迹。专家1000条在该定义下均为1.0，质量饱和是待分析现象。

加权保留10%uniform支持；E保留20%uniform。记录实际抽样来源、质量histogram、最大密度倍率、最终质量干预对uniform的total variation。GT不是Actor输入。

completed score含success标签；不能用其成功排序AUC证明提前预测。独立final评估记录Q与reward MC及soft MC代理；后者用GH12条件期望熵及原SAC epsilon约定。Q是期望，单轨迹相关性与校准区分。

## 泛化与可观测性

REFC/B/D10/E的best/final测试合法+2.5°/+5°、fixture y±1cm、柜体摩擦×0.9/1.1。原hinge下限0，不能保持任务不变测试−5°。把手变化通过fixture平移实现，未改单独几何。测量真实物理值及初态robot observation/GT变化，检查观察混叠；泛化失败不直接等同RL失败，也不把观察混叠当作唯一因果解释。

## 审计与交付

实际pilot已验证：95/95新增完成episode成功；保存HDF5的索引、真终止GT、回报及quality可复核。短运行只作为接口验证。

历史结果独立保护，主inventory记录44546个文件，另补充71个既有源文件。任务、专家数据、anchor及初始化SHA记录于preflight。

服务器执行队列和后处理持续运行；只有35个匹配训练与35个独立复测全部完成，才生成最终统计、图、报告和review ZIP。

- `results/phase10_quality_lwd/`：CSV、统计、图、source provenance、采样分布、保护审计。
- `datasets/phase10_quality_lwd/`：完整flat transitions与completed trajectory index。
- `checkpoints/phase10_quality_lwd/`：完整SAC checkpoint。
- `logs/phase10_quality_lwd/`：TensorBoard和运行日志。
- `docs/phase10_quality_lwd_report.md`：最终实际五seed报告。

未实现完整LWD/DIVL、QAM、VLA、π0.5；没有改变任务。完成后根据配对增益、持续在线成功、final稳定、泛化门槛决定下一阶段。
