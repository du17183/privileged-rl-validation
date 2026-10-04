# Phase 10：稳定在线 RL 的质量经验利用实验

生成时间：2026-09-30T14:03:52.530353+00:00。35/35 训练、35/35 独立复测全部完成。历史 44546 个科学文件未变更。

## 1. 比较口径与固定基础

所有组从相同 seed 的 **Phase9 C 最终 checkpoint** 继续，恢复 actor、critic、target critic、optimizers 和 alpha。冻结 anchor 仍是原 Phase8 B actor，未重新锚定到 Phase9 C。std≤0.01 同时用于采集、actor loss 和 target-Q。机器人 Actor/Critic 均只读取 26 维机器人观察，GT 仅用于原奖励、保护评估和本阶段完成轨迹质量计算。

Phase9 checkpoint未包含完整replay快照，因此全部组新建在线buffer。此前5001条成功只作为历史指标，不算作本阶段新增成功，也未伪称已恢复旧在线轨迹。REFC保留相同配方与模型/optimizer起点，replay采用同样受控的fresh初始化。

每组五 seed、追加 300k 交互；96 并行环境，batch256，每个向量步12次更新，共37500次更新/seed。持续 BC λ=10、KL权重1、lr3e-4、Progress Reward、Door任务、expert HDF5、reset、机器人配置全部保持。每10k保护评估32 episodes/模式，固定正式评估64 episodes/模式；独立复测更换进程和随机种子。

**不是第一次学会开门的 sample efficiency 实验。** 初始成功率接近天花板；横轴/AUC指追加训练交互，不包含此前学习和专家生成。只有 matched continuation 组间差值能回答本阶段经验利用问题。

|组|SAC replay方式|作用|
|---|---|---|
|REFC|50% expert +50% fresh online，含未完成episode|精确保留Phase9 C配方|
|A|100% fresh online；BC样本仍来自相同expert|用户要求的online replay对照|
|B|相同expert+已完成online池，按transition均匀|Uniform Success Replay|
|D05/D10/D20|B的同一池，加权temperature .5/1/2|隔离质量权重作用|
|E|B的同一池，质量最高quintile+均匀支持，保留所有边界并列|简化experience selection|

C是各组共用的 Progress Quality 诊断模块，没有额外网络或独立训练组。B与REFC/A还存在专家比例、未完成episode资格差异，不能将该对比全部归因于质量；D/E对B才是主要受控比较。

## 2. 完成轨迹质量与可迁移接口

`score = .55×success + .20×final_progress + .15×net_progress + .10×contact_stability`。进度裁剪到[0,1]；contact只在门实际移动超过0.02rad后计算双指接触比例。完整成功/部分成功/失败并非强制三档，保留连续值。

只在episode结束后打分；pending transition从B/D/E排除，保留在REFC/A。权重为`0.1×uniform + 0.9×normalize(length×exp((score−1)/T))`，轨迹内uniform抽transition。E为20%全部uniform+80%最高质量分位，分位边界并列全部保留。长短轨迹的基础质量密度相同，避免误把trajectory uniform与transition uniform混为一谈。

专家 1000 条/271,562 transitions，score全为1.0。这说明该质量定义无法进一步排序成功专家内部，而非证明所有成功轨迹的策略梯度效用完全相同。

保存HDF5 flat transitions与episode索引(start,length,stride)，保留robot、privileged、action、reward、next状态、done和return/value/success/quality。`trajectory_quality/trajectory.py`提供完成轨迹对象。Actor部署仅使用`state["robot"]`，不读取GT。

## 3. 五 seed 主结果

所有success均为实际受限随机采样策略；best/final为独立复测，AUC来自固定正式学习曲线。均值及t95 CI，CI未裁剪，越界表示小样本线性区间的局限。best根据训练验证选择，独立复测可能低于final，gap允许负值。

|组|Success AUC [95%CI]|独立Best [95%CI]|独立Final [95%CI]|Best−Final(pp)|在线成功/完成episode|回退/seed|Final种子SD(pp)|
|---|---|---|---|---|---|---|---|
|REFC Phase9 C 原配方续训|0.9654 [0.9043, 1.0264]|95.31% [88.05, 102.57]|96.88% [89.24, 104.51]|-1.56|4,981/5,031 (99.01%)|3.80|6.15|
|A 仅在线 replay|0.9585 [0.8963, 1.0207]|97.19% [91.98, 102.39]|95.00% [87.31, 102.69]|2.19|5,002/5,043 (99.19%)|4.00|6.19|
|B Uniform Success Replay|0.9580 [0.8936, 1.0225]|98.12% [92.92, 103.33]|95.00% [84.08, 105.92]|3.12|4,969/5,026 (98.87%)|3.00|8.80|
|D05 Quality Weighted T=0.5|0.9659 [0.9049, 1.0269]|95.94% [84.66, 107.22]|97.19% [92.98, 101.39]|-1.25|5,003/5,048 (99.11%)|3.00|3.39|
|D10 Quality Weighted T=1.0|0.9589 [0.8918, 1.0260]|99.06% [97.33, 100.80]|98.75% [96.22, 101.28]|0.31|4,989/5,042 (98.95%)|2.80|2.04|
|D20 Quality Weighted T=2.0|0.9608 [0.8991, 1.0225]|98.75% [96.22, 101.28]|97.19% [94.66, 99.72]|1.56|4,989/5,040 (98.99%)|2.20|2.04|
|E LWD-style selection|0.9625 [0.8975, 1.0274]|98.44% [96.50, 100.38]|97.19% [91.30, 103.07]|1.25|4,967/5,020 (98.94%)|3.20|4.74|

![Success and return](../results/phase10_quality_lwd/figures/success_reward_curves.png)

![AUC and endpoints](../results/phase10_quality_lwd/figures/auc_endpoints.png)

### 配对检验

|比较|AUC差值 [95%CI]|Final差值(pp) [95%CI]|AUC exact p|Final exact p|
|---|---|---|---|---|
|A-REFC|-0.0068 [-0.0141, 0.0004]|-1.88% [-9.56, 5.81]|0.1250|0.8750|
|B-REFC|-0.0073 [-0.0136, -0.0011]|-1.88% [-6.08, 2.33]|0.1250|0.5000|
|D05-B|0.0079 [0.0009, 0.0148]|2.19% [-5.02, 9.39]|0.1250|1.0000|
|D10-B|0.0009 [-0.0036, 0.0054]|3.75% [-4.99, 12.49]|0.6250|0.5000|
|D20-B|0.0028 [-0.0055, 0.0112]|2.19% [-8.13, 12.51]|0.3750|0.8125|
|E-B|0.0044 [-0.0006, 0.0095]|2.19% [-3.89, 8.26]|0.1250|0.5000|

五配对seed的双侧exact sign-flip最小p=.0625，不能声称p<.05。三个temperature的Holm校正见paired_tests.json；D10是预定主要温度，未从测试集选最优温度再做未校正结论。

### 冻结初始策略对照

Phase9 C源策略的独立受限随机成功率：96.25% [91.19, 101.31]。

|组|Final−冻结初始(pp) [95%CI]|
|---|---|
|REFC|0.62% [-6.96, 8.21]|
|A|-1.25% [-4.98, 2.48]|
|B|-1.25% [-12.17, 9.67]|
|D05|0.94% [-5.87, 7.74]|
|D10|2.50% [-2.74, 7.74]|
|D20|0.94% [-2.00, 3.88]|
|E|0.94% [-5.44, 7.31]|

## 4. 实际经验采样分布与质量饱和

这里统计实际37500×256次SAC抽样/seed，而非buffer名义配置。采样当时未完成的episode标为pending，未事后追溯改成成功。BC额外expert样本量相同，不并入SAC replay分母。

|组|Expert抽样|已完成Online成功|已完成Online失败|Pending|最高质量在线比例|选中轨迹比例|最大质量密度倍率|最终质量加权TV|
|---|---|---|---|---|---|---|---|---|
|REFC|50.00%|39.43%|0.81%|9.75%|98.44%|100.00%|1.000|0.00000|
|A|0.00%|79.35%|1.21%|19.43%|98.64%|100.00%|1.000|0.00000|
|B|69.83%|29.50%|0.67%|0.00%|98.46%|100.00%|1.000|0.00000|
|D05|70.12%|29.75%|0.14%|0.00%|98.56%|100.00%|1.008|0.00751|
|D10|70.07%|29.64%|0.30%|0.00%|98.51%|100.00%|1.007|0.00637|
|D20|69.97%|29.61%|0.42%|0.00%|98.45%|100.00%|1.004|0.00377|
|E|70.21%|29.66%|0.13%|0.00%|98.46%|99.23%|1.012|0.01109|

![Sampling distribution](../results/phase10_quality_lwd/figures/sampling_distribution.png)

质量全为1.0的并列成功经验，E会全部保留；不会随机保留20%再称为质量筛选。若大部分成功经验均饱和，实际筛选或权重改变很小。无收益只能说明本评分与当前成功率/数据分布的受控结果，不能推广为任何quality utilization都无效。

TV是同一最终完成轨迹池中，加权/筛选分布对transition-uniform的total variation，仅衡量本质量规则改变了多少采样质量；REFC/A未实施质量干预，TV记0，其固定source quotas由实际抽样表体现。

## 5. Quality与SAC Q的可靠性

completed quality直接包含success和最终门进度，因此其success ranking AUC会存在标签同义，不能与Phase5固定前缀预测AUC=.888直接等同。它是工装完成轨迹的可验证物理评价，不是提前预测器。

在独立final随机复测内额外捕获64条完整轨迹/seed，不增加rollout。同一冻结Actor和alpha，记录初始Q、普通reward MC，以及从下一动作起加入折扣条件期望熵的soft MC代理；熵用GH12及训练原有epsilon约定计算，避免tanh饱和动作无法反求logp。SAC Q应与soft return对照。在线采集中的Q随更新变化，只作为非平稳描述。

|组|Quality对reward MC Spearman(有效seed)|Q对reward MC Spearman(有效seed)|Q对soft MC Spearman(有效seed)|
|---|---|---|---|
|REFC|0.317 (3/5)|-0.007 (5/5)|-0.007 (5/5)|
|A|0.303 (4/5)|-0.039 (5/5)|-0.039 (5/5)|
|B|0.259 (3/5)|0.093 (5/5)|0.093 (5/5)|
|D05|0.206 (4/5)|0.088 (5/5)|0.088 (5/5)|
|D10|0.065 (4/5)|0.030 (5/5)|0.030 (5/5)|
|D20|0.221 (5/5)|0.041 (5/5)|0.041 (5/5)|
|E|0.169 (4/5)|0.054 (5/5)|0.054 (5/5)|

|组|初始Q均值 [seed t95CI]|Soft MC代理均值 [seed t95CI]|Q−softMC偏差 [seed t95CI]|双Q分歧均值|
|---|---|---|---|---|
|REFC|4.8248 [4.2783, 5.3714]|5.5302 [5.0096, 6.0508]|-0.7054 [-1.4139, 0.0032]|0.0097|
|A|4.8677 [4.2057, 5.5298]|5.4830 [4.9834, 5.9825]|-0.6152 [-1.0469, -0.1835]|0.0141|
|B|4.7034 [4.0596, 5.3472]|5.4850 [4.9746, 5.9954]|-0.7816 [-1.6213, 0.0582]|0.0132|
|D05|4.7939 [4.1495, 5.4382]|5.5987 [5.0420, 6.1553]|-0.8048 [-1.4473, -0.1623]|0.0194|
|D10|4.7551 [4.1024, 5.4079]|5.7297 [5.0773, 6.3821]|-0.9746 [-1.5529, -0.3963]|0.0178|
|D20|4.7137 [3.9845, 5.4429]|5.5718 [5.0252, 6.1183]|-0.8580 [-1.5622, -0.1539]|0.0102|
|E|4.7857 [4.0483, 5.5231]|5.6208 [5.1309, 6.1107]|-0.8351 [-1.5947, -0.0755]|0.0144|

这是相关性诊断，不能用正相关证明梯度收益，也不能用reward-only MC与soft Q差异直接证明Q错误。Q预测的是策略下期望，单条MC含未来随机性；低单轨迹相关不自动等于期望估计失准。Quality是否更适合经验利用最终由D/E对B的配对改进判定。全成功样本缺少失败对照，相关性检验能力有限。

## 6. 独立轻量泛化测试

训练环境完全未改。仅评估进程覆盖合法初始+2.5/+5°、cabinet/fixture y±1cm以及柜体碰撞材质静/动摩擦×0.9/1.1，验证真实物理值。把手变化通过fixture平移实现，未改单独把手几何。原铰链下限为0、标称起点0，−5°不可合法测试；没有裁剪后伪称完成±5°。

REFC/B/D10/E预定主要比较组测试best和final，其他温度与A只做nominal，避免根据结果挑温度补测而扩大规模。

|组|checkpoint|标称随机|+2.5°|+5°|fixture −1cm|fixture +1cm|摩擦×0.9|摩擦×1.1|
|---|---|---|---|---|---|---|---|---|
|REFC|best|95.31%|48.75%|0.00%|16.56%|90.62%|95.62%|99.06%|
|REFC|final|96.88%|52.19%|0.00%|19.06%|95.00%|95.94%|99.06%|
|B|best|98.12%|46.56%|0.00%|17.81%|93.12%|96.25%|98.44%|
|B|final|95.00%|51.25%|0.00%|18.44%|91.88%|96.56%|97.19%|
|D10|best|99.06%|47.81%|0.00%|15.00%|91.25%|95.31%|98.44%|
|D10|final|98.75%|51.88%|0.00%|15.62%|93.75%|94.38%|98.12%|
|E|best|98.44%|48.75%|0.00%|14.69%|95.31%|97.19%|98.12%|
|E|final|97.19%|52.50%|0.00%|15.31%|97.81%|95.94%|97.19%|

### 初始观察对变化的可区分性

|条件|robot观察最大变化(五seed平均)|GT最大变化(五seed平均)|
|---|---|---|
|nominal|0.00000000|0.00000000|
|angle_2p5|0.00000000|0.04363323|
|angle_5|0.00000000|0.08726646|
|handle_y_minus_1cm|0.00000000|0.01000023|
|handle_y_plus_1cm|0.00000000|0.01000023|
|friction_0p9|0.00000000|0.00000000|
|friction_1p1|0.00000000|0.00000000|

![Generalization](../results/phase10_quality_lwd/figures/generalization.png)

泛化失败不等于优化器无法完成标称任务。Actor输入缺少视觉、门初态和把手位置反馈；独立记录初态robot observation和GT相对nominal的变化，检查输入是否能区分变化。输入相同而GT变化只能证明初始观察混叠，不能证明之后完全无法通过机器人接触反馈恢复。本阶段未受控增加可部署观察来隔离原因，不能将所有失败直接归因于观测不足。

## 7. 结论与进入完整LWD/DIVL条件

1. **质量经验是否带来增益？** 预定D10对Uniform B的AUC差 0.0009 [-0.0036, 0.0054]，final差 3.75% [-4.99, 12.49]；E的AUC差 0.0044 [-0.0006, 0.0095]，final差 2.19% [-3.89, 8.26]。若区间跨0，不能认定稳定优于Uniform。

2. **环境进度能否评价经验？** 可以对完成轨迹作物理一致评价；当前score存在成功饱和，尚不能排序成功经验对策略梯度的实际效用。

3. **Quality是否比Q可靠？** 对已发生的物理结果有直接可核验依据；对预测未来回报、提前筛选、产生梯度增益则未由标签排序本身证明。具体MC/Q诊断见第5节。

4. **收益是速度还是最终成功率？** 用D/E对B的AUC差与独立final差分别回答；不能将高起点success误称为首次学习速度提升。上表与paired_tests.json给出两种效果及不确定性。

5. **完整LWD/DIVL条件：**

独立final点估计最高：D10，98.75%。尚无通过全部条件的质量经验利用方案；保持Phase9 C作为默认基础，不能用最高点估计替代配对统计与泛化门槛。

- D10：尚未满足。quality_auc_gain_ci_positive=False；final_no_material_drop=True；continual_online_success_each_seed=True；final_mean_ge_90=True；gap_mean_le_05=True；rejections_each_le_20pct=True；perturbations_mean_ge_80=False。

- E：尚未满足。quality_auc_gain_ci_positive=False；final_no_material_drop=True；continual_online_success_each_seed=True；final_mean_ge_90=True；gap_mean_le_05=True；rejections_each_le_20pct=True；perturbations_mean_ge_80=False。

本阶段只实现completed trajectory scoring、bounded weighted replay和tie-aware selection。未实现完整LWD/DIVL、QAM或VLA。下一阶段应根据上面的实际瓶颈决定：质量饱和时研究可区分成功轨迹的事先定义质量因素；泛化不足时单独验证可部署观察或可重复初态反馈；不在本报告中修改现有任务后追求漂亮数值。

## 8. 历史方法背景（非本阶段配对比较）

|历史方法|Success AUC|独立Final|口径|
|---|---|---|---|
|Phase2 普通SAC|0.183|45.3%|原奖励，500k，确定性|
|Phase2 Privileged Critic|0.089|0%|原奖励，500k，确定性|
|Phase6 持续BC约束B3|0.376|31.9%|原奖励，500k，确定性|
|Phase8 Progress Reward B|见Phase8报告|95.94%|进度奖励，确定性|
|Phase9 C|0.9584|97.50%|原Phase8 checkpoint起点，300k，随机std≤.01|

历史数值用于理解演进，不与本阶段追加300k的AUC直接做因果或显著性比较。纯BC短预训练约6%与含持续BC、expert混合、保护的SAC不构成同更新量控制，不能把差异全部归因于SAC。

## 9. 资源、交互成本与交付

正式追加training interactions：10,500,000（35×300k）；保护及正式学习曲线评估：85,530,656；独立复测：18,421,248（630测试，40,320 episodes）。pilot训练46,080，pilot评估510,880。

独立pilot接口复测：256 episodes、85,856 interactions，单独计入开销、不混入正式五seed推断。

评估交互是实际物理仿真交互，不能隐去后称真实机器人总样本效率。对真实工装部署应降低/重新设计验证成本，再验证一致性。eight B300共享现有RLinf负载：开始每卡2 learner/evaluator pairs、1536训练env，稳定后按吞吐偏好提高至每卡最多3 pairs、2304训练env；总35组及各组学习参数不变，没有停止其他项目任务。独立复测每卡最多2进程。

- CSV、配对结果、95%CI、抽样JSONL、图：`results/phase10_quality_lwd/`。

- 完整SAC checkpoints：`checkpoints/phase10_quality_lwd/`；TensorBoard：`logs/phase10_quality_lwd/`。

- 每条新在线轨迹的数据及完成episode索引：`datasets/phase10_quality_lwd/`。

- 来源hash及固定协议：`results/phase10_quality_lwd/preflight.json`；历史保护：`baseline_preservation.json`。

复现：先`source configs/runtime_env.sh`；使用`python -m experiments.phase10_quality_lwd.audit`检查接口，`train.py --arm ARM --seed SEED --device cuda:0`运行新输出目录。已有结果目录禁止覆盖。分析与报告命令只在35训练及独立复测全部完成后运行。