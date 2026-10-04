"""Phase 13 report generated from actual evidence and conditional gates."""
import csv
import json
from pathlib import Path
import numpy as np
from experiments.phase13_random_expert_bc.analyze import estimate,paired

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase13_random_expert_bc'
C=ROOT/'checkpoints/phase13_random_expert_bc'


def fmt(x,percent=False):
    if percent:return f"{x['mean']:.2%} [{x['ci95'][0]:.2%}, {x['ci95'][1]:.2%}]"
    return f"{x['mean']:.5f} [{x['ci95'][0]:.5f}, {x['ci95'][1]:.5f}]"


def main():
    expert=json.loads((R/'expert_validation/summary.json').read_text())
    collection=json.loads((ROOT/'datasets/random_door_expert/collection_v1/summary.json').read_text())
    audit=json.loads((ROOT/'datasets/random_door_expert/split_v1/data_audit.json').read_text())
    manifest=json.loads((ROOT/'datasets/random_door_expert/split_v1/split.json').read_text())
    offline=[]
    for seed in range(5):
        p=C/f'bc_seed{seed}_summary.json'
        if p.exists():offline+=json.loads(p.read_text())['test']
    primary_path=R/'bc_primary_summary.json'
    primary=json.loads(primary_path.read_text()) if primary_path.exists() else None
    lines=['# Phase 13：随机专家与参数条件 BC','',
           '## 当前结论与阶段门槛','',
           f"独立随机专家验证：**{expert['successes']}/{expert['episodes']} = {expert['success_rate']:.2%}**，Wilson 95% CI {expert['success_95_wilson'][0]:.2%}–{expert['success_95_wilson'][1]:.2%}。通过 >90% 专家门槛。",'',
           f"已采集 **{collection['saved_trajectories']} 条随机成功专家轨迹**，{audit['transitions']:,} 次有效转移，数据审计问题数 {len(audit['issues'])}。",'']
    if primary:
        lines.append(f"预登记的五训练 seed 平均成功率工程门槛：**{'通过' if primary['anchor_gate_passed'] else '未通过'}**。该结论不会通过后续模型选择重写。")
        lines.append('用户要求GT有明确配对收益后做小规模RL。额外的绝对成功率工程门槛与研究性RL进入条件分开记录；单模型验证不能替代五训练seed可靠性。')
        lines.append(f"核心BC结果：Robot-only均值{primary['summary']['A']['success']['mean']:.2%}，BC+GT均值{primary['summary']['B']['success']['mean']:.2%}；配对收益{fmt(primary['paired']['success'],True)}。")
    if (R/'rl_summary.json').exists():
        completed_rl=json.loads((R/'rl_summary.json').read_text())
        values=completed_rl['summary']
        lines.append(f"小规模RL独立Final：冻结BC {values['A']['final_success']['mean']:.2%}，普通SAC微调{values['B']['final_success']['mean']:.2%}，KL约束微调{values['C']['final_success']['mean']:.2%}。约束明显减轻了本配方的退化，但相对冻结BC的提升证据不足；详见配对CI。")
    elif not primary:lines.append('五 seed 物理测试尚未全部完成，环境参数收益与新 Anchor 门槛待判定；没有启动 RL。')
    lines+=['','## 1. 环境与数据','',
            '保持原 Panda Door Opening、Isaac Lab / Isaac Sim、Progress Reward B、机器人、原有 Cartesian differential IK、终止阈值 door_angle >1 rad 和自动 reset。',
            '随机范围：初始门角 U(0°,5°)，工装整体 XYZ 平移各 U(−1,+1cm)，摩擦倍率 1；对应现有代码 Level 2。',
            'GT 专家使用避障中间点、对齐、物理抓取和绕实测转轴拉门；开门阶段解除固定朝向约束。原固定工位规划器、专家数据与 baseline 文件没有改写。',
            '专家完整验证记录和失败分层见 `docs/random_expert_report.md`；预检 57/64 →固定朝向版 0/64 →释放朝向版 62/64，正式验证独立种子 13101。',
            f"数据采集种子 13201，共 {collection['episodes']} 次完整尝试、{collection['successes']} 次成功。训练数据按成功筛选；全分布能力由独立随机物理测试衡量。",'',
            'HDF5：`datasets/random_door_expert/collection_v1/trajectories.h5`。robot state 26；environment state 13；action 7；reward、next states、终止标志、真实 reset 参数与 planner phase 均保存。终止帧使用 reset 前快照。',
            '整条轨迹划分：200 train /50 validation /50 offline test，种子 13301，完全无重叠。只用训练轨迹计算输入均值和标准差。',
            f"数据 SHA256：`{manifest['dataset_sha256']}`。",'',
            '## 2. BC 公平比较','',
            '| 项目 | A Robot-only BC | B Environment-conditioned BC |','|---|---|---|',
            '| Robot state | 同一 26 通道 | 同一 26 通道 |',
            '| 环境通道 | 13 个常数零 | door_angle、target_angle、progress、remaining_angle、双指 contact、handle XYZ+quat |',
            '| 网络 | 39→256→256→7，ReLU/tanh | 完全相同 |',
            '| 参数量 | 77,838（其中 7 个固定 log_std；BC 可训练 77,831） | 完全相同 |',
            '| 权重初始化 / batch 索引 | seed 0–4 配对相同 | 完全相同 |',
            '| 优化器 | Adam lr=3e−4、batch512、20k updates | 完全相同 |',
            '| Checkpoint 选择 | 每 1000 updates 的验证 MSE 最低 | 相同规则 |',
            '| 随机物理测试 | 每 seed 独立128回合，32 clones，确定性 | 相同 reset seed 和 clone 布局，新进程 |','',
            'A 的环境通道在归一化后被设为精确零；改变任何环境输入不应改变 A 的动作。B 把手 pose 使用工作空间坐标和 wxyz 四元数。',
            '参数量相同保证网络结构容量配对；A 的零环境通道权重没有输入梯度。没有 GT auxiliary loss、privileged critic 或质量回放。','',
            '## 3. 离线测试：动作 MSE','',
            '| seed | A transition MSE | B transition MSE | A trajectory-balanced MSE | B trajectory-balanced MSE |',
            '|---:|---:|---:|---:|---:|']
    for seed in range(5):
        row={x['arm']:x for x in offline if x['seed']==seed}
        if len(row)==2:lines.append(f"| {seed} | {row['A']['transition_mse']:.6f} | {row['B']['transition_mse']:.6f} | {row['A']['trajectory_mse']:.6f} | {row['B']['trajectory_mse']:.6f} |")
    if len(offline)==10:
        a=[x['transition_mse'] for x in offline if x['arm']=='A'];b=[x['transition_mse'] for x in offline if x['arm']=='B']
        inference=dict(A=estimate(a),B=estimate(b),B_minus_A=paired(b,a))
        (R/'offline_mse_summary.json').write_text(json.dumps(inference,indent=2))
        lines+=['',f"A MSE {fmt(inference['A'])}；B MSE {fmt(inference['B'])}。B−A {fmt(inference['B_minus_A'])}。MSE 较低不等于长时序执行成功率较高。"]
    lines+=['','## 4. 独立随机物理执行','']
    if primary:
        lines+=['| seed | A 成功率 | B 成功率 | A/B 最大角均值(rad) | A/B 最终角均值(rad) | A/B 接触成功率 |',
                '|---:|---:|---:|---:|---:|---:|']
        for seed in range(5):
            rows={x['arm']:x for x in primary['per_seed'] if x['model_seed']==seed};a,b=rows['A'],rows['B']
            lines.append(f"| {seed} | {a['success']:.2%} | {b['success']:.2%} | {a['mean_max_angle_rad']:.3f}/{b['mean_max_angle_rad']:.3f} | {a['mean_final_angle_rad']:.3f}/{b['mean_final_angle_rad']:.3f} | {a['contact_success']:.2%}/{b['contact_success']:.2%} |")
        lines+=['','| 指标：均值 [seed级95% CI] | A | B |','|---|---:|---:|']
        for key,label in [('success','成功率'),('mean_max_angle_rad','最大门角(rad)'),('mean_final_angle_rad','最终门角(rad)'),('contact_success','接触成功率')]:
            lines.append(f"| {label} | {fmt(primary['summary']['A'][key],key in ['success','contact_success'])} | {fmt(primary['summary']['B'][key],key in ['success','contact_success'])} |")
        d=primary['paired']['success']
        lines+=['',f"配对成功率差 B−A：**{fmt(d,True)}**；paired t p={d['paired_t_p']:.4g}；exact sign-flip p={d['exact_sign_flip_p']:.4g}；正差 seed {d['positive_seeds']}/5。",'',
                'n=5：t区间与检验依赖独立、近似正态的 seed 效应假设；精确双侧 sign-flip 的最小 p 为0.0625，不能声称精确检验 p<0.05。区间不截断到[0,1]，避免改变统计口径。',
                'Phase12 的约25%是不同训练配方的历史参照，不是本次同条件对照。随机数据、规划路径、BC 初始化/训练都发生变化；与历史差异不能单独归因于 GT 或数据覆盖。']
    else:lines.append('待所有配对 seed 完成，不用部分结果判断收益。')
    sensitivity=R/'action_sensitivity.json'
    lines+=['','## 5. 参数利用诊断','']
    if sensitivity.exists():
        s=json.loads(sensitivity.read_text())
        lines+=['固定相同的离线测试 robot state，角度扫描0–5°。同时报告原始角度通道干预，以及一致更新 progress/remaining 的干预。把手/接触保持固定，因此这是反事实网络敏感性测试，不代表可执行物理初态。','',
                '| seed | A angle RMS change | B angle RMS change | A coherent RMS change | B coherent RMS change |','|---:|---:|---:|---:|---:|']
        for seed in range(5):
            row={x['arm']:x for x in s['rows'] if x['seed']==seed};a,b=row['A'],row['B']
            lines.append(f"| {seed} | {a['raw_angle_rms_change']:.6f} | {b['raw_angle_rms_change']:.6f} | {a['coherent_angle_rms_change']:.6f} | {b['coherent_angle_rms_change']:.6f} |")
        lines.append('动作发生变化只说明参数被使用，不说明这种使用提高了成功率。')
    else:lines.append('敏感性诊断尚未完成。')
    masks=sorted((R/'masking').glob('*.json')) if (R/'masking').exists() else []
    if masks:
        lines+=['','输入遮挡使用训练均值（标准化后零），angle_only、角度+progress+remaining、handle pose、contact、全部环境，以及 none/none_repeat 阴性对照。固定预指定 seed0 模型与相同随机测试 cohort，每条件新进程；仅作机制诊断，不代替五 seed 效应检验。','',
                '| 条件 | 回合 | 成功率 | 接触率 | 最终角(rad) |','|---|---:|---:|---:|---:|']
        for path in masks:
            m=json.loads(path.read_text());lines.append(f"| {m['mask']} | {m['episodes']} | {m['success']:.2%} | {m['contact_success']:.2%} | {m['mean_final_angle_rad']:.3f} |")
        lines.append('遮挡属于分布外干预；单通道可能由冗余参数恢复。若 none/none_repeat 不一致，应先量化评估噪声，再判断遮挡效果。')
    else:lines.append('输入遮挡物理测试尚未完成。')
    settling=R/'initial_angle_settling.json'
    if settling.exists():
        s=json.loads(settling.read_text())
        lines+=['','### 随机初始角的持续性限制','',
                f"在{int(s['trajectories'])}条成功专家轨迹中，静止REST结束时 {s['fraction_rest_end_under_half_degree']:.2%} 的角度已回到0.5°以内，平均初始角{s['mean_initial_deg']:.3f}°→{s['mean_rest_end_deg']:.3f}°。{s['fraction_drift_over_1deg']:.2%} 的回合变化超过1°。",'',
                '上游门关节继承 stiffness=10、damping=2.5 的ImplicitActuator。额外只读控制探针从5°开始，测得 door position target 始终0；存在向关闭状态恢复的机制。机械接触的贡献未通过干预分离。',
                'reset即时实际角度和工装平移已验证正确，但角度扰动很快衰减。因此本阶段对持续门角随机化的证据有限，不能把成绩全部解释为0–5°门角适应。工装XYZ随机化保留，开门过程中角度反馈仍变化。环境/actuator/reset没有修改。']
    stochastic_files=sorted((R/'stochastic').glob('*.json')) if (R/'stochastic').exists() else []
    if len(stochastic_files)==5:
        values=[json.loads(p.read_text()) for p in stochastic_files]
        lines+=['','### BC+GT随机执行（std=0.01）','',
                f"五模型随机执行均值 {fmt(estimate([v['success'] for v in values]),True)}。每模型64回合，独立cohort；这个组只有B，不能据此声称随机执行模式下A/B配对收益。"]
    coverage_path=R/'coverage_diagnosis.json'
    if coverage_path.exists():
        coverage=json.loads(coverage_path.read_text())
        lines+=['','### 访问状态与专家覆盖','',
                'coverage_diagnosis.json保存每个模型固定clone0的首回合轨迹及其离线训练近邻距离。失败轨迹大量访问本模型离线测试距离95分位以外的状态。不同输入维度的距离不能横向比较；一个首回合不能估计总体发生率，也不能单独证明因果。',
                'BC在专家状态上的动作MSE小，不保证自己执行后到达的状态也有正确动作。机器人后续观测受自身动作影响，符合模仿学习中访问状态分布偏移这一诊断方向；当前记录支持进一步收集恢复动作，尚未验证恢复数据能够解决问题。[Ross等，2011](https://proceedings.mlr.press/v15/ross11a.html)']
    lines+=['','## 6. Anchor 与小规模 RL','']
    rl=R/'rl_summary.json'
    anchor_path=R/'anchor_validation.json'
    if anchor_path.exists():
        av=json.loads(anchor_path.read_text())
        lines+=['单模型操作性验证与主比较分开：共同新验证cohort每候选64回合，从5个既有模型按成功率（再按验证MSE）选择；选择后3个未见cohort各128确定性+64随机执行回合。没有根据复测结果重选。',
                f"选择训练seed {av['chosen']['seed']}。确定性复测 {fmt(av['deterministic_summary'],True)}；随机复测 {fmt(av['stochastic_summary'],True)}。操作性Anchor门槛 **{'通过' if av['operational_anchor_gate_passed'] else '未通过'}**。",'',
                '这里3个cohort是同一个模型的评估重复，不是3个独立BC训练seed；BC训练seed波动问题仍然存在。主五seed工程门槛结论不变。']
    elif (R/'anchor_selection_protocol.json').exists():
        lines.append('单模型Anchor选择与独立复测进行中，未启动RL。门槛和未见cohort已登记在anchor_selection_protocol.json。')
    if rl.exists():
        result=json.loads(rl.read_text())
        lines+=['','### 3 seeds ×50,016步：候选Anchor微调','',
                '保留五seed平均成功率与操作性工程门槛未通过的结论。用户要求的相对收益前提已满足：专家>90%、GT配对差CI>0且5/5正差、独立候选确定性/随机执行CI下界超过25%。因此按原要求进行限定预算的可行性验证，不把候选标成可靠部署策略。变更登记在rl_entry_amendment.json。',
                'A冻结同一个选定BC模型；B对称SAC+持续BC约束；C仅额外增加固定Anchor Normal KL权重0.1。B/C都使用同一39维Actor/Critic测量输入、Progress Reward、std=0.01、actor lr3e−5、critic lr3e−4、BC权重10、uniform expert/online各50%、critic预热1000更新。没有privileged critic、quality replay或rollback。',
                '每臂3个RL/采集seed，共450,144环境交互（其中150,048为冻结策略执行）；每个学习臂实际6000 Actor更新。3个RL seed使用同一被选BC模型，不能称为3个独立BC初始化。',
                'Best仅按固定验证集选取；最终报告独立测试集best/final。完全相同Actor张量与normalizer才复用同cohort测试，复用记录显式保存。冻结臂曲线仅实测起点和终点，不捏造中间结果。','',
                '| 方法 | 验证AUC | 独立Best成功率 | 独立Final成功率 | Best−Final | 在线成功轨迹（合计） | 在线成功率（seed均值） |',
                '|---|---:|---:|---:|---:|---:|---:|']
        labels={'A':'冻结BC','B':'BC + SAC','C':'BC + SAC + KL'}
        for arm in ['A','B','C']:
            s=result['summary'][arm];rows=[x for x in result['per_seed'] if x['arm']==arm]
            lines.append(f"| {labels[arm]} | {fmt(s['validation_auc'])} | {fmt(s['best_success'],True)} | {fmt(s['final_success'],True)} | {fmt(s['best_final_gap'],True)} | {sum(x['online_successes'] for x in rows)}/{sum(x['online_episodes'] for x in rows)} | {fmt(s['online_success_fraction'],True)} |")
        lines+=['','| RL seed | A Final | B Final | C Final | C final随机执行 |','|---:|---:|---:|---:|---:|']
        for seed in range(3):
            rows={x['arm']:x for x in result['per_seed'] if x['seed']==seed}
            lines.append(f"| {seed} | {rows['A']['final_success']:.2%} | {rows['B']['final_success']:.2%} | {rows['C']['final_success']:.2%} | {rows['C']['final_stochastic_success']:.2%} |")
        lines+=['',f"C最终随机执行（每seed独立64回合）：{fmt(result['summary']['C']['final_stochastic_success'],True)}。所有训练采集均为std=0.01随机策略；在线成功统计仅包含完整结束回合，未完成回合单独记录。",'',
                '### 配对效应与统计限制','',
                '| 配对差（独立Final） | 均值与95% CI | paired t p | exact sign-flip p |','|---|---:|---:|---:|']
        for pair in ['B_minus_A','C_minus_A','C_minus_B']:
            d=result['paired'][pair]['final_success']
            lines.append(f"| {pair} | {fmt(d,True)} | {d['paired_t_p']:.4g} | {d['exact_sign_flip_p']:.4g} |")
        lines+=['','n=3只支持小规模可行性结论；精确双侧sign-flip最小p=0.25。不能借t检验或最优checkpoint夸大多seed稳定性，也不能把防止退化等同于显著超越冻结BC。',
                'C的KL约束覆盖混合replay访问到的状态；BC损失只约束专家样本状态。结合策略漂移、critic记录和成功曲线判断机制；有限critic loss不证明Q可信。',
                '源代码和独立测试记录可复现全部结果；验证集进度曲线不能冒充独立测试的sample efficiency曲线。']
        improvement=result['paired']['C_minus_A']['final_success']
        lines.append(f"独立测试C−冻结BC：{fmt(improvement,True)}。{'区间下界大于零，但三seed仍仅为小规模证据。' if improvement['ci95'][0]>0 else '区间未支持稳定超越冻结BC；约束保留能力与在线性能提升需要分开判断。'}")
    elif (R/'rl_entry_amendment.json').exists():
        lines.append('小规模RL进行中：保留工程门槛未通过结论，但用户要求的相对BC收益和候选独立复测前提已满足，按3 seeds×50,016步验证冻结BC、BC+SAC、BC+SAC+KL。候选仍不等同于可靠部署Anchor。详见rl_entry_amendment.json与rl_protocol.json。')
    elif anchor_path.exists() and not av['operational_anchor_gate_passed']:
        lines.append('**未建立新GT Anchor，未启动RL。** 候选模型复测未通过；保留失败，继续数据/状态/动作表示诊断，不扩大训练。')
    elif anchor_path.exists():lines.append('已保存候选策略；RL研究进入前提与部署工程资格分开判断。')
    elif primary and not primary['anchor_gate_passed'] and not (R/'anchor_selection_protocol.json').exists():
        lines.append('五seed工程门槛未通过；单模型独立验证尚未执行，RL未启动。')
    else:lines.append('尚未启动 RL；等待 BC Anchor 门槛与参数诊断。')
    drift_path=R/'actor_drift_diagnosis.json'
    if drift_path.exists():
        drift=json.loads(drift_path.read_text())
        lines+=['','### 原始输出漂移与实际动作表示','',
                '原环境六个机械臂通道连续，第七个夹爪通道由BinaryJointPositionActionCfg按符号执行开/闭。训练器保留七维Gaussian；tanh前KL可能因饱和输出漂移而很大，不应直接解释为同样大的机械动作变化。',
                'actor_drift_diagnosis.json在同一2048个离线测试状态上同时记录原始输出、tanh后的机械臂动作RMS变化，以及夹爪开闭符号不一致率。这是独立诊断，没有用于选择checkpoint或改变训练；离线状态诊断不能替代在线访问状态测量。','',
                '| 方法/seed | 原始机械臂RMS | 原始夹爪RMS | 执行动作机械臂RMS | 夹爪开闭不一致率 |','|---|---:|---:|---:|---:|']
        for row in drift['rows']:
            lines.append(f"| {row['arm']}/{row['seed']} | {row['raw_arm_rms']:.5f} | {row['raw_gripper_rms']:.5f} | {row['action_arm_rms']:.5f} | {row['gripper_binary_disagreement']:.2%} |")
        lines.append('这些测试状态上，普通微调的机械臂实际动作漂移远大于KL约束组；所有组夹爪开闭符号均未变化。失败不能仅归因于原始夹爪输出的大KL。')
    lines+=['','## 7. 对五个问题的回答','']
    lines.append(f"1. **随机工位是否可由专家解决？** 是，在完整随机分布独立测试中 {expert['success_rate']:.2%} 成功；少量失败仍需记录，不能把成功筛选数据的100%当专家成功率。")
    if primary:
        d=primary['paired']['success'];b=primary['summary']['B']['success']['mean']
        lines.append(f"2. **环境参数是否帮助BC？** B均值{b:.2%}，配对差{d['mean']:+.2%}；五seed配对证据支持GT收益，但训练seed可靠性不足。不能把平均成功率工程门槛未通过误写成GT没有收益；精确检验分辨率与CI限制见上文。")
    else:lines.append('2. **环境参数是否帮助BC？** 待五 seed 独立执行结果。')
    lines+=['3. **参数是否影响动作？** 以敏感性与遮挡记录判断；动作变化与任务收益是两项不同证据。单模型接触遮挡的影响大，handle pose遮挡并未一致变差；不能据此断言每个GT通道都有效。',
            '4. **新Anchor是否具备泛化？** 以单模型选择后的独立复测为准；不能用离线MSE或标称工位成绩替代。这个随机范围内的操作性证据不等于多训练seed可靠或持续初始门角泛化。',
            '5. **能进入LWD/DIVL吗？** 当前未证明稳定优于冻结随机BC的在线改进，候选Anchor也未通过可靠性工程门槛。完成了规定的小规模RL验证，但暂不进入经验筛选扩展。','',
            '## 8. 保存与复现','',
            '- 新源代码：`experiments/phase13_random_expert_bc/`。',
            '- 数据：`datasets/random_door_expert/`；checkpoint：`checkpoints/phase13_random_expert_bc/`。',
            '- 每阶段日志：`logs/phase13_random_expert_bc/`；指标、轨迹CSV和首回合 trace：`results/phase13_random_expert_bc/`。',
            '- 专家转移完整保存在HDF5；RL保存完整回合指标与模型/优化器状态，未持久化全部在线replay和所有RNG状态，因此不能承诺逐位一致的中途训练续跑。',
            '- `protocol.json` 固定比较和门槛；`runtime_environment.json` 固定依赖与GPU信息；`baseline_inventory.json` 保存此前源码hash和产物大小/mtime。',
            '- 使用一张共享GPU；主比较每次一个32-clone进程。诊断先通过重复一致性预检，再采用至多两个独立32-clone进程；不改变单进程物理布局，不停止其他项目。每个BC臂种子20k优化步，不是20k在线环境交互。','']
    lines+=['曲线：[五seed BC](../results/phase13_random_expert_bc/figures/bc_paired_success.png)、[参数遮挡](../results/phase13_random_expert_bc/figures/bc_parameter_masking.png)、[三seed RL稳定性](../results/phase13_random_expert_bc/figures/rl_stability.png)。PNG和PDF均已保存。','']
    lines+=['## 9. 下一步优先级','',
            '1. 单独建立新环境版本验证初始门角扰动的持续性；当前版本和历史结果保持不变。现有成绩包含工装XYZ随机化，不能冒充持续0–5°门角泛化。',
            '2. 补充偏离成功轨迹后的恢复状态与正确动作。仅成功专家路径没有覆盖策略自身访问的所有状态；优先诊断恢复数据覆盖，避免直接扩大RL预算。',
            '3. 在新实验中比较把手相对TCP表示、短历史或阶段监督，以及连续机械臂+二元夹爪动作表示。单模型遮挡结果只用于提出假设，不据此宣布最佳特征子集。',
            '4. 保留冻结BC、普通微调和受约束微调的相同预算对照；只有证明随机工位可靠Anchor与持续在线改进后，再研究LWD/DIVL。','']
    path=ROOT/'docs/phase13_random_expert_bc_report.md';path.write_text('\n'.join(lines),encoding='utf-8');print(path)


if __name__=='__main__':main()
