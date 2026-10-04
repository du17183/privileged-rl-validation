"""Populate the final report only from validated multi-seed measurements."""
import json
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase9_safe_online'

def metric(value):
    lo,hi=value['ci95'];return f"{value['mean']:.3f} [{lo:.3f}, {hi:.3f}]"

def main():
    export_audit=OUT/'deployment/export_audit.json'
    if not export_audit.exists():
        from experiments.phase9_safe_online.export_policy import main as export_main
        export_main()
    assert len(json.loads(export_audit.read_text()))==70
    summary=json.loads((OUT/'summary.json').read_text())
    tests=json.loads((OUT/'paired_tests.json').read_text())
    held=json.loads((OUT/'heldout_summary.json').read_text())
    readiness=json.loads((OUT/'readiness.json').read_text())
    cost=json.loads((OUT/'interaction_cost.json').read_text())
    speed=json.loads((OUT/'successful_completion_time.json').read_text())
    preservation=json.loads((OUT/'baseline_preservation.json').read_text())
    assert preservation['changed']==[]
    best=max(('A','B','C','D'),key=lambda a:summary[a]['policy_final']['mean'])
    bc_initial=[]
    for seed in range(5):
        with (ROOT/f'results/phase6_stable_online_rl/heldout_P6B1_seed{seed}_initial.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==1 and int(rows[0]['stochastic'])==0 and int(rows[0]['selected_steps'])==0
        bc_initial.append(rows[0])
    bc_successes=sum(round(float(r['heldout_success_rate'])*int(r['episodes'])) for r in bc_initial)
    bc_episodes=sum(int(r['episodes']) for r in bc_initial)
    bc_phase8=[]
    for seed in range(5):
        with (ROOT/f'results/phase8_progress_rl/eval_P8A_seed{seed}.csv').open() as stream:
            row=next(csv.DictReader(stream))
        assert int(row['env_steps'])==0
        bc_phase8.append(float(row['success']))
    historical_heldout=json.loads((ROOT/'results/phase8_progress_rl/heldout_summary.json').read_text())
    phase8_final={arm:next(r['success']['mean'] for r in historical_heldout if r['variant']==arm and r['mode']=='final' and r['noise']==0.0 and r['initial_angle_deg']==0.0 and r['cabinet_dy']==0.0 and r['friction_scale']==1.0) for arm in ('A','B')}
    lines=['# Phase 9：稳定在线改进验证报告','',
        '## 1. 完成范围与结论边界','',
        '完成 7 组 × 5 seeds × 300,000 条新增训练交互。A–D 为主要对照，CANN/D100/D30 为标准差与 replay 比例补充消融。所有运行从同 seed 的 Phase 8 B 最佳完整 SAC checkpoint 继续，没有重新 BC、增加新任务或实现 LWD/DIVL/QAM。', '',
        f"当前独立随机执行终点最高的主要组为 **{best}**：{metric(summary[best]['policy_final'])}。这属于本次数据上的探索性方案选择。",
        '', '高成功率、产生在线成功轨迹、以及真正通过在线更新提高策略是三个不同指标。本报告用相同 std 上限的冻结锚点作为额外反事实，避免把降低执行噪声的效果归因于 RL 学习。', '',
        f'历史BC起点补充：Phase6的纯BC完成、尚未SAC更新时，独立确定性复测为{bc_successes}/{bc_episodes}（{100*bc_successes/bc_episodes:.2f}%），5个seed各64次。逐seed成功率为'+', '.join(f"{100*float(r['heldout_success_rate']):.2f}%" for r in bc_initial)+'。此前约95.9%是持续BC约束配合SAC后的最佳checkpoint，最终仅31.9%；普通SAC接续的最终成功率为0。本阶段从Phase8 B已训练策略继续，未重新训练BC。来源：results/phase6_stable_online_rl/heldout_P6B1_seed{0..4}_initial.csv。', '',
        f'Phase8另一个口径：纯BC起点固定评测为{100*sum(bc_phase8)/5:.2f}%；A（原奖励baseline）最终独立确定性复测为{100*phase8_final["A"]:.2f}%，B（Progress Reward）为{100*phase8_final["B"]:.2f}%。这里的baseline已包含持续BC约束、50%专家replay和checkpoint保护，因此此前“baseline 80%”与“纯BC约6%”对应不同训练时点。Phase8随机在线成功仍为0；不得将确定性80%表述为随机探索成功率。来源：results/phase8_progress_rl/eval_P8A_seed{{0..4}}.csv、heldout_summary.json。', '',
        '纯BC只有3,000次更新，而Phase6 B3的在线阶段还持续做专家BC更新。现有实验没有相同总更新量的纯BC对照，因此无法把6%到95.9%的变化归因于SAC的净贡献。专家动作MSE约0.003也不等同于闭环成功；低BC成功的具体原因（训练量、关键接触动作误差、状态覆盖/可观测性）尚未由干预实验分离。Phase9各组BC约束相同，可比较anchor/std/replay组合；与冻结锚点的比较检验的是整个更新配方，不单独隔离RL与继续BC。', '',
        '## 2. 固定环境与可迁移边界','',
        '- b300-2，8 × NVIDIA B300；Isaac Sim 5.1.0.0，Isaac Lab 源码 v2.3.0（commit 3c6e67bb5c7ada942a6d1884ab69338f57596f77，安装元数据 0.47.2）。',
        '- GPU型号NVIDIA B300 SXM6 AC，每卡275,040 MiB，driver580.82.07，PyTorch CUDA12.8。运行环境与完整依赖快照：results/phase9_safe_online/environment_snapshot.json、environment_pip_freeze.txt。',
        '- Python 3.11.15，PyTorch 2.7.0+cu128，NumPy 1.26.0，h5py 3.16。',
        '- Panda Door，60 Hz 控制，最长 600 步/10 秒；成功条件门角度 >1 rad。原 reset、机器人、摩擦材料与 Phase 8 B signed progress reward 全部保留。',
        '- Actor/Critic 都仅接收既有 26 维机器人观察；门角度、把手、接触等 GT 用于既有 reward、评价和更新验收，没有直接输入 Actor/Critic。',
        '- 既有26维观察具体为9维joint position、9维joint velocity、3维末端位置、4维末端四元数、1维episode归一化时钟。没有视觉；固定初态下成功可能包含按时序重复执行的成分。本阶段未新增或修改这些输入。',
        '- 原专家集 1,000 条成功轨迹 / 271,562 transitions，文件 door_dataset/door_expert_1000.h5。额外冻结锚点数据各 seed 64 条确定性轨迹，只将成功轨迹预载。',
        '- 真实设备对应接口：机器人观察 → 控制动作；工装进度 → reward/更新验收；自动 reset → 再采集。仿真成功率不等同于硬件安全认证。', '',
        f"历史文件保护核验：{preservation['artifacts']:,} 个已有科学产物的大小/修改时间及小型源码哈希均保持不变；详情 results/phase9_safe_online/baseline_preservation.json。", '',
        '## 3. 实验设计','',
        '| 组 | KL 权重 | 执行/训练 std | Critic replay: expert / anchor success / fresh online |',
        '|---|---:|---|---|',
        '| A | 0 | 原始学习分布 | 50 / 0 / 50% |',
        '| B | 1 | 原始学习分布 | 50 / 0 / 50% |',
        '| C | 1 | pre-tanh std ≤0.01 | 50 / 0 / 50% |',
        '| D | 1 | pre-tanh std ≤0.01 | 50 / 25 / 25% |',
        '| CANN | 1 | 前 100k 从 0.1 线性降到 0.01，每 10k 调整 | 50 / 0 / 50% |',
        '| D100 / R1 | 1 | std ≤0.01 | 100 / 0 / 0% |',
        '| D30 / R3 | 1 | std ≤0.01 | 约30 / 35 / 35% |', '',
        'D 对应 R2。非专家份额中，冻结锚点成功与新在线探索各占一半；因此 R2/R3 的“online”池不是全部由新探索产生。所有来源分别计数，预载与评价成功永远不计入 online_successes。每 batch=256，D30 实际数量为 77/89/90。所有组另有同样的专家 BC regularization，λ=10。', '',
        '### 更新、分布与验收','',
        '- 每运行 96 个训练环境，12 次SAC更新/向量步，batch 256，lr 3e-4。每条训练交互0.125次SAC更新与Phase8相同；每seed精确37,500次SAC更新（各含critic/actor/temperature一步）。并行环境数不同，不能把本阶段continuation AUC与Phase8从零训练AUC当作同一个实验。',
        '- KL(new||anchor) 使用 pre-tanh Gaussian 的解析 KL；同一个当前 std 上限同时作用于 candidate 和冻结 anchor。锚点权重永久冻结。另记录未加 cap 的 raw KL。',
        '- 当双方std都被cap为0.01时，KL的均值项为sum((mu_new-mu_anchor)^2)/(2*0.01^2)。与原始std约0.6相比，相同KL系数下均值偏移惩罚约强3600倍。因此C-B还包含KL度量收紧，不能当作单因素sigma消融；稳定性可能来自强守成。需要结合cap-matched冻结锚点对照判断是否真正改进。',
        '- std 上限同时作用于在线采集、actor loss 与 target-Q action，避免只在评测时减少噪声。std=0.01 是 pre-tanh 无量纲标准差，不是关节角单位。末端位移 action 原 scale=0.05m，旋转=0.3rad，夹爪按符号控制。',
        '- cap是显式控制，不是证明raw std head自主学会了收缩；导出和部署必须保留这一执行分布定义。',
        '- A/B 保留 entropy target=-7；std 受限组使用固定专家参考集上、加 cap 的 anchor squashed entropy 减 1 nat（确定性 Gauss–Hermite 积分）。所以 C-B 是 std 控制及相容熵目标的组合干预，不能单独归因于 σ。raw network std 仍可能很大，实际执行 std 与 raw head 输出分别报告。',
        '- 采集策略在每个候选区间冻结。candidate 可以训练，但只有固定 GT 评价通过才替换采集策略；失败恢复区间前完整模型、target critic、optimizer 与温度。replay 和累计交互数不回退。',
        '- 主要组均使用同样的进度门控，所以 A–D 隔离的是 anchor/std/replay，不能据此声称门控本身的因果收益。区间内保存 raw candidate 与 accepted checkpoint，二者可审计；没有以 Q 选择策略。',
        '- 每 10k 新交互验收：固定 32 episodes，确定性与真实采样策略均测。成功/最大角/最终角/归一化 progress 下降 >0.10，或 regression rate 增加 >0.10 则拒绝。门控比较上一接受策略，仍允许小退化累积。另用不同固定 seed 做 64 episodes/模式的正式曲线。',
        '- 1k 区间 A/C/D pilot 各 10,080 步已保留。它们没有完整在线 episode（仅105物理时间步），仅验证更新/replay/恢复接口。1k 门控评价成本很高，正式组统一固定为10k；没有按组更改预算或只减少某组评价。',
        '- best 按真实采样 success 优先、随后确定性 success/progress 排序；final 为300k接受策略。best 可能就是 step0锚点，因此“最佳结果”不能默认是新学习产生。', '',
        '### 独立评价与统计','',
        '新进程、heldout seed=190000+seed，best/final 各64 episodes/条件/模式。主要组测 nominal 确定性、真实采样、独立 pre-tanh .01 噪声；同时测门初角 +2.5°/+5° 与工装横向 ±1cm（等效把手位移）。负初角会违反原铰链0..1.57rad限制，因此不测试 −5°。扰动仅修改独立评价器的 reset 默认状态，并保存实际初角/把手位姿校验。补充组只测 nominal。冻结 anchor 的 cap-matched counterfactual 使用相同 seed 的 step0独立测试。', '',
        '统计单位是5个训练 seed，不把episode当作独立训练样本。95% CI 为 t(4) 区间，未人为裁剪统计值；曲线概率阴影仅在图中裁剪到[0,1]。paired test 为32种符号翻转的精确双侧检验。5对样本最小非零双侧p值为0.0625，不能据此宣称p<0.05；多组对照探索性报告，不挑选单seed证明提升。', '',
        '## 4. 多 seed 主要结果','',
        '所有区间为均值 [95% CI]。AUC 为新增0–300k accepted-policy success曲线的归一化积分。best/final/gap来自独立复测；负gap表示独立复测final稍高，不代表曲线排序错误。', '',
        '| 组 | 随机策略 AUC | 独立 Best | 独立 Final | Best−Final | 新在线成功条数 | 拒绝次数 |',
        '|---|---|---|---|---|---|---|']
    for arm,s in summary.items():
        lines.append('| '+arm+' | '+' | '.join(metric(s[k]) for k in ('policy_auc','policy_best','policy_final','policy_gap','online_successes','rejections'))+' |')
    lines+=['','| 组 | 确定性 AUC | 独立确定性 Final | 新在线成功比例 | 独立轻噪声 Final |', '|---|---|---|---|---|']
    for arm,s in summary.items():
        lines.append('| '+arm+' | '+' | '.join(metric(s[k]) for k in ('det_auc','det_final','online_success_ratio','noise001_final'))+' |')
    lines+=['','| 组 | Final有效std | Final raw std | Final有效KL | 拒绝比例 | Final跨seed SD |','|---|---|---|---|---|---|']
    for arm,s in summary.items():
        lines.append('| '+arm+' | '+' | '.join(metric(s[k]) for k in ('policy_final_effective_std','policy_final_raw_std','policy_final_kl','rejection_fraction'))+f" | {s['policy_final']['sd']:.4f} |")
    lines+=['','### 验收前candidate诊断','',
        'raw candidate指标来自32-episode验收集，accepted曲线来自不同seed的64-episode正式集。raw曲线是各区间候选的诊断，拒绝后下一候选从上一接受状态继续，不能把它当作一条从未保护的连续训练分支。', '',
        '| 组 | Raw确定性candidate AUC | Accepted确定性AUC | Raw随机candidate AUC | Accepted随机AUC |','|---|---|---|---|---|']
    for arm,s in summary.items():
        lines.append('| '+arm+' | '+' | '.join(metric(s[k]) for k in ('det_raw_candidate_auc','det_auc','policy_raw_candidate_auc','policy_auc'))+' |')
    lines+=['','逐seed全部值、方差、首次成功时间、成功时间窗、原始/有效std、KL与replay来源见 per_seed.csv、summary.json、episodes_*.csv、updates_*.csv、online_windows_*.json。在线episode可能跨多个接受策略版本，记录first/last版本；评价episode为固定checkpoint执行。','','### 配对结果','', '| 比较 | 随机 AUC 差 [95% CI] | Final 差 [95% CI] | Final 双侧 p |','|---|---|---|---|']
    for name in ('B-A','C-B','C-A','D-C','CANN-C','D100-D','D30-D'):
        t=tests[name];lines.append(f"| {name} | {metric(t['policy_auc']['difference'])} | {metric(t['policy_final']['difference'])} | {t['policy_final']['exact_signflip_p_two_sided']:.4f} |")
    lines+=['','### 更新是否优于仅控制 std 的冻结锚点','', '| 组 | 相同 cap 冻结锚点 | Final−冻结锚点 [95% CI] | 配对 p |','|---|---|---|---|']
    for arm,s in summary.items():
        t=tests[f'{arm}-cap_matched_frozen_anchor']['policy']
        lines.append(f"| {arm} | {metric(s['policy_cap_matched_anchor'])} | {metric(t['difference'])} | {t['exact_signflip_p_two_sided']:.4f} |")
    lines+=['','### 相同cap冻结锚点的其他指标对照','',
        '以下均为真实采样策略的独立nominal复测。episode时长包含失败的600步timeout，时长下降可能来自成功率变化，不能直接当作成功episode执行速度提高。reward/progress/时长是补充分析；“未显示额外成功率提高”不等于所有指标均无改善。', '',
        '| 组 | Final mean episode steps | 时长差 Final−Anchor | Final mean reward return | Return差 Final−Anchor | Progress差 |','|---|---|---|---|---|---|']
    for arm,s in summary.items():
        lines.append('| '+arm+' | '+' | '.join(metric(s[k]) for k in ('policy_final_episode_steps','policy_delta_episode_steps','policy_final_return','policy_delta_return','policy_delta_progress'))+' |')
    lines+=['','### 成功episode的条件完成时间（探索性补充）','',
        '由总体mean episode steps与成功比例重建：(mean_steps−600*(1−success))/success。原任务只有成功和600步timeout终止，全部训练失败episode已验证为600步。p=0的组无法估计。单位为60Hz控制步；不作为额外推进门槛，也不能把条件完成速度当作RL交互样本效率。', '',
        '| 组 | Frozen anchor成功完成步数 | Final成功完成步数 | 配对差 Final−Anchor [95% CI] |','|---|---|---|---|']
    for arm,r in speed.items():
        if 'final' in r:
            lines.append(f"| {arm} | {metric(r['anchor'])} | {metric(r['final'])} | {metric(r['paired_final_minus_anchor']['difference'])} |")
        else:
            lines.append(f'| {arm} | N/A | N/A | N/A（至少一seed无成功） |')
    lines+=['','## 5. 小扰动与稳定性门槛','', '| 组 | 工程门槛 | 未通过项 | 显示优于冻结锚点 |','|---|---|---|---|']
    for arm,r in readiness.items():
        failed=', '.join(k for k,v in r['criteria'].items() if not v) or '无'
        lines.append(f"| {arm} | {r['engineering_gate']} | {failed} | {r['demonstrated_improvement_over_cap_matched_anchor']} |")
    lines+=['','操作性门槛于正式终点前固定：所有seed持续产生成功（至少3个10k窗口），随机final均值≥90%，mean best-final gap≤5pp/每seed≤10pp，每seed拒绝≤20%，轻噪声均值≥90%，各扰动条件/模式均值≥80%。它们用于项目推进决策，不能当作安全保证。','','| 组 | 条件 | Best随机 [95% CI] | Final随机 [95% CI] | Best确定性 [95% CI] | Final确定性 [95% CI] |','|---|---|---|---|---|---|']
    for arm in ('A','B','C','D'):
        for c in ('nominal','angle_2p5','angle_5','handle_y_minus_1cm','handle_y_plus_1cm'):
            vals=[next(r['success'] for r in held if r['arm']==arm and r['checkpoint']==selection and r['condition']==c and r['mode']==m) for m in ('policy','deterministic') for selection in ('best','final')]
            lines.append(f'| {arm} | {c} | '+ ' | '.join(metric(v) for v in vals)+' |')
    lines+=['','## 6. 五个研究问题','',
        '### 1）为什么原 SAC 会破坏成功策略？','',
        '同一冻结锚点、没有任何训练更新的诊断中，原始随机分布为0/320成功，σ上限0.01后为90.6%–100%/seed。该对照直接支持执行分布过宽会破坏精细接触控制。在线SAC还会优化熵与Q驱动的动作分布，在专家支持范围外采样，再用失败经验更新。没有单独的critic因果干预，所以Q误差、可观测性和策略漂移只能列为共同机制假设；不能宣称Phase9证明了全部退化原因。', '',
        '代码层面：现有BC regularization是MSE(tanh(mu), expert_action)，只约束动作均值，没有对log-std的直接监督。所以专家均值能力保持与随机执行失败可以同时存在。B把原始宽分布作为KL参考，也不能自动变为窄分布，缩小variance本身还会受到该参考的KL惩罚。C/D显式收缩reference与executed distribution，并与target-Q采样一致；这属于明确的算法干预，并不是证明SAC自行从成功经验学出了恰当的动作variance。', '',
        '### 2）Anchor 是否解决策略漂移？','',
        f"B−A 的独立随机final差为 {metric(tests['B-A']['policy_final']['difference'])}；应结合det_final与拒绝率判断均值能力是否被保持。KL限制距离，不能把原本过宽的anchor动作分布自动变成成功探索。raw KL小也不能替代真实进度测试。", '',
        '### 3）动作分布控制是否增加在线成功经验？','',
        f"C 的新在线成功均值为 {metric(summary['C']['online_successes'])}，D为 {metric(summary['D']['online_successes'])}；A为 {metric(summary['A']['online_successes'])}，B为 {metric(summary['B']['online_successes'])}。这些只来自训练采集，不含预载和评测。支持与否以五seed逐项结果为准，不能用0/0短测作为失败样本。", '',
        '### 4）成功经验进入 buffer 后是否持续提升？','',
        f"D−C 独立final差为 {metric(tests['D-C']['policy_final']['difference'])}。D的anchor成功数据已进入critic replay；所有在线episode也进入在线池。D30/D100提供比例对照。若没有稳定正向差或没有优于cap-matched冻结锚点，只能说形成了成功采集与受保护更新，不能说成功经验已经带来持续策略增益。", '',
        '冻结锚点在相同小std下本来就接近满分，留下的改进空间很小；同时有效KL较强。没有检测到增益不能直接推导“经验无效”，也可能是ceiling与更新约束共同限制。D100只训练专家replay却仍采集在线episode，能帮助区分产生成功与利用新经验；本实验没有无门控组或受限std但无anchor组，不能独立识别各保护组件的必要性。', '',
        '### 5）能否进入 LWD/DIVL？','',
        f"按用户稳定性与泛化标准，工程门槛通过组：{', '.join(a for a,r in readiness.items() if r['engineering_gate']) or '无'}。单独显示优于cap-matched冻结锚点的成功率组：{', '.join(a for a,r in readiness.items() if r['demonstrated_improvement_over_cap_matched_anchor']) or '无'}。进入下一阶段按工程门槛判断，增量成功率证据独立报告，不因接近满分的ceiling额外增加用户未要求的阻断条件。",
        '', '在通过条件前继续暂缓LWD/DIVL/QAM。下一步应围绕已经受控的探索分布做小规模确认：把保持能力与真实在线收益分开，验证接受更新率、cap-matched无更新对照、以及小扰动下观测是否足够。若主要障碍是初态泛化，固定机器人观察可能不足以定位变化的把手，需要独立论证可部署观测接口；本阶段没有改变环境或Actor输入。成功数据排序模块只有在更新收益可靠后再接入。', '',
        '下一次严格控制的稳定性确认可单独检验KL度量：保持std上限与entropy target不变，比较uncapped-reference KL或按variance归一化后的anchor强度，检验是否保留能力同时允许有效改变均值。仅作为后续设计，本阶段没有增开训练或实施LWD/DIVL。', '',
        '## 7. 交互成本与产物','',
        f"- 正式新增训练交互：{cost['training_interactions']:,}。", 
        f"- 固定门控与曲线评价交互：{cost['formal_and_guard_interactions']:,}。",
        f"- 独立评价：{cost['independent_episodes']:,} episodes / {cost['independent_interactions']:,}交互。",
        f"- Pilot：{cost['pilot_interactions']:,}训练 / {cost['pilot_evaluation_interactions']:,}评价交互。",
        f"- 独立扰动物理核查pilot评价交互：{cost['perturbation_pilot_evaluation_interactions']:,}，未进入正式推断。",
        '- 锚点准备初次汇总序列化失败，但64轨迹/seed保存完整；恢复读取同一HDF5后完成原始/受限随机评测。恢复deterministic交互数是按轨迹顺序重建估计，无法严格还原，不混入上述精确成本。日志/失败输出保留。没有把额外评价交互隐藏在300k训练预算之外来宣称真实机器人sample efficiency。',
        '- checkpoint：checkpoints/phase9_safe_online/P9{arm}_seed{seed}/；包含step0、10k节点、raw candidate、best、final和completed.json。采集replay在内存中运行，checkpoint不包含replay快照，不应当作可完整复现buffer的断点续训文件。',
        '- TensorBoard：logs/phase9_safe_online/P9{arm}_seed{seed}/；CSV/独立评价/统计/源代码快照与哈希：results/phase9_safe_online/。',
        '- 可部署策略导出：results/phase9_safe_online/deployment/。safe_online/deploy_policy.py 读取std_cap和26维机器人观察契约；确定性与采样推断已逐文件核对一致。导出包含是否仍为step0锚点，避免把未发生改进的策略标记成在线学习成果。仅导出产物，没有连接或执行真实机器人。',
        '- 已有source checkpoint哈希、expert/环境哈希、旧产物清单与训练源码清单均保存，可核查对照污染。', '',
        '## 8. 图表','',
        '![Success curves](../results/phase9_safe_online/figures/success_curves.png)', '',
        '![Online success and distribution](../results/phase9_safe_online/figures/online_and_distribution.png)', '',
        '![Independent checkpoints](../results/phase9_safe_online/figures/independent_checkpoints.png)', '',
        '![Generalization](../results/phase9_safe_online/figures/generalization.png)', '',
        '## 9. 重现入口','',
        '```bash','cd /home/xiaolong/privileged_rl_validation','source configs/runtime_env.sh',
        '.venv/bin/python experiments/phase9_safe_online/audit.py',
        '# train.py creates unique directories; do not relaunch into completed runs',
        '.venv/bin/python experiments/phase9_safe_online/analyze.py',
        '.venv/bin/python experiments/phase9_safe_online/plot.py',
        '.venv/bin/python experiments/phase9_safe_online/baseline_inventory.py --verify',
        '.venv/bin/python experiments/phase9_safe_online/finalize_report.py','```','']
    destination=ROOT/'docs/phase9_safe_online_report.md'
    destination.write_text('\n'.join(lines),encoding='utf-8')
    print(destination)

if __name__=='__main__':main()
