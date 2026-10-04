"""Write the final report only from complete five-seed, independently tested runs."""
import json
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
from experiments.phase10_quality_lwd.protocol import ARMS,PROTOCOL
from experiments.phase10_quality_lwd.analyze import summarize
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'
NAMES=dict(REFC='Phase9 C 原配方续训',A='仅在线 replay',B='Uniform Success Replay',
           D05='Quality Weighted T=0.5',D10='Quality Weighted T=1.0',D20='Quality Weighted T=2.0',E='LWD-style selection')


def stat(s,percent=False,pp=False):
    f=100 if percent or pp else 1
    unit='pp' if pp else '%' if percent else ''
    precision=2 if percent or pp else 4
    return f"{s['mean']*f:.{precision}f}{unit} [{s['ci95'][0]*f:.{precision}f}, {s['ci95'][1]*f:.{precision}f}]"


def main():
    summary=json.loads((OUT/'summary.json').read_text());tests=json.loads((OUT/'paired_tests.json').read_text())
    heldout=json.loads((OUT/'heldout_summary.json').read_text());gates=json.loads((OUT/'gates.json').read_text())
    quality=json.loads((OUT/'quality_diagnosis.json').read_text());qdiag=json.loads((OUT/'frozen_value_diagnosis.json').read_text())
    train=json.loads((OUT/'training_completed.json').read_text());he=json.loads((OUT/'heldout_completed.json').read_text())
    assert train['runs']==35 and he['jobs']==35
    preservation=json.loads((OUT/'baseline_preservation.json').read_text())
    if preservation['changed']:raise RuntimeError('Prior artifacts changed')
    preflight=json.loads((OUT/'preflight.json').read_text())
    historical=json.loads((ROOT/'results/phase8_progress_rl/summary.json').read_text())
    supplemental=json.loads((OUT/'supplemental_baseline_preservation.json').read_text())
    if supplemental['changed']:raise RuntimeError('Earlier source artifacts changed')
    records=json.loads((OUT/'per_seed.json').read_text())
    report=['# Phase 10：稳定在线 RL 的质量经验利用实验',
        f"\n复核时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}（北京时间）。35/35 训练、35/35 独立复测全部完成。历史 {preservation['artifacts']} 个科学文件及补充 {supplemental['artifacts']} 个既有源文件未变更。",
        '\n## 1. 比较口径与固定基础',
        '\n所有组从相同 seed 的 **Phase9 C 最终 checkpoint** 继续，恢复 actor、critic、target critic、optimizers 和 alpha。冻结 anchor 仍是原 Phase8 B actor，未重新锚定到 Phase9 C。std≤0.01 同时用于采集、actor loss 和 target-Q。机器人 Actor/Critic 均只读取 26 维机器人观察，GT 仅用于原奖励、保护评估和本阶段完成轨迹质量计算。',
        '\nPhase9 checkpoint未包含完整replay快照，因此全部组新建在线buffer。此前5001条成功只作为历史指标，不算作本阶段新增成功，也未伪称已恢复旧在线轨迹。REFC保留相同配方与模型/optimizer起点，replay采用同样受控的fresh初始化。',
        '\n每组五 seed、追加 300k 交互；96 并行环境，batch256，每个向量步12次更新，共37500次更新/seed。持续 BC λ=10、KL权重1、lr3e-4、Progress Reward、Door任务、expert HDF5、reset、机器人配置全部保持。每10k保护评估32 episodes/模式，固定正式评估64 episodes/模式；独立复测更换进程和随机种子。',
        '\n**不是第一次学会开门的 sample efficiency 实验。** 初始成功率接近天花板；横轴/AUC指追加训练交互，不包含此前学习和专家生成。只有 matched continuation 组间差值能回答本阶段经验利用问题。',
        '\n|组|SAC replay方式|作用|\n|---|---|---|',
        '|REFC|50% expert +50% fresh online，含未完成episode|精确保留Phase9 C配方|',
        '|A|100% fresh online；BC样本仍来自相同expert|用户要求的online replay对照|',
        '|B|相同expert+已完成online池，按transition均匀|Uniform Success Replay|',
        '|D05/D10/D20|B的同一池，加权temperature .5/1/2|隔离质量权重作用|',
        '|E|B的同一池，质量最高quintile+均匀支持，保留所有边界并列|简化experience selection|',
        '\nC是各组共用的 Progress Quality 诊断模块，没有额外网络或独立训练组。B与REFC/A还存在专家比例、未完成episode资格差异，不能将该对比全部归因于质量；D/E对B才是主要受控比较。',
        '\n## 2. 完成轨迹质量与可迁移接口',
        '\n`score = .55×success + .20×final_progress + .15×net_progress + .10×contact_stability`。进度裁剪到[0,1]；contact只在门实际移动超过0.02rad后计算双指接触比例。完整成功/部分成功/失败并非强制三档，保留连续值。',
        '\n只在episode结束后打分；pending transition从B/D/E排除，保留在REFC/A。权重为`0.1×uniform + 0.9×normalize(length×exp((score−1)/T))`，轨迹内uniform抽transition。E为20%全部uniform+80%最高质量分位，分位边界并列全部保留。长短轨迹的基础质量密度相同，避免误把trajectory uniform与transition uniform混为一谈。',
        f"\n专家 {preflight['expert_quality']['trajectories']} 条/{preflight['expert_quality']['transitions']:,} transitions，score全为1.0。这说明该质量定义无法进一步排序成功专家内部，而非证明所有成功轨迹的策略梯度效用完全相同。",
        '\n保存HDF5 flat transitions与episode索引(start,length,stride)，保留robot、privileged、action、reward、next状态、done和return/value/success/quality。`trajectory_quality/trajectory.py`提供完成轨迹对象。Actor部署仅使用`state["robot"]`，不读取GT。',
        '\n## 3. 五 seed 主结果',
        '\n所有success均为实际受限随机采样策略；best/final为独立复测，AUC来自固定正式学习曲线。均值及t95 CI，CI未裁剪，越界表示小样本线性区间的局限。best根据训练验证选择，独立复测可能低于final，gap允许负值。',
        '\n|组|Success AUC [95%CI]|独立Best [95%CI]|独立Final [95%CI]|Best−Final(pp)|在线成功/完成episode|回退/seed|Final种子SD(pp)|\n|---|---|---|---|---|---|---|---|']
    for arm in ARMS:
        s=summary[arm];sub=[r for r in records if r['arm']==arm]
        success=sum(r['online_successes'] for r in sub);episodes=sum(r['online_episodes'] for r in sub)
        report.append(f"|{arm} {NAMES[arm]}|{stat(s['policy_auc'])}|{stat(s['policy_best'],True)}|{stat(s['policy_final'],True)}|{100*s['policy_gap']['mean']:.2f}|{success:,}/{episodes:,} ({100*success/episodes:.2f}%)|{s['rollback_events']['mean']:.2f}|{100*s['policy_final']['sd']:.2f}|")
    report+=['\n![Success and return](../results/phase10_quality_lwd/figures/success_reward_curves.png)',
             '\n![AUC and endpoints](../results/phase10_quality_lwd/figures/auc_endpoints.png)',
             '\n### 配对检验',
             '\n|比较|AUC差值 [95%CI]|Final差值(pp) [95%CI]|AUC exact p|Final exact p|\n|---|---|---|---|---|']
    for comparison in ('A-REFC','B-REFC','D05-B','D10-B','D20-B','E-B'):
        c=tests[comparison]
        report.append(f"|{comparison}|{stat(c['policy_auc']['difference'])}|{stat(c['policy_final']['difference'],True,pp=True)}|{c['policy_auc']['exact_signflip_p']:.4f}|{c['policy_final']['exact_signflip_p']:.4f}|")
    report+=['\n五配对seed的双侧exact sign-flip最小p=.0625，不能声称p<.05。三个temperature的Holm校正见paired_tests.json；D10是预定主要温度，未从测试集选最优温度再做未校正结论。',
        '\nD05对B的AUC存在探索性正向信号，未校正t95 CI为正；exact p=0.125，三个温度家族Holm p=0.375，因此没有正式显著性证据。预定D10与E的AUC/Final差值区间都跨0。',
        '\n### 对Phase9 C原配方续训的补充比较',
        '\n这些比较同时包含source quota与完成episode资格变化，仅作为配方的次要描述，未替代预定D/E对B的质量隔离比较。',
        '\n|比较|AUC差 [95%CI]|Final差(pp) [95%CI]|\n|---|---|---|']
    for arm in ('D05','D10','D20','E'):
        c=tests[f'{arm}-REFC']
        report.append(f"|{arm}-REFC|{stat(c['policy_auc']['difference'])}|{stat(c['policy_final']['difference'],True,pp=True)}|")
    report+=[
        '\n### 冻结初始策略对照',
        f"\nPhase9 C源策略的独立受限随机成功率：{stat(summary['REFC']['policy_frozen_initial'],True)}。",
        '\n旧Phase9独立复测为97.50%；本阶段同一checkpoint更换heldout随机种子后得到96.25%，对应312/320与308/320成功。两次测试差异不表示模型在重新加载时退化；本阶段增量比较使用同一新协议下的冻结对照。',
        '\n|组|Final−冻结初始(pp) [95%CI]|\n|---|---|']
    for arm in ARMS:report.append(f"|{arm}|{stat(summary[arm]['policy_gain_vs_frozen'],True,pp=True)}|")
    report+=['\n## 4. 实际经验采样分布与质量饱和',
        '\n这里统计实际37500×256次SAC抽样/seed，而非buffer名义配置。采样当时未完成的episode标为pending，未事后追溯改成成功。BC额外expert样本量相同，不并入SAC replay分母。',
        '\n|组|Expert抽样|已完成Online成功|已完成Online失败|Pending|最高质量在线比例|选中轨迹比例|最大质量密度倍率|最终质量加权TV|\n|---|---|---|---|---|---|---|---|---|']
    for arm in ARMS:
        s=summary[arm];fq=np.mean([r['full_quality_fraction'] for r in quality if r['arm']==arm])
        report.append(f"|{arm}|{100*s['sample_expert_fraction']['mean']:.2f}%|{100*s['sample_online_success_fraction']['mean']:.2f}%|{100*s['sample_online_failure_fraction']['mean']:.2f}%|{100*s['sample_online_partial_fraction']['mean']:.2f}%|{100*fq:.2f}%|{100*s['selected_fraction']['mean']:.2f}%|{s['quality_density_max']['mean']:.3f}|{s['final_quality_mass_total_variation']['mean']:.5f}|")
    report+=['\n![Sampling distribution](../results/phase10_quality_lwd/figures/sampling_distribution.png)',
        '\n质量全为1.0的并列成功经验，E会全部保留；不会随机保留20%再称为质量筛选。若大部分成功经验均饱和，实际筛选或权重改变很小。无收益只能说明本评分与当前成功率/数据分布的受控结果，不能推广为任何quality utilization都无效。',
        '\nTV是同一最终完成轨迹池中，加权/筛选分布对transition-uniform的total variation，仅衡量本质量规则改变了多少采样质量；REFC/A未实施质量干预，TV记0，其固定source quotas由实际抽样表体现。',
        '\n## 5. Quality与SAC Q的可靠性',
        '\ncompleted quality直接包含success和最终门进度，因此其success ranking AUC会存在标签同义，不能与Phase5固定前缀预测AUC=.888直接等同。它是工装完成轨迹的可验证物理评价，不是提前预测器。',
        '\n在独立final随机复测内额外捕获64条完整轨迹/seed，不增加rollout。同一冻结Actor和alpha，记录初始Q、普通reward MC，以及从下一动作起加入折扣条件期望熵的soft MC代理；熵用GH12及训练原有epsilon约定计算，避免tanh饱和动作无法反求logp。SAC Q应与soft return对照。在线采集中的Q随更新变化，只作为非平稳描述。',
        '\n|组|Quality对reward MC Spearman(有效seed)|Q对reward MC Spearman(有效seed)|Q对soft MC Spearman(有效seed)|\n|---|---|---|---|']
    for arm in ARMS:
        entries=[r for r in qdiag if r['arm']==arm]
        vals=[]
        for key in ('quality_vs_reward_mc_rho','q_vs_reward_mc_rho','q_vs_soft_mc_rho'):
            numbers=[r[key] for r in entries if r[key] is not None and np.isfinite(r[key])]
            vals.append(f"{np.mean(numbers):.3f} ({len(numbers)}/5)" if numbers else '无定义：质量/回报无变异')
        report.append(f"|{arm}|{'|'.join(vals)}|")
    report+=['\n|组|初始Q均值 [seed t95CI]|Soft MC代理均值 [seed t95CI]|Q−softMC偏差 [seed t95CI]|双Q分歧均值|\n|---|---|---|---|---|']
    for arm in ARMS:
        entries=[r for r in qdiag if r['arm']==arm]
        report.append(f"|{arm}|{stat(summarize([r['initial_q_mean'] for r in entries]))}|{stat(summarize([r['soft_mc_mean'] for r in entries]))}|{stat(summarize([r['q_soft_mc_bias'] for r in entries]))}|{np.mean([r['twin_q_disagreement'] for r in entries]):.4f}|")
    report+=['\n这是相关性诊断，不能用正相关证明梯度收益，也不能用reward-only MC与soft Q差异直接证明Q错误。Q预测的是策略下期望，单条MC含未来随机性；低单轨迹相关不自动等于期望估计失准。Quality是否更适合经验利用最终由D/E对B的配对改进判定。全成功样本缺少失败对照，相关性检验能力有限。',
        '\n## 6. 独立轻量泛化测试',
        '\n训练环境完全未改。仅评估进程覆盖合法初始+2.5/+5°、cabinet/fixture y±1cm以及柜体碰撞材质静/动摩擦×0.9/1.1，验证真实物理值。把手变化通过fixture平移实现，未改单独把手几何。原铰链下限为0、标称起点0，−5°不可合法测试；没有裁剪后伪称完成±5°。',
        '\nREFC/B/D10/E预定主要比较组测试best和final，其他温度与A只做nominal，避免根据结果挑温度补测而扩大规模。',
        '\n|组|checkpoint|标称随机|+2.5°|+5°|fixture −1cm|fixture +1cm|摩擦×0.9|摩擦×1.1|\n|---|---|---|---|---|---|---|---|---|']
    conditions=['nominal','angle_2p5','angle_5','handle_y_minus_1cm','handle_y_plus_1cm','friction_0p9','friction_1p1']
    for arm in ('REFC','B','D10','E'):
        for selection in ('best','final'):
            values=[next(t['success']['mean'] for t in heldout if t['arm']==arm and t['checkpoint']==selection and t['condition']==condition and t['mode']=='policy') for condition in conditions]
            report.append(f"|{arm}|{selection}|"+'|'.join(f'{100*v:.2f}%' for v in values)+'|')
    report+=['\n### 初始观察对变化的可区分性',
             '\n|条件|robot观察最大变化(五seed平均)|GT最大变化(五seed平均)|\n|---|---|---|']
    for condition in conditions:
        t=next(t for t in heldout if t['arm']=='REFC' and t['checkpoint']=='final' and t['condition']==condition and t['mode']=='policy')
        report.append(f"|{condition}|{t['initial_robot_obs_delta']['mean']:.8f}|{t['initial_gt_delta']['mean']:.8f}|")
    report+=['\n![Generalization](../results/phase10_quality_lwd/figures/generalization.png)',
        '\n泛化失败不等于优化器无法完成标称任务。Actor输入缺少视觉、门初态和把手位置反馈；独立记录初态robot observation和GT相对nominal的变化，检查输入是否能区分变化。输入相同而GT变化只能证明初始观察混叠，不能证明之后完全无法通过机器人接触反馈恢复。本阶段未受控增加可部署观察来隔离原因，不能将所有失败直接归因于观测不足。',
        '\n## 7. 结论与进入完整LWD/DIVL条件']
    c=tests['D10-B'];e=tests['E-B']
    report += [f"\n1. **质量经验是否带来增益？** 尚未证明稳定提升。预定D10对Uniform B的AUC差 {stat(c['policy_auc']['difference'])}，final差 {stat(c['policy_final']['difference'],True,pp=True)}；E的AUC差 {stat(e['policy_auc']['difference'])}，final差 {stat(e['policy_final']['difference'],True,pp=True)}。这些区间均跨0。D05的AUC正向信号保留为探索性结果。",
        '\n2. **环境进度能否评价经验？** 可以对完成轨迹作物理一致评价；当前score存在成功饱和，尚不能排序成功经验对策略梯度的实际效用。',
        '\n3. **Quality是否比Q可靠？** 对已发生的物理结果有直接可核验依据；对预测未来回报、提前筛选、产生梯度增益则未由标签排序本身证明。具体MC/Q诊断见第5节。',
        '\n4. **收益是速度还是最终成功率？** D05有小幅AUC信号，D10最终成功率点估计最高，E点估计轻微改善；当前五seed结果不足以确认稳定的学习速度或最终成功率收益。初始高成功率、99%左右的在线成功经验、评分饱和共同限制了增益空间。E因边界并列实际保留99.23%轨迹，质量加权最终TV仅约0.004–0.011，实际干预较弱。这些是已观察机制线索，不证明任何quality利用都无效。',
        '\n5. **完整LWD/DIVL条件：**']
    point_best=max(ARMS,key=lambda a:summary[a]['policy_final']['mean'])
    supported=[a for a in ('D10','E') if gates[a]['ready_for_full_lwd_divl']]
    report.append(f"\n独立final点估计最高：{point_best}，{100*summary[point_best]['policy_final']['mean']:.2f}%。"+
        (f"满足预定条件的候选：{', '.join(supported)}；仍需更大统计功效验证正式显著性。" if supported else
         '尚无通过全部条件的质量经验利用方案；保持Phase9 C作为默认基础，不能用最高点估计替代配对统计与泛化门槛。'))
    for arm,gate in gates.items():
        report.append(f"\n- {arm}：{'满足预定工程/探索性增益条件' if gate['ready_for_full_lwd_divl'] else '尚未满足'}。"+'；'.join(f"{k}={v}" for k,v in gate['criteria'].items())+'。')
    report += ['\n本阶段只实现completed trajectory scoring、bounded weighted replay和tie-aware selection。未实现完整LWD/DIVL、QAM或VLA。下一阶段应根据上面的实际瓶颈决定：质量饱和时研究可区分成功轨迹的事先定义质量因素；泛化不足时单独验证可部署观察或可重复初态反馈；不在本报告中修改现有任务后追求漂亮数值。',
        '\n## 8. 历史方法背景（非本阶段配对比较）',
        '\n|历史方法|Success AUC|独立Final|口径|\n|---|---|---|---|',
        '|Phase2 普通SAC|0.183|45.3%|原奖励，500k，确定性|',
        '|Phase2 Privileged Critic|0.089|0%|原奖励，500k，确定性|',
        '|Phase6 持续BC约束B3|0.376|31.9%|原奖励，500k，确定性|',
        f"|Phase8 baseline A|{historical['A']['auc']['mean']:.4f}|80.00%|持续BC、expert mixing、保护；原奖励，确定性独立复测|",
        f"|Phase8 Progress Reward B|{historical['B']['auc']['mean']:.4f}|95.94%|进度奖励，确定性|",
        '|Phase9 C|0.9584|97.50%|原Phase8 checkpoint起点，300k，随机std≤.01|',
        '\n历史数值用于理解演进，不与本阶段追加300k的AUC直接做因果或显著性比较。纯BC短预训练约6%与含持续BC、expert混合、保护的SAC不构成同更新量控制，不能把差异全部归因于SAC。',
        '\n## 9. 资源、交互成本与交付']
    eval_train=sum(r['evaluation_steps'] for r in records)
    pilot_markers=[p for p in (ROOT/'checkpoints/phase10_quality_lwd').glob('*_smoke/completed.json')]
    pilots=[json.loads(p.read_text()) for p in pilot_markers]
    independent_pilot=[]
    for p in (OUT/'heldout_smoke').glob('*_seed?.json'):
        if not p.name.startswith('q_diagnosis_'):independent_pilot+=json.loads(p.read_text())
    report += [f"\n正式追加training interactions：{train['steps']:,}（35×300k）；保护及正式学习曲线评估：{eval_train:,}；独立复测：{he['evaluation_steps']:,}（{he['tests']}测试，{he['episodes']:,} episodes）。pilot训练{sum(p['steps'] for p in pilots):,}，pilot评估{sum(p['cumulative_eval_steps'] for p in pilots):,}。",
        f"\n独立pilot接口复测：{sum(r['episodes'] for r in independent_pilot)} episodes、{sum(r['eval_env_steps'] for r in independent_pilot):,} interactions，单独计入开销、不混入正式五seed推断。",
        '\n评估交互是实际物理仿真交互，不能隐去后称真实机器人总样本效率。对真实工装部署应降低/重新设计验证成本，再验证一致性。eight B300共享现有RLinf负载：开始每卡2 learner/evaluator pairs、1536训练env，稳定后按吞吐偏好提高至每卡最多3 pairs、2304训练env；总35组及各组学习参数不变，没有停止其他项目任务。独立复测每卡最多2进程。',
        '\n- CSV、配对结果、95%CI、抽样JSONL、图：`results/phase10_quality_lwd/`。',
        '\n- 完整SAC checkpoints：`checkpoints/phase10_quality_lwd/`；TensorBoard：`logs/phase10_quality_lwd/`。',
        '\n- 每条新在线轨迹的数据及完成episode索引：`datasets/phase10_quality_lwd/`。',
        '\n- 来源hash及固定协议：`results/phase10_quality_lwd/preflight.json`；历史保护：`baseline_preservation.json`。',
        '\n复现：先`source configs/runtime_env.sh`；使用`python -m experiments.phase10_quality_lwd.audit`检查接口，`train.py --arm ARM --seed SEED --device cuda:0`运行新输出目录。已有结果目录禁止覆盖。分析与报告命令只在35训练及独立复测全部完成后运行。']
    (ROOT/'docs/phase10_quality_lwd_report.md').write_text('\n'.join(report),encoding='utf-8')
    print('Final Phase10 report written from complete independently tested data')

if __name__=='__main__':main()
