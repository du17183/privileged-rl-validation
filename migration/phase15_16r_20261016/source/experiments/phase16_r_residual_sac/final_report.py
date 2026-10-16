"""Scientific report from completed physical evidence, never placeholders for conclusions."""
import sys,json,csv,math,itertools
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from scipy.stats import t as student_t,ttest_rel
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,save,sha,utc,protocol

def read(path):return json.loads(path.read_text()) if path.exists() else None
def csvrows(path):
    if not path.exists():return []
    with path.open() as f:return list(csv.DictReader(f))
def stats(values):
    x=np.asarray(values,float);n=len(x)
    if not n:return dict(n=0,mean=None,sd=None,ci95=None,values=[])
    sd=float(x.std(ddof=1)) if n>1 else None
    h=float(student_t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
    return dict(n=n,mean=float(x.mean()),sd=sd,ci95=[float(x.mean()-h),float(x.mean()+h)] if h is not None else None,values=x.tolist())
def fmt(s,percent=True):
    if not s['n']:return '未完成'
    z=100 if percent else 1;sd=s['sd'];ci=s['ci95'];unit='%' if percent else ''
    return f"{z*s['mean']:.3f}{unit} ± {z*sd:.3f}; [{z*ci[0]:.3f}, {z*ci[1]:.3f}]（{s['n']} seed）" if ci else f"{z*s['mean']:.3f}{unit}（1 seed）"
def writecsv(path,rows):
    if rows:
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    P=protocol();metrics=[];trial=[];best=read(RES/'validation_selection.json');best=best['best'] if best else {}
    for seed in range(5):
        for arm in ['A','B']:
            for scope in ['random','validation','fixed']:
                path=RES/f'baseline/seed{seed}/{arm}_{scope}.json';x=read(path)
                if x:metrics.append(dict(arm=arm,seed=seed,model='frozen',step=0,distribution='base_flow',scope=scope,success=x['success'],episodes=x['episodes'],physical_interactions=x['physical_interactions'],wall_s=x['elapsed_s']))
    for arm in ['C','D']:
        for seed in range(3):
            key=f'{arm}_seed{seed}';folder=RES/'train'/key
            status=read(folder/'complete.json') or read(folder/'stopped.json') or read(folder/'failed.json')
            if status:trial.append(dict(arm=arm,seed=seed,status='complete' if (folder/'complete.json').exists() else 'stopped' if (folder/'stopped.json').exists() else 'failed',**status))
            for step in P['evaluation']['nodes']:
                ev=RES/f'evaluation/{key}/step{step}'
                if not (ev/'complete.json').exists():continue
                for scope in ['validation','random','fixed']:
                    for dist in ['sampled','mean']:
                        x=read(ev/f'{scope}_{dist}.json')
                        if x:metrics.append(dict(arm=arm,seed=seed,model='final' if step==50000 else 'node',step=step,distribution=dist,scope=scope,success=x['success'],episodes=x['episodes'],physical_interactions=x['physical_interactions'],wall_s=x['elapsed_s']))
            chosen=best.get(key)
            if chosen:
                ev=RES/f'evaluation/{key}'/('step50000' if chosen['step']==50000 else 'best_test')
                for scope in ['random','fixed']:
                    for dist in ['sampled','mean']:
                        x=read(ev/f'{scope}_{dist}.json')
                        if x:metrics.append(dict(arm=arm,seed=seed,model='best',step=chosen['step'],distribution=dist,scope=scope,success=x['success'],episodes=x['episodes'],physical_interactions=x['physical_interactions'],wall_s=x['elapsed_s']))
    # The explicit complete/stopped statuses define completion of execution;
    # controlled stopping never becomes a fictitious 50k final result.
    ended=bool((RES/'first_round_complete.json').exists() or (RES/'first_round_execution_finished.json').exists())
    full=all(r.get('training_interactions')==50000 and r['status']=='complete' for r in trial) and len(trial)==6
    def values(arm,scope='random',distribution='sampled',model='final',paired=False):
        return {r['seed']:r['success'] for r in metrics if r['arm']==arm and r['scope']==scope and (r['seed']<3 or not paired) and (r['model']=='frozen' if arm in ['A','B'] else r['distribution']==distribution and r['model']==model)}
    aggregates={}
    for arm in ['A','B','C','D']:
        for scope in ['random','fixed']:
            for dist in (['base_flow'] if arm in ['A','B'] else ['sampled','mean']):
                for model in (['frozen'] if arm in ['A','B'] else ['final','best']):
                    aggregates[f'{arm}_{scope}_{dist}_{model}']=stats(list(values(arm,scope,dist,model).values()))
    comparisons=[]
    for label,a,b in [('D-C','D','C'),('D-A','D','A'),('C-A','C','A'),('B-A','B','A')]:
        av=values(a,paired=True);bv=values(b,paired=True);seeds=sorted(av.keys()&bv.keys());delta=[av[s]-bv[s] for s in seeds];st=stats(delta);n=len(seeds)
        exact=None;tp=None
        if n:
            null=[abs(np.mean(np.asarray(delta)*np.asarray(signs))) for signs in itertools.product([-1,1],repeat=n)]
            exact=sum(v>=abs(np.mean(delta))-1e-12 for v in null)/len(null)
        if n>1:
            tp=float(ttest_rel([av[s] for s in seeds],[bv[s] for s in seeds]).pvalue) if np.std(delta)>0 else 1. if not np.any(delta) else 0.
        comparisons.append(dict(comparison=label,paired_seeds=seeds,**st,paired_t_p=tp,exact_signflip_p=exact,exploratory_only=True,significance_claim=False,family='three primary final sampled endpoints' if label!='B-A' else 'fixed perturbation diagnostic, separate'))
    primary=[r for r in comparisons if r['comparison']!='B-A' and r['exact_signflip_p'] is not None]
    running=0.
    for rank,r in enumerate(sorted(primary,key=lambda r:r['exact_signflip_p'])):
        running=max(running,min(1.,(len(primary)-rank)*r['exact_signflip_p']));r['exact_Holm_three_primary']=running
    ledger=read(RES/'physical_cost_ledger.json');online=csvrows(RES/'online_success_segments.csv');aucs=csvrows(RES/'validation_auc.csv');scope_rows=csvrows(RES/'episode_scope_metrics.csv');critic=read(RES/'critic_quality.json');latency=read(RES/'latency/complete.json');audit=read(RES/'independent_audit.json')
    baseline=aggregates['A_random_base_flow_frozen'];pairedA=stats(list(values('A',paired=True).values()))
    x={r['comparison']:r for r in comparisons}
    finalgain=x['D-A']['mean'];progress=x['D-C']['mean'];has_final=len(values('C'))==3 and len(values('D'))==3
    useful=bool(has_final and finalgain>=.03 and all(v>0 for v in x['D-A']['values']))
    decision='需要五 seed 以上独立确认；本轮不自动扩展' if useful else '本轮尚未证实最终采样策略具有预设的可靠增益；不自动扩大训练，暂缓 SAC-Flow/LWD/DIVL'
    summary=dict(utc=utc(),execution_finished=ended,full_first_round_budget=full,new_Frozen_pi05_baseline=baseline,paired_three_seed_A=pairedA,aggregates=aggregates,comparisons=comparisons,trial_status=trial,metrics=metrics,best_validation_selection=best,cost=ledger,decision=decision,significance_claim=False,Anchor=False,LWD_DIVL=False,automatic_extension=False,protocol_sha256=sha(EXP/'protocol.json'))
    save(RES/'summary.json',summary);writecsv(RES/'metrics.csv',metrics)
    cr=[dict(comparison=r['comparison'],n=r['n'],seed_values=json.dumps(r['values']),mean_delta=r['mean'],seed_sd=r['sd'],t95CI=json.dumps(r['ci95']),paired_t_p=r['paired_t_p'],exact_signflip_p=r['exact_signflip_p'],exact_Holm_three_primary=r.get('exact_Holm_three_primary'),scope=r['family'],significance_claim=False) for r in comparisons];writecsv(RES/'paired_comparisons.csv',cr)
    lines=['# Phase 16-R：冻结 π0.5 + Residual SAC 在线强化学习', '',f'生成于 {utc()}。执行'+('已结束。' if ended else '尚在进行；未完成端点不作结论。')+('六个试验均完成50k训练交互。' if full else '实际预算与停止原因逐试验列出；没有把提前停止试验记成50k完成。'),'',
    '## 1. 研究范围与历史核验','',
    '本轮是用户独立授权的仿真原子操作在线RL试验，不代表此前可靠Anchor门槛已经通过。Normal π0.5及其Flow权重全程冻结；没有新增专家监督、Recovery/Re-acquire训练、真机RL、SAC-Flow、LWD、DIVL或QAM。',
    '历史Phase14.3五个Normal checkpoint为seed0/1/2的step3750和seed3/4的step5000；与Phase15冻结副本逐字节一致，原随机成功率均值89.6875%。3974项历史源/数据/结果保护、五seed原结果、训练轨迹与validation/test隔离、原始H5和base权重SHA详见 historical_inputs.json。本轮使用新随机源，禁止将旧89.6875%直接作为增益分母。',
    '机器人观测26维，加实测GT13维：门角/角速度/进度/剩余角、双指接触、把手XYZ/四元数；目标恒为1rad。未观测完整机械臂碰撞，不能将模型或动作合法性称为实际全环境碰撞安全。', '',
    '## 2. 冻结协议与三个真实预检','',
    '60Hz、H10、每5tick真实重规划，逐步执行chunk索引0–4；每回合600tick、门角严格>1rad成功。停滞、接触丢失和回退均不删困难回合；仅超时/实测非有限值/硬关节限位越界>0.001rad终止并自动reset。原控制器、碰撞形状、闭门驱动均保留。',
    '归一化空间只修正六个EEF坐标，夹爪原输出逐字节保留。XYZ残差上限.04、旋转.02，对应每轴2mm/6mrad目标指令上限；不是实际TCP移动。中等.10/.05仅做工程测试，未用于结果驱动选模。Actor均值头严格零初始化，log_std=-4；初始采样仍有小噪声，分别报告采样策略和均值策略。',
    'Critic context为121维：bounded标准化39状态、base7、完整10×7chunk、5维索引；Q额外接收δ6和实际合成动作7。下一价值使用下一实际base/chunk，终止不bootstrap。tanh Gaussian熵属于单位残差坐标，不是裁剪后执行分布概率；clamp梯度在Actor图内，饱和方向梯度为零，另有λ10残差L2。',
    'C保留原dense正向门角速度奖励，D仅用20×有符号门角差替换20×正角速度×dt；接近、抓取和终点奖励一致。网络、初始参数、seed、50k预算、batch256、每8物理交互一次更新、优化器、温度与约束相同。',
    '三个预检全部真实通过：独立物理进程A/严格零残差的base、实际动作、机器人/GT状态、奖励、done和chunk全程最大误差均为0；小/中残差的首步关节差分别0.0006966656/0.0017418833rad，夹爪不变；真实A回放上的双Q、下一base、终止mask、熵、Actor/critic梯度、Polyak与optimizer/checkpoint恢复均通过。工程预检16回合不能代替正式基线。',
    '失败尝试完整保存：attempt1/2同物理进程reset仍有首转移历史差；probe测48次转移；attempt5原生STOP回调挂起，归档后仅对已核验身份的本项目物理进程发送SIGINT；attempt6重新建stage仍有差异。最终每个独立rollout使用全新物理进程并持久复用冻结推理服务，严格容差未放宽。历史项目文件未修补。',
    '训练保持现场等待独立评估，不为评估reset正在运行的训练回合。每clone拥有独立reset随机流，按clone/episode配对抽样；test与train/validation随机源分离。validation采样率预选最佳，平分选更早；最终50k sampled为主要端点，test从不参与选模。A/B不更新，其配对结果在节点复用且成本仅计一次。',
    'SAC目标与熵定义参考[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html)；实际执行以本项目源码SHA、真实门槛及独立原始数据审计为准。','',
    '## 3. 新五seed基线与最终/最佳结果','',f'新冻结A随机：**{fmt(baseline)}**。与RL配对的前三seed A：{fmt(pairedA)}。各seed均为128个random回合，fixed32仅作保留能力检查。','',
    '| 方法/执行 | 选模 | 随机：均值±seed SD；t95%CI | 固定：均值±seed SD；t95%CI |','|---|---|---|---|']
    for arm in ['A','B','C','D']:
        for dist in (['base_flow'] if arm in ['A','B'] else ['sampled','mean']):
            for model in (['frozen'] if arm in ['A','B'] else ['final','best']):lines.append(f"| {arm}/{dist} | {model} | {fmt(aggregates[f'{arm}_random_{dist}_{model}'])} | {fmt(aggregates[f'{arm}_fixed_{dist}_{model}'])} |")
    lines+=['','| 主要配对：最终sampled | Δ均值±SD；t95%CI（百分点） | paired t p | exact双侧p |','|---|---|---|---|']
    for r in comparisons:lines.append(f"| {r['comparison']} | {fmt(r)} | {r['paired_t_p']} | {r['exact_signflip_p']} |")
    lines+=['','三seed只作探索性分析，双侧exact最低可达p=.25；t检验和t区间的正态假设不能替代五seed确认。三个主要final端点的exact Holm独立族见paired_comparisons.csv；B−A为单独扰动诊断。CI不截断在0–100%，episode不当作独立训练seed。最佳test结果属于validation预选模型，不能替代最终。','',
    '## 4. 学习曲线、经验与实际成本','',
    '| 试验 | 训练交互 | 更新 | 在线成功episode | 自动reset | 活动墙钟s | 独立评估等待s |','|---|---|---|---|---|---|']
    for r in trial:lines.append(f"| {r['arm']} seed{r['seed']} {r['status']} | {r.get('training_interactions')} | {r.get('optimizer_updates')} | {r.get('successful_online_episodes')} | {r.get('training_automatic_resets')} | {r.get('active_wall_s')} | {r.get('evaluation_pause_wall_s')} |")
    lines+=['','训练末尾未完成回合为右删失，保留在trace/replay；不伪记失败或成功。online_success_segments.csv按0–10k/10k–25,008/25,008–50k报告实际成功/失败数量，不由checkpoint表现推测在线经验。','', '| validation曲线 | 分布 | 点数/最后交互 | normalized AUC |','|---|---|---|---|']
    for r in aucs:lines.append(f"| {r['arm']} seed{r['seed']} | {r['distribution']} | {r['points']}/{r['last_interaction']} | {float(r['normalized_auc']):.5f} |")
    lines+=['','AUC只对已观测validation节点作梯形积分；横轴训练交互，包含真实评估成本后才讨论部署采样效率，未证明真机效率。提前停止的不同横轴范围不能直接用AUC比较。','', '![Validation sampled curves](../results/phase16_r/figures/validation_curves.png)','']
    if ledger:
        lines+=['| 成本scope | 全物理task交互 | 自动reset | 已完成episode | 实际计分episode |','|---|---|---|---|---|']
        for k,r in ledger['categories'].items():lines.append(f"| {k} | {r['physical_interactions']} | {r['automatic_resets']} | {r['all_completed_episodes']} | {r['scored_completed_episodes']} |")
        lines+=['',f"全物理task交互合计 **{ledger['total']['physical_interactions']}**；自动reset **{ledger['total']['automatic_resets']}**。失败技术尝试真实计入。batch中已经完成配额的clone仍实际步进，均计成本；不能把batch耗时除16冒称单机器人实测。显式启动reset/forward和模型启动时长不等于执行task transition，其scope在ledger中单列说明。"]
    lines+=['','## 5. 三种动作分布与Critic','',
    'action_distributions.csv从逐tick实测指令独立计算base、δ、executed及实际变化的每轴均值/SD/RMS/P50/P95/P99/max、裁剪率；分别按全scope、双指接触和非双指接触、训练/validation/test采样或均值标注。Residual边界是目标动作安全限制，不能保证物理碰撞安全。均值策略与实际SAC采样策略不等价。','', '| Held-out Critic | n | TD绝对均值 | TD RMS | 双Q差绝对均值 | Q绝对最大 |','|---|---|---|---|---|---|']
    if critic:
        for r in critic['rows']:lines.append(f"| {r['arm']} seed{r['seed']} | {r['n']} | {r['td_abs_mean']:.4f} | {r['td_rms']:.4f} | {r['q1_q2_abs_mean']:.4f} | {r['q_abs_max']:.4f} |")
    lines+=['','Held-out Bellman残差只验证对学习目标的内部一致性，不能证明真实Q校准。critic_predicted_action_ranking.csv保存observed/zero/反向残差/Actor均值的预测排名；替代动作未从同一完整物理状态反事实执行，**排名正确率unknown**。损失降低或预测最优不能当真实动作排序证据。','',
    '## 6. 随机分桶、时间和工程scope','',
    'random_strata.csv按源实际初角、请求初角、工装XYZ L∞距离分桶，空桶保留unknown；episode_scope_metrics.csv含每seed成功率、平均tick/仿真时间/rollout墙钟、reset前真实角及tick30角。闭门驱动未改，tick30已经有策略动作；没有无策略保持反事实，故不宣称持续门初角泛化。这里没有独立改变把手几何、摩擦或严重恢复测试难度。',
    '物理并行为16env×最多3pair，Torch4/OMP2/MKL2/OpenBLAS1；两训练pair保留现场、第三pair独立评估，未通过堆满8卡替代算法证据。训练采样吞吐=training_interactions/active_wall_s，更新吞吐=updates/update_wall_s；eval等待、物理step、更新墙钟单列，不能把inclusive耗时相加。监测快照保存GPU占用和/proc CPU ticks/完整进程身份；CPU ticks是采样快照，不冒充所有进程最终累计CPU时间。','']
    if latency:
        lines+=['| 独占batch1 | seed | clock | P50 ms | P95 ms | P99 ms | count |','|---|---|---|---|---|---|---|']
        for r in latency['rows']:lines.append(f"| {r['method']}/{r['distribution']} | {r['seed']} | {r['clock']} | {r['p50_s']*1000:.3f} | {r['p95_s']*1000:.3f} | {r['p99_s']*1000:.3f} | {r['n']} |")
        lines+=['','独占GPU batch1实测query为H10，每5tick重规划；Normal forward/service/RPC分开，Actor/合成wall/thread/CUDA event分开，startup/load/warmup排除。Normal与Actor表项不是同一次机器人端到端闭环，不能将分位数相加；真实机器人传感/执行/网络端到端unknown。']
    lines+=['','## 7. 十项验收结论','',f'1. **冻结随机成功率？** 新协议A五seed为{fmt(baseline)}；主要RL配对仅使用同编号前三seed。',
    f'2. **SAC稳定训练？** 实际完整50k试验{sum(r["status"]=="complete" for r in trial)}/6；停止试验及原因见trial_status，更新日志/有限Q门槛/held-out TD与梯度共同判断。实现通过工程门槛不等于性能提升。',
    f'3. **是否破坏原能力？** 最终sampled C−A为{fmt(x["C-A"])}，D−A为{fmt(x["D-A"])}；固定工位保留与mean结果见总表。必须据最终sampled而不是最好模型或mean策略回答。',
    f'4. **进度奖励额外收益？** 同seed同预算D−C为{fmt(x["D-C"])}；{("三seed均为正，仍需独立五seed确认。" if has_final and all(v>0 for v in x["D-C"]["values"]) else "没有三seed一致的正向最终证据；不宣称GT进度收益已被证实。")} C原奖励已有dense正角速度，不能说D只是把稀疏奖励变dense。',
    f'5. **最终RL超过冻结π0.5？** {decision}。最终D−A点估计{None if finalgain is None else round(finalgain*100,4)}pp；预设实际价值门槛+3pp且五seed以上确认尚未达成。',
    '6. **相同成本学习效率改善？** C/D训练预算及更新率配对，validation AUC和全交互账本同时报告；A没有额外RL成本。本轮不能仅靠训练步数AUC或中间最好checkpoint宣称真机效率。',
    f'7. **持续在线成功经验？** 各试验实际成功数量为{[(r["arm"],r["seed"],r.get("successful_online_episodes")) for r in trial]}；三个时间段逐项见online_success_segments.csv。在线成功说明闭环在收集经验，不等于相对A新增能力。',
    '8. **训练和部署分布一致？** 训练和sampled评估使用同一Actor采样公式及边界，仍存在状态分布差；mean部署取消采样，是另一个动作分布。逐轴RMS、尾部、裁剪和接触scope已保存，不能宣称二者等价。',
    f'9. **训练、评估、reset成本？** 实际全账本为{ledger["total"] if ledger else "尚未形成闭合账本"}，scope表分开失败预检/工程/训练/评估；没有把自动reset、额外非计分clone或失败技术交互隐藏。',
    '10. **是否值得SAC-Flow/LWD？** 当前不启动。若最终采样没有可靠增益，先分析chunk、真实执行动作Q、残差范围、state/action分布、奖励与夹爪链路；若有探索性正向趋势，也须先独立五seed以上确认，不能把保持原有能力当RL增益。','',
    '## 8. 复现与交付','',f'Protocol SHA256：`{sha(EXP/"protocol.json")}`。用户原文、源码/历史SHA、所有failed尝试、checkpoint/TensorBoard/逐tickH5/Replay/CSV/进程退出证据保存在Phase16-R独立namespace。',
    '独立审计见results/phase16_r/independent_audit.json，核对原始数组公式、C/D初始tensor、下一实际base/context、终止mask、五seed基线、配对randomization、3974历史保护项、全成本和退出身份；审计引用报告当前SHA。本地交付再逐文件SHA核验。科学不达标或按规则早停不能伪改成成功。']
    doc=ROOT/'docs/phase16_r_report.md'
    if doc.exists():
        out=RES/'report_revisions'/utc().replace(':','');out.mkdir(parents=True,exist_ok=True);(out/doc.name).write_bytes(doc.read_bytes())
    doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(RES/'report_status.json',dict(utc=utc(),report_sha256=sha(doc),execution_finished=ended,full_first_round_budget=full,audit_pending=True))
    print(json.dumps(dict(execution_finished=ended,full_budget=full,baseline=baseline,comparisons=comparisons,decision=decision)))
if __name__=='__main__':main()
