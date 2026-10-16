"""Read actual completed files; pending results remain explicitly unknown."""
import sys,json,csv,math
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from scipy.stats import t as student_t
from experiments.phase16_r_residual_sac.common import ROOT,EXP,RES,save,sha,utc,protocol

def read(p):return json.loads(p.read_text()) if p.exists() else None
def stats(values):
    a=np.asarray(values,float);n=len(a)
    if not n:return dict(n=0,mean=None,sd=None,ci95=None)
    sd=float(a.std(ddof=1)) if n>1 else None
    half=float(student_t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
    return dict(n=n,mean=float(a.mean()),sd=sd,ci95=[float(a.mean()-half),float(a.mean()+half)] if half is not None else None,values=a.tolist())
def fmt(s):
    if s['mean'] is None:return '尚未完成'
    ci=s['ci95'];return f"{100*s['mean']:.2f}% ± {100*s['sd']:.2f}; [{100*ci[0]:.2f},{100*ci[1]:.2f}] (n={s['n']})" if ci is not None else f"{100*s['mean']:.2f}% (n=1)"
def table_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    P=protocol();hist=read(RES/'historical_inputs.json')
    base=[];methods={k:[] for k in ['A','B','C_sampled','D_sampled','C_mean','D_mean']};metrics=[];nodes=[];cost=[]
    for seed in range(5):
        for arm in ['A','B']:
            for scope in ['random','fixed','validation']:
                path=RES/f'baseline/seed{seed}/{arm}_{scope}.json';r=read(path)
                if r:
                    metrics.append(dict(arm=arm,seed=seed,step=0,scope=scope,distribution='pi05_flow',success=r['success'],episodes=r['episodes'],elapsed_s=r['elapsed_s'],physical_interactions=r['physical_interactions']))
                    if scope=='random' and arm=='A':base.append(r['success'])
                    if scope=='random' and seed<3:methods[arm].append(r['success'])
    for arm in ['C','D']:
        for seed in range(3):
            complete=read(RES/f'train/{arm}_seed{seed}/complete.json')
            if complete:cost.append(dict(arm=arm,seed=seed,**complete))
            for step in P['evaluation']['nodes']:
                folder=RES/f'evaluation/{arm}_seed{seed}/step{step}'
                if not (folder/'complete.json').exists():continue
                for scope in ['validation','random','fixed']:
                    for dist in ['sampled','mean']:
                        r=read(folder/f'{scope}_{dist}.json')
                        if r:
                            metrics.append(dict(arm=arm,seed=seed,step=step,scope=scope,distribution=dist,success=r['success'],episodes=r['episodes'],elapsed_s=r['elapsed_s'],physical_interactions=r['physical_interactions']))
                            if scope=='validation':nodes.append(dict(arm=arm,seed=seed,step=step,distribution=dist,success=r['success']))
                            if scope=='random' and step==50000:methods[arm+'_'+dist].append(r['success'])
    aggregates={k:stats(v) for k,v in methods.items()};baseline=stats(base)
    paired={}
    for label,a,b in [('D-C','D_sampled','C_sampled'),('D-A','D_sampled','A'),('C-A','C_sampled','A'),('B-A','B','A')]:
        av={r['seed']:r['success'] for r in metrics if (r['arm']+'_'+r['distribution'] if r['arm'] in ['C','D'] else r['arm'])==a and r['scope']=='random' and r['step']==(50000 if r['arm'] in ['C','D'] else 0)}
        bv={r['seed']:r['success'] for r in metrics if (r['arm']+'_'+r['distribution'] if r['arm'] in ['C','D'] else r['arm'])==b and r['scope']=='random' and r['step']==(50000 if r['arm'] in ['C','D'] else 0)}
        seeds=sorted(av.keys()&bv.keys());paired[label]=dict(**stats([av[s]-bv[s] for s in seeds]),paired_seeds=seeds,exploratory_only=True,significance_claim=False)
    # Every completed physical rollout is counted once, including engineering and
    # discarded technical attempts. Partial traces are not reported as complete.
    interactions=dict(evaluation=0,engineering=0,technical_failed_attempts=0,training=0,automatic_resets=0)
    for path in RES.rglob('*.json'):
        r=read(path)
        if isinstance(r,dict) and 'trace_sha256' in r and 'physical_interactions' in r:
            category='technical_failed_attempts' if 'attempt001_zero_reset_mismatch' in path.parts else 'engineering' if 'engineering' in path.parts else 'evaluation'
            interactions[category]+=r['physical_interactions'];interactions['automatic_resets']+=r['automatic_resets']
    for x in cost:interactions['training']+=x['training_interactions'];interactions['automatic_resets']+=x['training_automatic_resets']
    interactions['total_physical_interactions']=sum(interactions[k] for k in ['evaluation','engineering','technical_failed_attempts','training'])
    complete=bool((RES/'first_round_complete.json').exists() and len(cost)==6)
    summary=dict(utc=utc(),first_round_complete=complete,historical_random_mean=.896875,new_Frozen_pi05_baseline=baseline,aggregates=aggregates,paired=paired,metrics=metrics,validation_nodes=nodes,cost=interactions,trials=cost,Anchor=False,LWD_DIVL=False,extension_started=False,protocol_sha256=sha(EXP/'protocol.json'))
    save(RES/'summary.json',summary);table_csv(RES/'metrics.csv',metrics);table_csv(RES/'validation_nodes.csv',nodes)
    lines=['# Phase 16-R：冻结 π0.5 + Residual SAC', '',f'实际文件更新时间：{utc()}。状态：'+('三 seed 第一轮已完成；仍为探索性结论。' if complete else '正在执行；下述未完成指标不作结果结论。'),'',
    '## 1. 范围与原模型核验','',
    '本轮是用户单独授权的仿真原子操作在线 RL 可行性验证。旧的可靠随机工位 Anchor 门槛仍未通过。本轮不训练 Recovery、Re-acquire 或 π0.5 Flow Matching，不开展真机训练、LWD、DIVL、QAM。',
    'Phase14.3 Normal C 的五 seed 原随机成功率为87.50/90.625/85.9375/90.625/93.75%，均值89.6875%；Phase15对应副本 SHA 完全相同。该数字与本轮新协议结果不能直接相减计算增益。新旧数据、权重与源码保护项见 historical_inputs.json。',
    '每个 RL seed 对应同编号冻结 Normal seed，A/B/C/D在该seed内完全配对；没有挑最高基线seed。', '',
    '## 2. 预先冻结协议','',
    '60Hz、H=10、每5tick重规划并依次执行索引0–4；每个回合600tick，成功门角严格>1rad。只在超时或实测非法物理状态/硬关节越界终止，停滞/接触丢失/回退不删样本。随机工装XYZ±1cm、请求初角0–5°、原闭门驱动不改；没有独立改变把手形状。实际初角与tick30角在逐episodeCSV保存，初角闭合后不宣称保持初角泛化。',
    '首次预检失败保存于attempt001_zero_reset_mismatch：首次动作与可观测初态相同，但重复reset后的首个物理转移不同；未开始RL。v2在新环境包装中显式清除旧驱动目标，保留关闭门驱动，重新执行严格零残差预检；具体是否通过以physical_gate.json为准。',
    '小残差上限为归一化XYZ±.04、旋转±.02，对应每轴2mm/6mrad目标增量，不是实测TCP移动；中等上限仅作工程接入测试，无结果驱动选模。夹爪完全复制π0.5。零均值初始化不等于初始采样动作严格为零：log_std=-4，采样与均值分别评估。',
    'Critic输入39维标准化状态、当前base、完整chunk与索引，以及六维残差和实际合成七维动作。回放保存下一实际base/chunk，终止上下文置零且不bootstrap。熵是单位残差坐标的tanh Gaussian密度，不冒充裁剪后执行分布密度。clamp实际梯度进入Actor损失；饱和位置梯度为零，另有残差L2项。',
    'C保留原正向门角速度积分；D仅将它替换成20×有符号角差，接近/接触/终奖一致。网络、初始化、训练seed、50k交互、batch256、每8交互一次更新、优化器/温度与残差约束相同。没有额外专家动作。',
    '训练物理进程保留完整现场等待独立评估，不为评估reset未完成训练回合。validation只负责预选最佳，最终50k是主要端点，test不选模。A/B不更新，重复节点复用同一固定配对评估，成本计一次。',
    'SAC方程参考：[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html)。项目执行以固定源码SHA和实际预检为准。','',
    '## 3. 本轮新基线与最终结果','',f'新冻结基线五 seed：{fmt(baseline)}。', '', '| 方法 | 最终随机成功率：均值±seed SD；t95%CI |','|---|---|']
    for method in methods:lines.append(f'| {method} | {fmt(aggregates[method])} |')
    lines+=['','三 seed RL比较只作探索性判断，不声明显著。原数据五 seed不替代新的RL五 seed。配对Δ和CI见summary.json；价值阈值为最终随机至少+3pp且需5+seed确认。','', '## 4. 交互与执行成本','',json.dumps(interactions,ensure_ascii=False), '',
    '训练、评估、工程接入和失败技术尝试分别记账，包含批内未计分却实际步进的env。未完成训练回合是删失，不伪记成功或失败。训练步数AUC不能单独证明真机样本效率；实际总交互与自动reset见统计。','',
    '## 5. 十项验收问题','',
    f'1. **冻结随机成功率？** 本轮{fmt(baseline)}；完整五seed结束后才作Frozen_pi05_baseline。',
    '2. **SAC能稳定训练？** 以实际梯度/Q目标/终止/双Q/熵/restore门槛及全部50k原始曲线判断；未完成时unknown。',
    '3. **是否损伤原能力？** 比较同测试下最终采样C/D−A，以及固定工位保留；不能仅用均值执行判断。',
    '4. **进度奖励额外收益？** 仅用同seed、同预算、同接口D−C；当前配对统计见summary，不完成不下结论。',
    '5. **最终超过冻结π0.5？** 最终50k sampled端点D−A/C−A为主；最佳中间模型不代替最终。三seed不足以确认五seed持续增益。',
    '6. **等交互成本学习效率改善？** C/D相同训练预算，validation曲线与全物理交互单独报告；A零训练，比较时必须包含RL额外交互成本。',
    '7. **持续产生成功经验？** online_episodes.csv记录成功episode和发生步数；各段成功数量与reset数量必须来自实际数据。',
    '8. **训练与部署分布一致？** base、残差采样、实际执行三条动作逐tick保存；训练采样与test采样同分布公式，均值部署是另一分布，不能说二者相同。',
    '9. **训练/评估/reset多少？** 上述成本表为已完成文件账本，活动/未完成部分需从progress/原始trace另标；不能把已完成账本当成最终成本。',
    '10. **值得SAC-Flow/LWD？** 当前不启动。只有最终采样增益、连续成功经验、分布漂移可控且五seed独立复测证实后才讨论；无增益优先排查chunk、Q、执行动作、约束、奖励和夹爪。', '',
    '## 6. 复现与审计','',
    '全部配置、用户原文、冻结SHA、failed尝试、逐tick H5、Replay、checkpoint、TensorBoard、独立节点评估保存在新namespace。独立审计完成前，报告不称最终交付或新的Anchor。']
    doc=ROOT/'docs/phase16_r_report.md'
    if doc.exists():
        archive=RES/'report_revisions'/utc().replace(':','');archive.mkdir(parents=True,exist_ok=True);(archive/doc.name).write_bytes(doc.read_bytes())
    doc.write_text('\n'.join(lines)+'\n')
    save(RES/'report_status.json',dict(utc=utc(),report_sha256=sha(doc),first_round_complete=complete,final_audit_pending=True))
    print(json.dumps(dict(first_round_complete=complete,baseline=baseline,completed_trials=len(cost),cost=interactions)))

if __name__=='__main__':main()
