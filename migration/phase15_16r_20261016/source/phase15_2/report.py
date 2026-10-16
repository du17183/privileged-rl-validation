"""Raw episode statistics, paired model-seed tests and evidence-limited reports."""
import csv
import itertools
import json
from collections import Counter,defaultdict
import h5py
import numpy as np
from scipy import stats
from phase15_2.common import ROOT,RESULTS,read,save_json,sha256
from phase15_2.safe_retreat.retreat_analysis import analyze,STAGES
from phase15_2.run import ART,GRADES,MAIN

METHODS=['A','B','C','D','E']

def interval(values):
    x=np.asarray(values,float);mean=float(x.mean())
    half=float(stats.t.ppf(.975,len(x)-1)*stats.sem(x)) if len(x)>1 else float('nan')
    return dict(mean=mean,low=mean-half,high=mean+half,sd=float(x.std(ddof=1)))

def paired(x,y):
    d=np.array(x)-np.array(y);null=np.array([np.mean(d*np.array(s)) for s in itertools.product([-1,1],repeat=5)])
    p=float(np.mean(np.abs(null)>=abs(d.mean())-1e-12))
    return dict(**interval(d),exact_signflip_p=p,t_test_p=float(stats.ttest_rel(x,y).pvalue) if np.std(d)>0 else None,
        signflip_minimum_two_sided_p=.0625)

def csv_file(name,rows):
    path=RESULTS/'metrics'/name;path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def pct(d):return f"{d['mean']*100:.2f}% [{d['low']*100:.2f}, {d['high']*100:.2f}]"

def main():
    seeds=defaultdict(lambda:defaultdict(list));per_seed=[];stage_rows=[];router_rows=[];failures=[];bins=[];timing=[]
    output_sha={};total_trials=0;physical_steps=0
    for method in METHODS:
        for seed in range(5):
            for test in MAIN+GRADES:
                path=RESULTS/f'test/{method}_seed{seed}{"_grades" if test in GRADES else ""}/{test}.json'
                d=read(path);rows=read(path.with_suffix('.telemetry.json'));n=len(rows)
                if n!=d['episodes'] or max(r['original_episode_ticks'] for r in rows)>600:raise RuntimeError('Raw coverage/budget failed')
                success=sum(r['success'] for r in rows)/n
                if abs(success-d['success'])>1e-9:raise RuntimeError('Episode success mismatch')
                seeds[method][test].append(success);total_trials+=n
                steps=d.get('env_steps',d.get('interactions',0));physical_steps+=steps
                output_sha[str(path.relative_to(ROOT))]=sha256(path)
                output_sha[str(path.with_suffix('.telemetry.json').relative_to(ROOT))]=sha256(path.with_suffix('.telemetry.json'))
                per_seed.append(dict(method=method,seed=seed,test=test,episodes=n,success=success,interactions=steps))
                analysis=[analyze(r) for r in rows]
                for j,name in enumerate(STAGES):
                    count=sum(a['stages'][j] for a in analysis);preceding=sum(a['stages'][j-1] for a in analysis) if j else n
                    # Stages may be skipped by Recovery-only routes. Conditional denominator is intersection, not invented sequential flow.
                    joint=sum(a['stages'][j] and (a['stages'][j-1] if j else True) for a in analysis)
                    stage_rows.append(dict(method=method,seed=seed,test=test,stage=name,episodes=n,reached=count,
                        episode_rate=count/n,previous_reached=preceding,intersection=joint,
                        conditional_rate=joint/preceding if preceding else None))
                for field in ['retreat_attempts','retreat_plans','pregrasp_plans','checked_joint_safe','collision_free_model_safe_pose',
                        'retreat_candidate_found','sampled_retreat_checks','sampled_joint_safe','sampled_collision_free']:
                    stage_rows.append(dict(method=method,seed=seed,test=test,stage=field,episodes=n,reached=sum(a[field] for a in analysis)))
                router_rows.append(dict(method=method,seed=seed,test=test,episodes=n,
                    switches_mean=np.mean([a['switches'] for a in analysis]),
                    reacquire_repeats_mean=np.mean([a['reacquire_repeats'] for a in analysis]),
                    trigger_rule_violation_proxy=sum(a['trigger_rule_violation_proxy'] for a in analysis),
                    **{field:sum(a[field] for a in analysis) for field in ['handoff_5s_relapses','handoff_5s_observed','handoff_5s_timeout_censored','handoff_5s_success_competing']}))
                for category,count in Counter(a['category'] for a in analysis).items():
                    failures.append(dict(method=method,seed=seed,test=test,category=category,count=count,episode_denominator=n))
                if test=='natural_severe':
                    cohort=RESULTS/f'cohorts/test/seed{seed}/natural_severe.h5'
                    with h5py.File(cohort,'r') as h:
                        distances={key:float(g.attrs['distance']) for key,g in h.items()}
                    for lo,hi in [(.10,.125),(.125,.175),(.175,.25)]:
                        chosen=[r for r in rows if lo<=distances[r['cohort_id']]<hi];count=sum(r['success'] for r in chosen)
                        bins.append(dict(method=method,seed=seed,lower_m=lo,upper_m=hi,n=len(chosen),successes=count,
                            success=count/len(chosen) if chosen else None))
                with h5py.File(path.with_suffix('.stage_trace.h5'),'r') as h:
                    trace=h['steps'][:]
                    if trace['actual_arm_collision_known'].any():raise RuntimeError('Unknown collision instrumentation overwritten')
                for role in ['router','planner','control','entry_check','transition']:
                    values=d['inference'].get(role+'_latency',{})
                    if values:timing.append(dict(method=method,seed=seed,test=test,role=role,scope='shared_gpu_batch32_physical_control',**values))
                requests=read(path.with_suffix('.inference.json'))
                for role in ['normal','recovery']:
                    selected=[r for r in requests if r['module']==role]
                    for field in ['latency_s','service_s','rpc_s']:
                        values=[r[field] for r in selected]
                        if values:timing.append(dict(method=method,seed=seed,test=test,role=role+'_'+field,
                            scope='shared_gpu_variable_inference_batch',samples=len(values),
                            **{f'p{p}_s':float(np.quantile(values,p/100)) for p in [50,95,99]}))
            art_values=[seeds[method][test][seed] for test in ART]
            seeds[method]['artificial_macro'].append(float(np.mean(art_values)))
    aggregates=[]
    for method in METHODS:
        for test,values in seeds[method].items():
            if len(values)!=5:raise RuntimeError('Missing model seed')
            aggregates.append(dict(method=method,test=test,seeds=5,**interval(values)))
    comparisons=[]
    for left,right in [('B','A'),('C','B'),('D','C'),('E','D'),('E','A')]:
        for test in ['fixed','random','artificial_macro','natural_severe','full_loop']:
            comparisons.append(dict(left=left,right=right,test=test,**paired(seeds[left][test],seeds[right][test])))
    ordered=sorted(range(len(comparisons)),key=lambda i:comparisons[i]['exact_signflip_p']);previous=0
    for rank,i in enumerate(ordered):
        previous=max(previous,min(1,(len(ordered)-rank)*comparisons[i]['exact_signflip_p']))
        comparisons[i]['holm_exact_p']=previous
    deployment=[]
    for seed in range(5):
        d=read(RESULTS/f'deployment_latency/seed{seed}.json')
        for module,v in d.items():deployment.append(dict(seed=seed,module=module,scope='exclusive_gpu_batch1_no_simulation',
            p50_rpc_s=v['p50_rpc_s'],p95_rpc_s=v['p95_rpc_s'],p99_rpc_s=v['p99_rpc_s'],budget_met=v['budget_met']))
    # Gates concern E, pre-specified as the full candidate. Mean-only versus lower-CI evidence is explicit.
    index={(r['method'],r['test']):r for r in aggregates}
    effects={(r['left'],r['right'],r['test']):r for r in comparisons}
    def effect(left,right,test):
        row=effects[left,right,test]
        return f"{left}−{right}: {row['mean']*100:+.2f} percentage points，95%CI [{row['low']*100:+.2f}, {row['high']*100:+.2f}]，精确p={row['exact_signflip_p']:.4f}，Holm p={row['holm_exact_p']:.4f}"
    gates={test:dict(threshold=threshold,mean=index['E',test]['mean'],ci_low=index['E',test]['low'],
        mean_pass=index['E',test]['mean']>threshold if test=='natural_severe' else index['E',test]['mean']>=threshold,
        lower_ci_pass=index['E',test]['low']>threshold if test=='natural_severe' else index['E',test]['low']>=threshold)
        for test,threshold in [('fixed',.95),('random',.90),('artificial_macro',.85),('natural_severe',.50),('full_loop',.90)]}
    def relapse(method):
        rs=[r for r in router_rows if r['method']==method and r['test']=='natural_severe']
        num=sum(r['handoff_5s_relapses'] for r in rs);den=sum(r['handoff_5s_observed'] for r in rs)
        return dict(n=num,denominator=den,rate=num/den if den else None,timeout_censored=sum(r['handoff_5s_timeout_censored'] for r in rs),
            success_competing=sum(r['handoff_5s_success_competing'] for r in rs))
    ra,re=relapse('A'),relapse('E')
    reduction=re['rate'] is not None and ra['rate'] is not None and re['rate']<ra['rate']
    summary=dict(aggregates=aggregates,comparisons=comparisons,gates=gates,
        relapse={'A':ra,'E':re},repeated_failure_point_estimate_reduced=reduction,
        success_criteria_met=all(g['mean_pass'] for g in gates.values()) and reduction,
        all_lower_ci_gates_met=all(g['lower_ci_pass'] for g in gates.values()),
        natural60_percent_preference=index['E','natural_severe']['mean']>=.60,
        total_formal_episode_trials=total_trials,recorded_physical_interactions=physical_steps,
        frozen_model_seeds=5,training=False,rl_updates=0,phase16_automatically_authorized=False,
        code_manifest_sha=sha256(RESULTS/'code_manifest.json'),protocol_sha=sha256(RESULTS/'protocol.json'),
        selection_sha=sha256(RESULTS/'selection.json'),output_sha=output_sha)
    save_json(RESULTS/'summary.json',summary)
    for name,rows in [('per_seed.csv',per_seed),('aggregate.csv',aggregates),('stage.csv',stage_rows),('router.csv',router_rows),
            ('failure.csv',failures),('distance_bins.csv',bins),('paired.csv',comparisons),('timing.csv',timing),('deployment.csv',deployment)]:csv_file(name,rows)
    columns=['fixed','random','artificial_macro','natural_severe','full_loop']
    lines=['# Phase15.2 冻结模型恢复链报告','',
        '正式结果来自五组冻结 Normal/Recovery 模型；无训练、无在线 RL。A 是本轮新测试集上的 Phase15.1 最终系统。',
        '旧 Phase15.1 的自然恢复 35.25% 使用另一批起点，不能与新结果直接做配对提升。','',
        '|方法|固定|随机|人工宏平均|自然严重偏离|完整闭环|','|---|---|---|---|---|---|']
    for method in METHODS:lines.append('|'+method+'|'+'|'.join(pct(index[method,c]) for c in columns)+'|')
    lines+=['','区间为五个模型 seed 的 t 95% CI，未裁剪到 [0,1]。自然每个模型 seed 使用其独立源模型的100个新失败起点；每个 seed 内 A–E 配对。',
        '共享人工压力起点不提供独立环境 seed 置信度。五 seed 精确双侧 sign-flip 最小 p=0.0625，报告配对 t、精确检验和25终点 Holm 校正；不声称精确 p<0.05。','',
        '## 成功门槛','',f"预设完整候选 E 均值门槛和重复失败率点估计下降：{summary['success_criteria_met']}。下限 CI 门槛：{summary['all_lower_ci_gates_met']}。",
        '门槛未通过时不得进入 RL。即使通过，本流程也不会自动启动 Phase16。','',
        '## 关键问题与证据','',
        '1. Phase15.1 自然失败主要为撤离无单调安全出口、剩余时间不足、交还 Normal 后失败；本轮分阶段统计见 failure/stage CSV。',
        '2. 自然恢复新撤离收益：'+effect('B','A','natural_severe')+'。不把模型碰撞拒绝视为实测碰撞。',
        '3. 几何 pregrasp 与 progress guard 合并收益：'+effect('E','D','natural_severe')+'。不能单独因果归因于几何；几何候选仅由原 train split 决定。',
        '4. C 的有效抓取要求双指接触、位置/姿态/关节支持、保持与小幅探测；可靠性需结合后续失败，而不是只看抓取判定次数。',
        '5. 严格交还整体效果：'+effect('C','B','natural_severe')+'。状态分布误差可定位提前交还，不能单凭结果证明唯一原因。',
        '6. 平滑过渡收益：'+effect('D','C','natural_severe')+f"。A/E 交还5秒重新失败证据：{json.dumps(summary['relapse'],ensure_ascii=False)}。超时删失和成功竞争终点保留。",
        f"7. E 自然恢复 {pct(index['E','natural_severe'])}，>50%：{gates['natural_severe']['mean_pass']}，≥60%：{summary['natural60_percent_preference']}。",
        f"8. E 人工恢复 {pct(index['E','artificial_macro'])}，≥85%：{gates['artificial_macro']['mean_pass']}。",
        f"9. E 随机执行 {pct(index['E','random'])}，≥90%：{gates['random']['mean_pass']}。",
        f"10. E 完整闭环 {pct(index['E','full_loop'])}，均值≥90%：{gates['full_loop']['mean_pass']}，CI下限达标：{gates['full_loop']['lower_ci_pass']}。",
        f"11. 可靠随机 Anchor 证据门槛：{summary['success_criteria_met']}；仍需物理机器人验证。",
        '12. 不自动进入 Phase16；门槛未通过则继续分析恢复控制、时间预算、训练支持与子任务数据需求。',
        '13. 独占 GPU batch1 推理见 deployment.csv；router/planner/control 的并发批处理分位数见 timing.csv，不等同单机器人部署。','',
        '## 科学限制与协议','',
        '新自然起点由冻结 Phase15.1 B/15/15 策略在新 reset stream 实际失败后保存；10–24cm、120–420tick 内均匀 reservoir 取一条。自然距离分桶不足时保留空桶/低样本，不补人工代理。',
        'Normal/Recovery 训练 horizon10；执行/预测15使用真实较长 suffix 的外推，不做动作重复或 padding。总 episode600tick，恢复不会补时。',
        'Collision/IK 字段表示原几何模型的拒绝和有限候选求解结果；IK未找到不证明数学不可达，arm/wrist物理碰撞未装传感器则未知。',
        '原 cabinet collision OBB 导出不保证包含门板，门法向从既有 USD visual 薄轴和 rigid-body GT 取得，仅用于方向。碰撞判定仍按原模型，不能称完整场景碰撞保证。',
        'A/B Stage5 使用原 regrasp 判定作基准代理，C/D/E 才使用新增保持/探测/严格入口；不能把不同定义的 Stage5 率当同一分类器精度。Retreat joint/collision-free 比例是采样模型检查，非实测物理碰撞率。',
        '误触发只有阈值违例 proxy；没有无干预反事实不能识别真正误触发率。5秒失败率存在10秒预算删失，不能忽略分母。',
        '仿真等待推理；8GPU吞吐不能抵消单机器人控制周期延迟。候选动作模型安全检查不保证动态物理安全。','',
        f"正式 trial 数 {total_trials}。协议 SHA {summary['protocol_sha']}；validation选择 SHA {summary['selection_sha']}。"]
    (ROOT/'docs/phase15_2_recovery_chain_report.md').write_text('\n'.join(lines)+'\n')
    natural=[r for r in failures if r['test']=='natural_severe'];counts=defaultdict(Counter)
    for r in natural:counts[r['method']][r['category']]+=r['count']
    stage=['# Phase15.2 阶段失败诊断','',
        '逐步状态在 *.stage_trace.h5：每10tick、Router/阶段变化、终点记录；状态为动作前，终点/阶段为动作后。',
        '阶段率按 episode 和条件交集分别报告，允许 Recovery 路径跳过规划阶段；重新抓取不等于最终成功。','',
        '|方法|失败分类|数量|','|---|---|---|']
    for method in METHODS:
        for category,n in counts[method].items():stage.append(f'|{method}|{category}|{n}|')
    stage+=['','所有源样本、失败日志、原 collision/600tick 约束保留。详见 metrics/stage.csv、failure.csv、router.csv、distance_bins.csv。',
        '发生计划阶段超时、交还后停滞或 Normal 再次失败时，优先查看剩余预算、关节/相对姿态支持与阶段耗时。不能将规划失败归因于未执行的 Recovery 模型。']
    (ROOT/'docs/phase15_2_failure_stage_analysis.md').write_text('\n'.join(stage)+'\n')
    save_json(RESULTS/'analysis_provenance.json',dict(report_code_sha256=sha256(__file__),
        retreat_analysis_sha256=sha256(ROOT/'phase15_2/safe_retreat/retreat_analysis.py'),
        reports={name:sha256(ROOT/'docs'/name) for name in ['phase15_2_failure_stage_analysis.md','phase15_2_recovery_chain_report.md']},
        control_manifest_sha256=sha256(RESULTS/'code_manifest.json')))

if __name__=='__main__':main()
