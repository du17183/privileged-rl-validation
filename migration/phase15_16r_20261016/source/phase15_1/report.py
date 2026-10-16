"""Five frozen model seeds, paired tests, conditional stage denominators and limits."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import stats
from phase15_1.common import ROOT,RESULTS,save_json,sha256
from phase15_1.run import ART,GRADES,MAIN


def load(path):return json.loads(Path(path).read_text())


def ci(values):
    x=np.asarray(values,float);mean=float(x.mean());sd=float(x.std(ddof=1))
    half=float(stats.t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
    return dict(mean=mean,seed_sd=sd,ci95_low=mean-half,ci95_high=mean+half,seeds=len(x))


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    selected=load(RESULTS/'selection.json');rows=[];stage_rows=[];data={};files={}
    methods=['A','B','C','D','E_final']
    for method in methods:
        data[method]={}
        tests=MAIN+GRADES
        for seed in range(5):
            out=RESULTS/f'test/{method}_seed{seed}';per={}
            for test in tests:
                directory=Path(str(out)+'_grades') if test in GRADES else out
                path=directory/(test+'.json');result=load(path);tele=load(path.with_suffix('.telemetry.json'))
                cohort_manifest=load(RESULTS/'cohorts/test/manifest.json')
                expected=64 if test=='fixed' else 128 if test in ['random','full_loop'] else cohort_manifest[test]['episodes']
                if result['episodes']!=expected or len(tele)!=expected:raise RuntimeError('Formal episode coverage mismatch')
                if result.get('rl_updates')!=0:raise RuntimeError('Forbidden updates')
                files[str(path.relative_to(ROOT))]=sha256(path)
                files[str(path.with_suffix('.csv').relative_to(ROOT))]=sha256(path.with_suffix('.csv'))
                per[test]=result['success']
                rows.append(dict(method=method,seed=seed,test=test,episodes=expected,success=result['success'],
                    max_angle_rad=result['max_angle_rad'],final_angle_rad=result['final_angle_rad'],
                    interactions=result['interactions'],rl_updates=0))
                plans=[p for r in tele for p in r['planner']]
                stage=dict(method=method,seed=seed,test=test,episodes=expected,planner_attempts=len(plans),
                    episodes_with_planner=sum(bool(r['planner']) for r in tele),
                    safe_retreat_success=sum(bool(p.get('safe_retreat')) for p in plans),
                    pregrasp_success=sum(bool(p.get('reached_pregrasp')) for p in plans),
                    regrasp_success=sum(bool(p.get('regrasp')) for p in plans),
                    handbacks=sum(bool(p.get('handback')) for p in plans),
                    final_success_after_handback=sum(bool(p.get('handback')) and bool(p.get('final_task_success')) for p in plans),
                    final_success_episodes=sum(bool(r['success']) for r in tele),
                    progress_resumed_episodes=sum(any(e['flags'][3] for e in r['events']) for r in tele),
                    safe_retreat_applicable=result['inference'].get('controller',method)!='A')
                stage_rows.append(stage)
            per['artificial_macro']=float(np.mean([per[k] for k in ART]))
            data[method][str(seed)]=per
    metrics=[]
    for method in methods:
        for test in list(data[method]['0']):
            metric=dict(method=method,test=test,**ci([data[method][str(s)][test] for s in range(5)]))
            metrics.append(metric)
    paired=[]
    for a,b in [('B','A'),('C','B'),('D','C'),('E_final','A'),('E_final',selected['controller'])]:
        for test in ['fixed','random','artificial_macro','natural_severe','full_loop']:
            d=np.array([data[a][str(s)][test]-data[b][str(s)][test] for s in range(5)])
            observed=abs(float(d.mean()))
            perm=[abs(float((d*np.array(sign)).mean())) for sign in itertools.product([-1,1],repeat=5)]
            exact=float(np.mean(np.array(perm)>=observed-1e-12))
            t=stats.ttest_1samp(d,0).pvalue if np.std(d)>1e-12 else (1. if observed<1e-12 else 0.)
            paired.append(dict(comparison=a+'-'+b,test=test,**ci(d),paired_t_p=float(t),exact_signflip_p=exact))
    # One declared family: all listed paired comparisons across five main endpoints.
    ordered=sorted(range(len(paired)),key=lambda i:paired[i]['paired_t_p']);adjusted=0.
    for rank,i in enumerate(ordered):
        adjusted=max(adjusted,min(1.,paired[i]['paired_t_p']*(len(paired)-rank)))
        paired[i]['paired_t_holm_p']=adjusted
    write_csv(RESULTS/'per_seed_metrics.csv',rows);write_csv(RESULTS/'metrics.csv',metrics)
    write_csv(RESULTS/'stage_metrics.csv',stage_rows);write_csv(RESULTS/'paired_tests.csv',paired)
    final={m['test']:m for m in metrics if m['method']=='E_final'}
    point=dict(fixed=final['fixed']['mean']>=.95,random=final['random']['mean']>=.90,
        artificial=final['artificial_macro']['mean']>=.85,natural=final['natural_severe']['mean']>.50,
        preferred_natural=final['natural_severe']['mean']>=.60,full_loop=final['full_loop']['mean']>=.90)
    timing=[];deadline=[]
    for seed in range(5):
        out=RESULTS/f'test/E_final_seed{seed}'
        batch1=load(RESULTS/f'deployment_latency/seed{seed}.json')
        for module,measure in batch1.items():
            timing.append(dict(seed=seed,module=module,horizon=measure['horizon'],execute_ticks=measure['execute_ticks'],
                mean_model_s=measure['mean_model_s'],mean_rpc_s=measure['mean_rpc_s'],budget_s=measure['replan_budget_s'],
                budget_met=measure['budget_met']))
            deadline.append(bool(measure['budget_met']))
    write_csv(RESULTS/'timing.csv',timing)
    nominal=all(deadline)
    success_gate=all(point[k] for k in ['fixed','random','artificial','natural','full_loop'])
    report=ROOT/'docs/phase15_1_reacquire_latency_report.md'
    summary=dict(selection=selected,metrics=metrics,paired_tests=paired,point_thresholds=point,
        nominal_pi_replan_budget_met=nominal,task_success_gate=success_gate,
        real_robot_realtime_validated=False,phase16_allowed=False,
        phase16_reason='Requires task gates and measured planner/control deadlines; simulation pauses for inference and is not a real-time robot validation.',rl_updates=0)
    save_json(RESULTS/'summary.json',summary)
    lines=['# Phase15.1 Re-acquire 与推理时延报告','',
        '## 设计与冻结输入','',
        '冻结Phase15五个Normal与五个Recovery模型，仅修改重新接近控制器与推理工程/调度。没有BC、RL、SAC、Residual RL、LWD/DIVL或新训练数据。任务、reward、reset、600tick预算与原控制接口保持不变。',
        '先复现旧方法64个自然严重偏离状态；随后只用独立validation选择候选。正式test使用未见reset seed以及独立评测cohort。自然cohort由冻结Normal+Recovery生成，只有源episode实际失败后才保存，不能用人工扰动替代。',
        '人工四级距离每级至少100个独立起点，每种控制器再以五个冻结模型seed匹配评测。graded测试在相同物理handoff强制进入Re-acquire，隔离重新接近能力；自然及完整闭环测试保留实际Router决策。',
        f'锁定候选：`{json.dumps(selected,ensure_ascii=False)}`。','',
        '## 五seed正式结果','',
        '| 方法 | 指标 | 平均成功率 | 95% CI | seed SD |','|---|---|---:|---:|---:|']
    for row in metrics:
        lines.append(f'| {row["method"]} | {row["test"]} | {row["mean"]*100:.2f}% | [{row["ci95_low"]*100:.2f}, {row["ci95_high"]*100:.2f}]% | {row["seed_sd"]*100:.2f}pp |')
    lines+=['','A=旧GT Re-acquire；B=受检查的撤离/六维pregrasp/有效抓取直线路径；C=B+碰撞感知路径搜索；D=C+迟滞；E_final=validation锁定的控制器与优化推理调度。B对A同时改变多个控制环节，不能单独归因于Safe Retreat。','',
        'CI为五冻结模型seed的Student-t区间，表中不截断区间；它不是把全部episode当独立训练seed的区间。配对检验和Holm校正在paired_tests.csv。五seed双侧精确sign-flip最小p为0.0625，不能据此声称p<0.05。','',
        '## 阶段拆解','',
        'stage_metrics.csv同时记录episode分母、planner尝试次数、撤离/预抓取/重抓取/交还数量及交还后最终成功数量。每次触发可能与每个episode不同，不能混用分母。A没有显式Safe Retreat，其该项为不适用。碰撞模型拒绝与实际物理碰撞必须区分；未安装的臂/腕接触观测不能解释为零碰撞。','',
        '## Action chunk与实时性','',
        '5/10/15/20/30及不同Normal/Recovery调度均在validation评估。模型训练预测长度为10；15–30是实际更长suffix推理的外推，绝不重复末动作伪造预测。BF16和persistent worker是原始基线已有能力；state-only模型无图像流水线。',
        '精确state-bin文本缓存仅用于相同输入；每次更新state都重新计算prefix KV。CUDA Graph失败记录和编译数值差异记录保留。compile_approx为明确标记的近似BF16融合后端，<=0.01实际动作单位的误差仍必须通过匹配闭环成功率门槛，不能称为位一致推理。','',
        '| Seed | 模块 | 预测/执行长度 | Forward ms | RPC ms | 重规划预算 ms | 预算通过 |','|---:|---|---|---:|---:|---:|---|']
    for t in timing:
        lines.append(f'| {t["seed"]} | {t["module"]} | {t["horizon"]}/{t["execute_ticks"]} | {t["mean_model_s"]*1000:.2f} | {t["mean_rpc_s"]*1000:.2f} | {t["budget_s"]*1000:.2f} | {t["budget_met"]} |')
    lines+=['','吞吐调度允许每GPU四个独立32环境评测进程组，8卡配置上限1024环境；它是配置容量，不是实测最大FPS。原始同卡并发worker时延包含争用，不用于单机器人部署结论。Graph缓存预算每worker48GiB，超预算的新非batch1形状使用等价unroll并记录fallback。',
        '所有并发评测结束后，单机器人时延按batch=1、每GPU一个独占模型worker重新测量，不用8卡吞吐推算单机器人延迟。正式原始结果还保存每episode调用次数、状态阶段ticks、RPC/service时延、显存、deadline misses、端到端控制耗时；平均有效周期=实际活动控制ticks/60÷该episode模型调用次数，Planner阶段不调用π0.5。',
        '仿真等待推理且不推进物理时钟；满足某个chunk nominal预算仍不能等同真实机器人闭环实时通过。路径搜索与碰撞检查的CPU耗时、通信与实时GT采集也需要包含在部署预算中。','',
        '## 十三个问题','',
        '1. 旧严重偏离主要在起点拒绝；新方法的具体堵点见阶段计数及原始失败分类，不能用IK solver失败证明数学不可达。',
        '2. 撤离提供原方法没有的起点逃逸能力；是否足够由B/C/D撤离计数与最终成功证明。复合消融不能证明单一组件的独立因果效应。',
        '3. 六维pregrasp的效果见到达率、姿态误差与C−B路径搜索比较，未到达不能称问题解决。',
        '4. 交还后是否继续打开由final_success_after_handback/handbacks单独回答，抓住不等于任务成功。',
        '5. 新旧提升值及95%配对CI见paired_tests.csv；负结果完整保留。',
        f'6. 自然严重偏离五seed均值={final["natural_severe"]["mean"]*100:.2f}%；>50%：{point["natural"]}，≥60%：{point["preferred_natural"]}。',
        f'7. 完整闭环均值={final["full_loop"]["mean"]*100:.2f}%；≥90%：{point["full_loop"]}。',
        f'8. validation选择执行长度={selected["schedule"]}；不能根据test再挑最佳。',
        '9. 各状态理论周期为执行ticks/60；实际episode平均周期与调用次数需从phase_ticks及pi_calls计算，包含Planner空调用阶段。',
        f'10. π0.5 nominal重规划预算通过={nominal}；真实闭环仍需验证，Planner/CPU与输入采集可能继续构成瓶颈。',
        f'11. 任务成功门槛全部通过={success_gate}。未通过就不能称可靠Randomized Anchor System。',
        '12. 本阶段不自动进入Phase16。先确认阶段门槛及真实端到端安全控制时间预算，未满足条件继续定位恢复/控制问题。',
        '13. 完整LWD/DIVL仍后置；冻结推理/重新抓取评估没有证明在线RL增量收益。','',
        '## 复现与证据','',
        '结果：phase15_1/results/metrics.csv、per_seed_metrics.csv、stage_metrics.csv、paired_tests.csv、timing.csv、summary.json；逐episode CSV、轨迹输入cohort、推理请求、控制器阶段和故障日志均保留。frozen_verification.json验证历史源码与权重未改。所有性能均来自真实Isaac Lab物理执行，不使用训练loss代替成功率。']
    revision_note=RESULTS/'code_versions/v9/throughput_revision.md'
    if revision_note.exists():lines+=['',revision_note.read_text(encoding='utf-8')]
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(RESULTS/'delivery_verification.json',dict(passed=True,formal_metrics=len(rows),aggregate_metrics=len(metrics),
        stage_rows=len(stage_rows),paired_tests=len(paired),report=str(report),report_sha256=sha256(report),
        formal_input_and_output_sha256=files,rl_updates=0))


if __name__=='__main__':main()
