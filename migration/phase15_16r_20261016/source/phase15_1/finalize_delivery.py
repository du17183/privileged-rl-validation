"""Append audited, quantitative conclusions and package readable delivery."""
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
from scipy import stats
from phase15_1 import final_audit

ROOT = Path('/home/xiaolong/privileged_rl_validation').resolve()
R = ROOT/'phase15_1/results'
MARKER = '## 独立审计后的最终结论'


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def writecsv(p, rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    a=load(R/'final_audit.json')
    assert a['passed'] and load(R/'experiment_complete.json')['complete']
    metrics=list(csv.DictReader((R/'metrics.csv').open()))
    by={(x['method'],x['test']):x for x in metrics}
    stages=list(csv.DictReader((R/'stage_metrics.csv').open()))
    si={(x['method'],int(x['seed']),x['test']):x for x in stages}
    conditional=[]
    groups=[(m,'natural_severe') for m in ['A','B','C','D','E_final']]+[('E_final',t) for t in ['grade_05','grade_10','grade_15','grade_20']]
    names=[('safe_retreat_success','planner_attempts'),('pregrasp_success','planner_attempts'),('regrasp_success','pregrasp_success'),('handbacks','planner_attempts'),('final_success_after_handback','handbacks'),('final_success_episodes','episodes')]
    for method,test in groups:
        for numerator,denominator in names:
            if method=='A' and numerator=='safe_retreat_success':
                continue
            entries=[si[(method,s,test)] for s in range(5)]
            den=sum(int(x[denominator]) for x in entries);num=sum(int(x[numerator]) for x in entries)
            rates=[int(x[numerator])/int(x[denominator]) for x in entries if int(x[denominator])]
            mean=float(np.mean(rates)) if rates else None
            sd=float(np.std(rates,ddof=1)) if len(rates)>1 else None
            half=float(stats.t.ppf(.975,len(rates)-1)*sd/np.sqrt(len(rates))) if len(rates)>1 else None
            conditional.append(dict(method=method,test=test,numerator=numerator,denominator=denominator,count=num,denominator_count=den,pooled_fraction=num/den if den else None,mean_seed_fraction=mean,seed_sd=sd,ci95_low=mean-half if half is not None else None,ci95_high=mean+half if half is not None else None,valid_seed_denominators=len(rates)))
    writecsv(R/'stage_conditional_metrics.csv',conditional)
    failure=[dict(category=k,count=v,total_failed_model_start_trials=518,fraction_failed=v/518,total_model_start_trials=800,fraction_all=v/800) for k,v in a['natural_terminal_failure_classes'].items()]
    writecsv(R/'natural_terminal_failure_classes.csv',failure)
    archive=R/'final_audit_revisions/generated_report'
    archive.mkdir(parents=True,exist_ok=True)
    docs=[ROOT/'docs/phase15_1_reacquire_latency_report.md',ROOT/'docs/phase15_1_severe_failure_analysis.md']
    for p in docs+[R/'delivery_verification.json']:
        target=archive/p.name
        if not target.exists():
            shutil.copy2(p,target)
    lines=[MARKER,'',
        '本阶段全部正式评测已完成，并通过独立原始结果审计；任务门槛未全部达成。审核包含300组方法×模型seed×测试条件、32,000次物理评测执行，65行五seed聚合指标、25项配对检验及Holm校正、10行独占GPU时延。这里的32,000是评测执行次数；不同模型重复使用共同cohort，并非32,000个独立训练seed或全部独立初态。',
        '40个冻结源码/权重文件重新计算SHA一致；18个共同validation/test cohort、26个源cohort、锁定selection、逐episode CSV/telemetry与统计已重算核验。自然validation/test各160个不同源episode，各由五源模型提供32个；自然源episode失败标签成立，分割身份无交集。全部压力测试终点完整，所有计费prefix+恢复预算≤600tick。审计细节见`phase15_1/results/final_audit.json`。','',
        '### 主要方法与正式结果','',
        '| 方法 | 固定 | 随机 | 四类人工失败宏均值 | 自然严重偏离 | 完整闭环 |',
        '|---|---:|---:|---:|---:|---:|']
    for m in ['A','B','C','D','E_final']:
        values=[float(by[(m,t)]['mean'])*100 for t in ['fixed','random','artificial_macro','natural_severe','full_loop']]
        lines.append('| '+m+' | '+' | '.join(f'{x:.2f}%' for x in values)+' |')
    lines+=['',
        'E_final是事前validation锁定的B控制器、15tick执行/15步真实预测、graph后端，正式test后没有重新选择。A/B/C/D对照使用原5tick执行、10步预测；B与E_final的差异同时涉及执行和预测长度，不能分离为单独的chunk因果效应。Graph数值及匹配物理保真检查通过；这里的成功率差异不能归因于“GPU加速改善学习”。',
        'E_final自然恢复35.25%（五模型seed 95% CI 33.46%–37.04%），未达到>50%或≥60%门槛。固定99.38%通过95%点估计门槛；随机88.75%未达到90%；人工四类宏均值72.40%未达到85%；完整闭环90.63%通过90%点估计门槛，但95% CI为83.60%–97.65%，不能据此声称真实成功率已被证明≥90%。总体task gate为False。',
        'B自然恢复31.00%，A为30.875%，配对变化仅+0.125pp，95% CI −2.47至+2.72pp，Holm p=1，不能声称受检查撤离整体显著提高自然恢复。E_final相对A为+4.375pp，配对95% CI +1.27至+7.48pp，原始t p=0.01735，25项Holm p=0.32961，精确sign-flip p=0.0625；不能声称校正后显著。相对同控制器B，E_final自然恢复+4.25pp（CI +3.60至+4.90pp，原始配对t p=0.00005391，Holm p=0.001294，精确p=0.0625）；参数t检验有证据，五seed精确检验不达0.05。两种检验分开报告，不能选择性只展示一个。',
        'E_final相对B随机成功率低3.59pp，完整闭环低5.00pp，两个差异的配对CI包含0、Holm p=1。这是方向性负结果。B随机92.34%、完整闭环95.63%的点估计较高；E_final在自然恢复和独占时延方面有优势，本阶段没有一个方案通过全部门槛。',
        '历史原seed0严重偏离2/64=3.125%仅是旧cohort诊断复现。它与新的独立自然cohort不是匹配对照，不能把3.125%→35.25%写成同测试集的提升。科学方法比较使用共同新test的A/B/C/D/E_final。','',
        '### 严重偏离与交还瓶颈','',
        'E_final在共同160个自然起点×五模型上执行800次，成功282、失败518。795次进入Re-acquire，撤离544/795=68.43%，到达pregrasp385/795=48.43%，有效重新抓取并交还370/795=46.54%，交还后最终成功278/370=75.14%。另有4次成功未经过Re-acquire交还。','',
        '| 失败终点分类 | 次数 | 占518次失败 |',
        '|---|---:|---:|']
    labels={'safe_retreat_no_monotone_escape':'找不到通过原安全检查的单调撤离','time_budget':'剩余600tick预算耗尽','handback_then_task_failure':'交还Normal后任务失败','checked_straight_path_not_found':'找不到通过检查的直线路径','no_reacquire_attempt':'没有Re-acquire接管'}
    for row in sorted(failure,key=lambda x:-x['count']):
        lines.append(f'| {labels.get(row["category"],row["category"])} | {row["count"]} | {row["fraction_failed"]*100:.2f}% |')
    lines+=['',
        '这些分类按每次失败episode最后一次planner尝试归类，互斥且总数518；它们不是五个独立样本集。没有安全撤离是算法/模型检查拒绝，不能改称实测物理碰撞或证明真实几何不可恢复。时间耗尽可能覆盖多个前序问题，不能单独证明需要放宽600tick。','',
        '| 人工偏离 | 到达pregrasp/500 | 有效抓取交还/500 | 最终成功/500 | 交还后成功率 |',
        '|---|---:|---:|---:|---:|']
    for t,label in [('grade_05','5cm'),('grade_10','10cm'),('grade_15','15cm'),('grade_20','20cm')]:
        s=a['aggregate_stages']['E_final/'+t]
        rate=f'{s["final_success_after_handback"]/s["handbacks"]*100:.2f}%' if s['handbacks'] else '不适用'
        lines.append(f'| {label} | {s["pregrasp_success"]} | {s["handbacks"]} | {s["final_success_episodes"]} | {rate} |')
    lines+=['',
        '10cm与15cm档位已经大多重新抓住，但交还后分别仅42/495与0/475完成开门；20cm主要卡在重新抓取。因而不能只看pregrasp或contact称恢复问题解决。交还状态、门角、接触姿态、剩余预算与Normal后续动作需要逐步区分；现有证据不能唯一判定是分布外交还、控制接口、姿态或预算原因。相应阶段条件率与五seed CI另存`stage_conditional_metrics.csv`；原stage_metrics.csv保留每episode/每尝试分母。','',
        '### 时延、吞吐与下一步边界','',
        '全部仿真进程结束后，五张独占GPU分别执行batch=1测试，每卡一个推理worker。Normal RPC均值50.17–51.43ms，Recovery 51.53–59.28ms；各组P95最高59.60ms，20个热身后计量样本中均没有超过250ms名义预算。每模块每seed先运行3次不计入热时延，冷启动不纳入这些均值。执行15tick=250ms，4Hz重规划；低于名义预算并未证明实时机器人部署成立。',
        'Planner start路径计算的原记录均值约1.95ms、P95约4.85ms；完整planner.action组调用墙钟耗时在每个当时活跃clone累计，含并发争用，自然恢复该累计字段每episode均值约37.95s。它不是独占单机器人planner时延，也不是单次start路径计算时延。RPC、Planner、GT采集、控制周期和硬件安全停机需要独占端到端测量，不能仅凭forward/RPC宣称部署可用。',
        '1024是调度配置容量。主评测完成后的空卡属任务收尾，不能从利用率推断仍需增大吞吐；本阶段没有测得硬件绝对最大FPS。严格torch.compile与compile_approx均有数值拒绝证据，未进入正式候选；使用的是验证通过的graph，保留回退等价unroll证据。',
        '建议下一阶段优先做两类小规模诊断：①对自然起点无法单调撤离的状态检查安全模型与物理接触，并在保持碰撞规则下验证可行出口；②针对10–15cm抓取交还后的失败，固定相同交还状态和原预算，记录Normal动作、接触保持与门角增量，区分交还兼容性和时间预算。随机初态仍未达90%门槛，保留B原调度与E_final分别作为成功率和时延对照。先修复恢复/交还闭环，再考虑新增训练或在线学习；本阶段不自动进入Phase16，更没有进入LWD/DIVL。','',
        '本次没有训练任何模型。实验完成标记表示评测与报告齐备，不表示恢复或真实部署目标已达成。']
    for p in docs:
        original=p.read_text(encoding='utf-8')
        prefix=original.split(MARKER)[0].rstrip()
        p.write_text(prefix+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    verification=load(R/'delivery_verification.json')
    verification.update(report_sha256=sha(docs[0]),severe_report_sha256=sha(docs[1]),independent_raw_audit_passed=True,generated_report_archive=str(archive),task_goals_met=False)
    (R/'delivery_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    final_audit.audit()
    files=docs+[R/n for n in ['metrics.csv','per_seed_metrics.csv','stage_metrics.csv','stage_conditional_metrics.csv','natural_terminal_failure_classes.csv','paired_tests.csv','timing.csv','summary.json','selection.json','engineering_selection.json','frozen_inputs.json','frozen_verification.json','delivery_verification.json','final_audit.json','experiment_complete.json','collection_plan_v9.json','collection_plan_v11.json']]
    files += list((R/'deployment_latency').glob('*.json'))
    files += [R/'cohorts/validation/manifest.json',R/'cohorts/test/manifest.json',R/'regression_v11/gate.json',R/'code_versions/v9/throughput_revision.md',ROOT/'phase15_1/final_audit.py',ROOT/'phase15_1/finalize_delivery.py']
    manifest=dict(files={str(p.relative_to(ROOT)):sha(p) for p in files},includes_weights=False,includes_full_raw_trajectories=False,remote_raw_evidence_retained=True,task_goals_met=False,phase16_allowed=False)
    mp=R/'local_delivery_manifest.json'
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files.append(mp)
    archive_path=R/'phase15_1_readable_delivery.tar.gz'
    with tarfile.open(archive_path,'w:gz') as tar:
        for p in files:
            tar.add(p,arcname=str(p.relative_to(ROOT)))
    print(json.dumps(dict(delivery=str(archive_path),sha256=sha(archive_path),files=len(files),bytes=archive_path.stat().st_size),ensure_ascii=True))


if __name__=='__main__':
    main()
