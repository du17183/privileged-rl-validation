"""Quantify saved per-axis moments and observed runtime; no new interactions."""
import sys,json,csv,math,os
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,save,sha,utc

def read(p):return json.loads(p.read_text()) if p.exists() else None
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def write(p,data):
    if data:
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    summary=read(RES/'summary.json');assert summary['execution_finished']
    grouped={}
    for row in rows(RES/'action_distributions.csv'):
        if row['arm'] not in ['C','D'] or int(row['axis'])>=6:continue
        key=tuple(row[k] for k in ['arm','seed','step','scope','distribution','contact_scope','action_kind'])
        grouped.setdefault(key,[]).append(row)
    moments=[]
    for key,data in sorted(grouped.items()):
        assert len(data)==6 and len({v['n'] for v in data})==1
        arm,seed,step,scope,dist,contact,kind=key
        actual_step=int(step)
        if scope=='training':
            folder=RES/f'train/{arm}_seed{seed}'
            status=read(folder/'complete.json') or read(folder/'stopped.json') or read(folder/'technical_fault_preservation.json')
            actual_step=status['training_interactions']
        moments.append(dict(arm=arm,seed=int(seed),step=actual_step,scope=scope,distribution=dist,contact_scope=contact,action_kind=kind,n=int(data[0]['n']),normalized_L2_RMS=math.sqrt(sum(float(v['rms'])**2 for v in data)),normalized_centered_L2_RMS=math.sqrt(sum(float(v['std'])**2 for v in data)),normalized_mean_L2=math.sqrt(sum(float(v['mean'])**2 for v in data)),clip_fraction=float(data[0]['clip_fraction']),scope_note='six-coordinate vector moments reconstructed exactly from same-sample per-axis moments; no averaged percentile or physical TCP conversion'))
    write(RES/'action_vector_moments.csv',moments)
    throughput=[]
    for status in summary['trial_status']:
        folder=RES/f'train/{status["arm"]}_seed{status["seed"]}'
        real=read(folder/'complete.json') or read(folder/'stopped.json') or read(folder/'technical_fault_preservation.json')
        if not real:continue
        steps=real['training_interactions'];updates=real['optimizer_updates'];active=real.get('active_wall_s');update=real.get('update_wall_s')
        throughput.append(dict(arm=status['arm'],seed=status['seed'],actual_interactions=steps,updates=updates,active_wall_s=active,evaluation_pause_wall_s=real.get('evaluation_pause_wall_s'),update_wall_s=update,physics_step_wall_s=real.get('physics_step_wall_s'),interactions_per_active_second=steps/active if active else None,updates_per_update_second=updates/update if update else None,scope='aggregate16env actual interactions; native training active wall excludes validation pause, inclusive timers not additive'))
    write(RES/'training_throughput.csv',throughput)
    observed={};gpu_samples={i:[] for i in range(3)};loads=[];times=[]
    for p in sorted((RES/'monitor').glob('*_v3.json')):
        sample=read(p);times.append(sample['utc']);loads.append(sample['cpu_load'][0])
        for row in sample['processes']:
            if row.get('pid_reused') or 'cpu_user_ticks' not in row:continue
            key=(row['pid'],row['start_ticks'])
            ticks=row['cpu_user_ticks']+row['cpu_system_ticks']
            if key not in observed or ticks>observed[key]['last_observed_cpu_ticks']:
                observed[key]=dict(pid=row['pid'],start_ticks=row['start_ticks'],argv=row['argv'],last_observed_cpu_ticks=ticks,observed_at_utc=sample['utc'],scope='last observed cumulative process CPU ticks since birth; lower bound on exit total, includes all process threads, final CPU total unknown')
        for line in sample['gpu'].splitlines():
            index,free,util=line.split(',');i=int(index)
            if i in gpu_samples:gpu_samples[i].append(float(util.strip().rstrip('%').strip()))
    write(RES/'observed_process_cpu.csv',list(observed.values()))
    runtime=dict(utc=utc(),monitor_snapshots=len(times),first_snapshot=times[0] if times else None,last_snapshot=times[-1] if times else None,gpu_0_2_utilization_sample_mean={str(i):sum(v)/len(v) if v else None for i,v in gpu_samples.items()},load1_sample_mean=sum(loads)/len(loads) if loads else None,observed_cumulative_cpu_ticks=sum(v['last_observed_cpu_ticks'] for v in observed.values()),cpu_tick_hz=os.sysconf('SC_CLK_TCK'),final_all_process_cpu_seconds='unknown: sampled last observations are not exit counters',not_time_weighted=True,monitor_overhead_closed_loop='unknown: no matched instrumented/uninstrumented runtime experiment',physical_interactions_added=0,source_sha256=sha(Path(__file__)))
    save(RES/'runtime_scope.json',runtime)
    doc=ROOT/'docs/phase16_r_report.md';old=doc.read_text();folder=RES/'report_revisions'/('supplement_'+utc().replace(':',''));folder.mkdir(parents=True,exist_ok=False);(folder/doc.name).write_bytes(doc.read_bytes())
    lines=['','## 9. 实测动作分布与吞吐补充','',
        '下表为六维归一化残差向量的L2 RMS和相对base的实际变化L2 RMS；由同一批逐轴二阶矩精确合成，未将分位数平均。训练行使用实际完成预算，接触行按动作前双指接触状态筛选；相同公式仍可能遇到不同状态分布。','',
        '| 组/seed | scope/分布/预算 | 接触scope | n tick指令 | 残差L2 RMS | 实际变化L2 RMS | 裁剪比例 |','|---|---|---|---|---|---|---|']
    changes={(r['arm'],r['seed'],r['step'],r['scope'],r['distribution'],r['contact_scope']):r for r in moments if r['action_kind']=='actual_change'}
    for row in moments:
        if row['action_kind']!='residual' or row['scope'] not in ['training','random'] or row['contact_scope'] not in ['all','bilateral_contact']:continue
        if row['scope']=='random' and row['step']!=50000:continue
        key=tuple(row[k] for k in ['arm','seed','step','scope','distribution','contact_scope']);change=changes[key]
        lines.append(f"| {row['arm']} seed{row['seed']} | {row['scope']}/{row['distribution']}/{row['step']} | {row['contact_scope']} | {row['n']} | {row['normalized_L2_RMS']:.6f} | {change['normalized_L2_RMS']:.6f} | {100*row['clip_fraction']:.4f}% |")
    lines+=['','base和executed的向量均值/波动及所有逐轴尾部见action_vector_moments.csv与action_distributions.csv。裁剪统计为当前采样指令的六坐标clip元素比例；实际末端移动和全环境碰撞安全不由归一化指令推出。', '', '| 组/seed | 实际交互/s活动时间 | 更新/s更新时间 | 活动wall s | 评估等待s | 更新inclusive s | 物理step inclusive s |','|---|---|---|---|---|---|---|']
    for r in throughput:lines.append(f"| {r['arm']} seed{r['seed']} | {r['interactions_per_active_second']} | {r['updates_per_update_second']} | {r['active_wall_s']} | {r['evaluation_pause_wall_s']} | {r['update_wall_s']} | {r['physics_step_wall_s']} |")
    lines+=['',f"GPU0–2利用率采样均值（不作时间加权、包含训练等待与评估）为 {runtime['gpu_0_2_utilization_sample_mean']}%。CPU load1采样均值为 {runtime['load1_sample_mean']}；各拥有进程最后一次/proc累计CPU观测见observed_process_cpu.csv，是最终CPU成本的下界，不能冒称退出总计。观测窗口 {runtime['first_snapshot']} 至 {runtime['last_snapshot']}，{runtime['monitor_snapshots']}个快照。",'',
        '调度工程修复完整归档于source_versions/supervision011_selected_gpu：旧CPU调度器把其他GPU的任何项目占用也作为阻塞条件。仅对完整身份核验的本任务CPU调度器发送SIGINT，原helper自然退出；四个seed0训练/推理进程保持原PID/出生时刻/完整argv并自然重挂PPID1，新调度器接管等待和选卡。未向任何原生worker或其他项目进程发送信号，未reset训练状态、chunk、Replay或优化器。后续调度只检查选用GPU无外来compute且free≥65GB，最多3pair不变，其他GPU项目完整记录并保留。被接管worker的原父进程exit code不可获得，明确为unknown；保存实际进程退出、worker_complete及完整训练状态作为退出证据。',
        'D−C隔离的是同GT输入下有符号进度奖励替换的贡献；本轮没有无GT输入同数据/同预算训练组，不能把该差值解释为全部工装GT输入收益。三seed结果、最佳模型、均值执行及没有退化均不构成可靠Anchor或真机可迁移性证明。','']
    doc.write_text(old+'\n'.join(lines),encoding='utf-8')
    status=read(RES/'report_status.json');status.update(report_sha256=sha(doc),audit_pending=True,supplement_source_sha256=sha(Path(__file__)));save(RES/'report_status.json',status)
    print(json.dumps(dict(action_vector_moments=len(moments),training_throughput=throughput,report_sha256=sha(doc),physical_interactions_added=0)))
if __name__=='__main__':main()
