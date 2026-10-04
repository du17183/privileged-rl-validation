"""Audit all attempts, assess the expert gate, and write evidence reports."""
import argparse,csv,json,hashlib
from collections import Counter
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    result=ROOT/'results/phase14_1_recovery_generation'/a.run_id
    data=ROOT/'datasets/recovery_expert/phase14_1'/a.run_id
    protocol=json.loads((result/'protocol.json').read_text());rows=[];summaries={};issues=[];analyses=[];hashes={};labels=0
    successful_labels=failed_labels=invalid_labels=0
    for case in protocol['targets']:
        summaries[case]=json.loads((data/case/'summary.json').read_text())
        with (data/case/'attempts.csv').open() as f:rows.extend(list(csv.DictReader(f)))
        for filename in ['trajectories.h5','failed_trajectories.h5']:
            path=data/case/filename;hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            with h5py.File(path) as h:
                for key,g in h.items():
                    if 'action' not in g:continue
                    count=len(g['action']);labels+=count
                    if filename=='trajectories.h5':successful_labels+=count
                    elif bool(g.attrs['valid_perturbation']):failed_labels+=count
                    else:invalid_labels+=count
                    for field in ['robot_state','environment_state','state','next_robot_state','next_environment_state','next_state','reward','done','door_angle','contact_state']:
                        if field not in g or len(g[field])!=count:issues.append(f'{case}/{filename}/{key}: {field} length mismatch')
                    for state,nextstate in [('robot_state','next_robot_state'),('environment_state','next_environment_state'),('state','next_state')]:
                        if count>1 and not np.allclose(g[nextstate][:-1],g[state][1:],atol=1e-5):issues.append(f'{case}/{key}: {state} discontinuity')
                    for field in ['action','robot_state','environment_state','next_state']:
                        if not np.isfinite(g[field][:]).all():issues.append(f'{case}/{key}: nonfinite {field}')
                    if not bool(g['done'][-1,0]):issues.append(f'{case}/{key}: missing terminal')
                    if filename=='failed_trajectories.h5' and bool(g.attrs['valid_perturbation']):
                        analyses.append(dict(case=case,trajectory=key,**json.loads(g.attrs['failure_analysis'])))
    n=sum(s['valid_attempts'] for s in summaries.values());success=sum(s['success_count'] for s in summaries.values())
    expert_pass=success/n>.9 and summaries['ee_offset']['success_rate']>.9
    qualification=dict(overall_success=success/n,success_count=success,valid_attempts=n,
                       ee_success=summaries['ee_offset']['success_rate'],expert_gate=expert_pass,
                       audit_issues=issues,ready_for_phase14_2=expert_pass and not issues,
                       stage=protocol['stage'],bc_executed=False,rl_executed=False,
                       limitation='Injected disturbances are distinct from natural severe deviations; no universal recovery claim. Initial door angle settles under inherited drive.')
    # A pilot never authorizes Phase14.2, even if its small sample succeeds.
    qualification['ready_for_phase14_2'] &= protocol['stage']=='formal'
    (result/'qualification.json').write_text(json.dumps(qualification,indent=2))
    (result/'data_audit.json').write_text(json.dumps(dict(expert_labels=labels,successful_labels=successful_labels,
        failed_recovery_labels=failed_labels,invalid_attempt_labels=invalid_labels,issues=issues,hashes=hashes),indent=2))
    (result/'failure_analysis.json').write_text(json.dumps(analyses,indent=2,ensure_ascii=False))
    failures=Counter(item['label'] for item in analyses)
    lines=['# Phase14.1 自动恢复专家生成报告','',f'运行：{a.run_id}；阶段：{protocol["stage"]}。','',
           '## 协议','',
           '当前冻结 Phase13 BC+GT 策略执行到预指定阶段，主动注入扰动，GT专家使用原 Cartesian/DLS 接口接管。机器人无瞬移、接管无reset、时钟不重置、原10秒episode与reward不变。门角回退是明确标记的30°→20°门关节状态注入，不作为恢复控制动作，也不混入专家标签。',
           '', '末端XYZ各±5/10/20mm，按18个桶平衡配额；报告目标与实际偏移。接触丢失为实际打开夹爪；停滞通过原接口持续非零旋转指令制造，并验证门角连续24步变化<0.002rad。未达到注入条件的尝试单独存储，不能作为成功恢复。',
           '', '## 实际结果','', '|类型|有效恢复成功/尝试|成功率|Wilson95% CI|无效/未触达尝试|','|---|---:|---:|---|---:|']
    for case,s in summaries.items():lines.append(f'|{case}|{s["success_count"]}/{s["valid_attempts"]}|{s["success_rate"]:.2%}|{s["wilson95"][0]:.2%}–{s["wilson95"][1]:.2%}|{s["invalid_attempts"]}|')
    lines.extend(['',f'总体：**{success}/{n}={success/n:.2%}**。专家门槛（总体与末端偏离均>90%）：**{expert_pass}**。',
                  '', '## 末端偏离分桶','', '|方向与幅度|成功/尝试|','|---|---:|'])
    for bucket,counts in summaries['ee_offset']['buckets'].items():lines.append(f'|{bucket}|{counts["success"]}/{counts["n"]}|')
    lines.extend(['','## 数据与后续资格','',f'有效成功恢复标签{successful_labels}条，失败恢复标签{failed_labels}条，无效注入后专家标签{invalid_labels}条；审计问题{len(issues)}。只有有效成功组可用于后续成功纠正BC，无效组不能冒充成功恢复。成功与失败HDF5分开，策略/注入前缀独立保存；原始action、next_state、done与reset参数可追溯。',
                  '',f'Phase14.2资格：**{qualification["ready_for_phase14_2"]}**。本阶段BC/RL均未训练。',
                  '', '## 结论与限制','',
                  '最难恢复类型以本轮分层结果判定。末端偏离负例通过真实抓取、目标误差、关节余量、阶段耗时分解。局部IK追踪失败不证明全局不可达；原传感器不能直接确认全机械臂碰撞，碰撞原因不虚构。',
                  '', '本结果只验证规定扰动和当前策略触达的状态。人工5–20mm偏移不能替代Phase14自然偏离测试。已有门驱动使初始门角回到关闭附近，不能宣称持续初角泛化。',
                  '', '失败详见 `docs/recovery_failure_analysis.md` 与本运行failure_analysis.json。所有历史结果保留。'])
    text='\n'.join(lines)+'\n';(result/'report.md').write_text(text)
    docs=ROOT/'docs';docs.mkdir(exist_ok=True)
    (docs/'phase14_1_recovery_generation_report.md').write_text(text)
    analysis=['# Phase14.1恢复失败分析','',f'运行{a.run_id}，有效恢复失败{len(analyses)}。','',
              '|分类|数量|','|---|---:|']
    analysis.extend(f'|{label}|{count}|' for label,count in failures.items())
    analysis.extend(['', '分类依据实际阶段、误差、关节余量及剩余时间；confidence=diagnostic是证据支持的候选机制，不是因果证明。未测得全臂碰撞，因此不把碰撞猜测写成已确认原因。',
                     '', '|类型/轨迹|分类|接管tick|剩余tick|最终阶段|位置误差m|姿态误差rad|最小关节余量rad|OPEN步数|',
                     '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
    for item in analyses:
        e=item.get('evidence',{})
        if e:analysis.append(f'|{item["case"]}/{item["trajectory"]}|{item["label"]}|{e["takeover_tick"]}|{e["remaining_ticks"]}|{e["final_phase"]}|{e["final_position_error"]:.4f}|{e["final_rotation_error"]:.4f}|{e["min_joint_margin"]:.4f}|{e["open_ticks"]}|')
    (docs/'recovery_failure_analysis.md').write_text('\n'.join(analysis)+'\n')
    print(json.dumps(qualification),flush=True)


if __name__=='__main__':main()
