"""Finalize the injection report with natural-state checks and diagnostics."""
import argparse,csv,json,math,sys,hashlib,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,h5py,pinocchio as pin
from recovery_expert.failure_classifier import classify,LABELS
ROOT=Path(__file__).resolve().parents[2]


def wilson(success,n):
    z=1.959963984540054;p=success/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-r,c+r]


def multiply(a,b):
    w,x,y,z=np.moveaxis(a,-1,0);v,i,j,k=np.moveaxis(b,-1,0)
    return np.stack((w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v),-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='formal_v1');a=p.parse_args()
    result=ROOT/'results/phase14_1_recovery_generation'/a.run_id
    data=ROOT/'datasets/recovery_expert/phase14_1'/a.run_id
    natural=ROOT/'datasets/recovery_expert/phase14_1/natural_crosscheck_v1'
    qualification=json.loads((result/'qualification.json').read_text())
    formal_analysis=json.loads((result/'failure_analysis.json').read_text())
    manifest=json.loads((result/'data_audit.json').read_text())
    summaries={case:json.loads((data/case/'summary.json').read_text()) for case in ['ee_offset','contact_loss','door_regression','stagnation']}
    with (natural/'attempts.csv').open() as f:natural_rows=[r for r in csv.DictReader(f) if int(r['triggered'])]
    natural_summary=json.loads((natural/'summary.json').read_text())
    groups=defaultdict(lambda:[0,0])
    for r in natural_rows:groups[r['trigger']][0]+=int(r['success']);groups[r['trigger']][1]+=1
    natural_audit=[];natural_analysis=[];natural_labels=0
    base=ROOT/'datasets/random_door_expert';keys=json.loads((base/'split_v1/split.json').read_text())['splits']['train']
    with h5py.File(base/'collection_v1/trajectories.h5') as h:
        refs=[]
        for key in keys:
            g=h[key];idx=np.flatnonzero(g['planner_phase'][:].flatten()==0)[-1];refs.append(g['observation'][idx,21:25])
    refs=np.asarray(refs);refs=np.where((refs*refs[:1]).sum(-1,keepdims=True)<0,-refs,refs)
    ref=refs.mean(0);ref/=np.linalg.norm(ref)
    urdf=Path(sys.prefix)/'lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf'
    model=pin.buildModelFromUrdf(str(urdf));idx=[model.joints[model.getJointId(f'panda_joint{i}')].idx_q for i in range(1,8)]
    lower=model.lowerPositionLimit[idx];upper=model.upperPositionLimit[idx]
    for name in ['trajectories.h5','failed_trajectories.h5']:
        path=natural/name;manifest['hashes'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with h5py.File(path) as h:
            for key,g in h.items():
                robot=g['observation'][:];gt=g['state'][:];next_robot=g['next_observation'][:];next_gt=g['next_state'][:]
                count=len(robot);natural_labels+=count
                for field,nextfield in [('observation','next_observation'),('environment_state','next_environment_state'),('state','next_state')]:
                    if count>1 and not np.allclose(g[nextfield][:-1],g[field][1:],atol=1e-5):natural_audit.append(f'{name}/{key}: discontinuity {field}')
                if not g['done'][-1,0]:natural_audit.append(f'{name}/{key}: nonterminal')
                if name!='failed_trajectories.h5':continue
                phase=g['planner_phase'][:].reshape(-1);angle=gt[:,0]
                normal=np.stack((np.cos(angle),np.sin(angle),np.zeros_like(angle)),-1)
                target=gt[:,2:5].copy();target[phase==0]=robot[0,18:21]
                free=np.isin(phase,[1,9]);target[free]-=.10*normal[free];target[free,2]+=.055
                target[phase==2]-=.06*normal[phase==2];target[np.isin(phase,[3,4])]+=.03*normal[np.isin(phase,[3,4])]
                yaw=np.stack((np.cos(angle/2),np.zeros_like(angle),np.zeros_like(angle),np.sin(angle/2)),-1)
                quat=multiply(yaw,np.broadcast_to(ref,yaw.shape));quat[np.isin(phase,[0,1])]=robot[0,21:25]
                quat[np.isin(phase,[5,6])]=robot[np.isin(phase,[5,6]),21:25]
                pe=np.linalg.norm(robot[:,18:21]-target,axis=-1)
                re=2*np.arccos(np.clip(np.abs((quat*robot[:,21:25]).sum(-1)),0,1))
                margin=np.minimum(robot[:,:7]-lower,upper-robot[:,:7]).min(-1)
                trace=dict(action=list(g['action'][:]),planner_phase=list(g['planner_phase'][:]),
                           position_error=list(pe[:,None]),rotation_error=list(re[:,None]),joint_margin=list(margin[:,None]),
                           contact_state=list(gt[:,9:11]>.5),episode_tick=list(g['episode_tick'][:]),next_state=list(next_gt))
                classification=classify(trace,False)
                classification.update(case='natural_'+str(g.attrs['trigger']),trajectory=key,
                                      eef_distance_start=float(np.linalg.norm(robot[0,18:21]-gt[0,2:5])),
                                      eef_distance_final=float(np.linalg.norm(next_robot[-1,18:21]-next_gt[-1,2:5])),
                                      joint_pos_start=robot[0,:7].tolist(),joint_pos_final=next_robot[-1,:7].tolist(),
                                      final_angle=float(next_gt[-1,0]),phase_ticks={str(i):int((phase==i).sum()) for i in np.unique(phase)},
                                      metric_note='Natural check diagnostics reconstructed from recorded state; margin uses bundled URDF hard limits. OPEN position error uses handle distance, not the unavailable hinge lead waypoint. Phase in original collector is after transition.')
                natural_analysis.append(classification)
    manifest.update(natural_labels=natural_labels,natural_audit_issues=natural_audit)
    (result/'data_audit.json').write_text(json.dumps(manifest,indent=2))
    (result/'natural_crosscheck.json').write_text(json.dumps(dict(summary=natural_summary,groups=groups,failures=natural_analysis,audit_issues=natural_audit),indent=2,ensure_ascii=False))
    qualification.update(overall_wilson95=wilson(qualification['success_count'],qualification['valid_attempts']),
                         phase14_2_scope='Controlled BC experiment using qualified injected-disturbance data; no generalized-anchor claim',
                         natural_recovery_qualified=False,natural_success=natural_summary['recovery_success_rate'],
                         natural_ee_success=groups['ee_offset'][0]/groups['ee_offset'][1])
    (result/'qualification.json').write_text(json.dumps(qualification,indent=2))
    perturb_rows={}
    for case in summaries:
        with (data/case/'attempts.csv').open() as f:perturb_rows[case]=list(csv.DictReader(f))
    extra=['','## 独立自然偏离交叉验证','',
           '保持同一冻结策略、原自然失败检测器和真实接管，32 clones每clone两次触发接管；无主动注入，无状态写入，无时钟reset。seed141301，64次自然恢复，与主动扰动正式集分别统计，不能合并成同一恢复分布。',
           '',f'自然恢复：**{natural_summary["recovery_successes"]}/64={natural_summary["recovery_success_rate"]:.2%}**，Wilson95% CI {natural_summary["recovery_95_wilson"][0]:.2%}–{natural_summary["recovery_95_wilson"][1]:.2%}。',
           '', '|自然触发类型|成功/尝试|','|---|---:|']
    extra.extend(f'|{case}|{s}/{n}|' for case,(s,n) in groups.items())
    extra.extend(['', '**5–20mm人工末端偏离200/200，并不等于严重自然偏离已解决：自然末端偏离0/5。** 新专家可为当前规定扰动自动提供成功纠正数据；整个自然访问状态分布的恢复能力未过90%。',
                  '', '历史Phase14 138/160与本轮自然49/64使用不同测试种子和clone配额，不能据此宣称新专家因果退化。',
                  '', '## 扰动生成率及真实幅度','', '|类型|有效/全部尝试|未开始扰动|已开始但无效|','|---|---:|---:|---:|'])
    for case,rows in perturb_rows.items():
        unreached=sum(not int(r['perturbation_started']) for r in rows)
        invalid=sum(int(r['perturbation_started']) and not int(r['valid_perturbation']) for r in rows)
        extra.append(f'|{case}|{summaries[case]["valid_attempts"]}/{len(rows)}|{unreached}|{invalid}|')
    extra.extend(['','末端有效条件：实测三维偏移距目标≤max(目标幅度×25%,1.5mm)；不是声称实际偏移精确等于标称值。','',
                  '|标称幅度|实测目标轴绝对偏移均值mm|最小–最大mm|','|---|---:|---:|'])
    for mm in [5,10,20]:
        rows=[r for r in perturb_rows['ee_offset'] if int(r['valid_perturbation']) and abs(abs(float(r['target_mm']))-mm)<.01]
        values=[abs(float(r[['actual_dx','actual_dy','actual_dz'][int(r['axis'])]]))*1000 for r in rows]
        extra.append(f'|{mm}|{np.mean(values):.2f}|{min(values):.2f}–{max(values):.2f}|')
    extra.extend(['','## 修复过程与因果边界','',
                  '|预检版本|末端|接触丢失|门角回退|停滞|','|---|---:|---:|---:|---:|'])
    for run in ['pilot_v2','pilot_v3','pilot_v4','pilot_v5']:
        cells=[]
        for case in summaries:
            s=json.loads((ROOT/'datasets/recovery_expert/phase14_1'/run/case/'summary.json').read_text());cells.append(f'{s["success_count"]}/{s["valid_attempts"]}')
        extra.append('|'+run+'|'+'|'.join(cells)+'|')
    extra.extend(['', 'v3缩短近把手状态的重复退让，并恢复原开门动作幅度；v4仅在专家指令中补偿已验证的工具Jacobian映射，原环境IK不变；v5把门角回退后的退让和姿态调整分开，先离开门板附近再转姿态。开发预检复用种子，不能作为独立统计提升证据。正式不同种子下门角回退85/100，仍是主动扰动最困难类型。',
                  '', '机器人全臂碰撞未被原传感器直接测量。靠近门板时位置/姿态追踪受阻与分阶段退让的结果支持接近路径问题这一诊断方向，但不证明所有失败由碰撞或Jacobian单独造成。',
                  '', '## 最终五个问题','',
                  '1. **哪些失败最难恢复？** 指定主动扰动中为门角回退85%；自然偏离中为末端偏离0/5（小样本）。',
                  '2. **为什么末端偏离曾失败？** 自然偏离包含与轻微笛卡尔扰动不同的完整关节姿态、接触历史和时间预算；现有局部恢复未解决这些状态。原工具映射误差已有几何证据，但补偿不能单独解释或解决全部失败。根因未完全确证。',
                  '3. **GT专家能否自动生成数据？** 能，无人工遥操作，正式500有效尝试得到484成功纠正轨迹，所有16失败也保留。补充自然验证49成功/15失败分别存储。',
                  '4. **数据是否达到BC要求？** 指定扰动正式总体96.8%、末端100%，数据连续性审计通过。只使用有效成功组作为成功纠正标签；无效注入、失败和前缀不能混入成功标签。自然状态覆盖仍不足。',
                  '5. **能否进入Phase14.2？** 可以开展限定数据分布的Robot-only/Robot+GT受控BC验证；不能宣称已有全分布恢复Anchor。把自然负例保留为独立压力测试，本阶段未训练BC/RL。',
                  '', f'总体Wilson95% CI：{qualification["overall_wilson95"][0]:.2%}–{qualification["overall_wilson95"][1]:.2%}。本报告是专家执行成功的episode统计，并非五个训练seed的策略提升统计。'])
    report=ROOT/'docs/phase14_1_recovery_generation_report.md'
    report.write_text(report.read_text()+'\n'.join(extra)+'\n');(result/'report.md').write_text(report.read_text())
    combined=formal_analysis+natural_analysis
    lines=['# Phase14.1恢复失败分析','',f'正式主动扰动失败{len(formal_analysis)}；补充自然恢复失败{len(natural_analysis)}。两组分别统计。','',
           '|分类|主动扰动|自然偏离|','|---|---:|---:|']
    fcount=Counter(x['code'] for x in formal_analysis);ncount=Counter(x['code'] for x in natural_analysis)
    lines.extend(f'|{label}|{fcount[code]}|{ncount[code]}|' for code,label in LABELS.items())
    lines.extend(['', '分类是证据支持的诊断标签，不是根因证明。局部追踪/限位风险不证明全局IK不可达；碰撞未测得，表中0不能解释为没有碰撞。自然记录的目标误差由状态重构，OPEN用到把手距离，关节余量用URDF硬限位；正式记录使用运行时目标与soft limits，二者不可直接混同。',
                  '', '|来源/轨迹|分类|接管tick|剩余tick|最终阶段|位置误差m|姿态误差rad|关节余量rad|OPEN步数|','|---|---|---:|---:|---:|---:|---:|---:|---:|'])
    for item in combined:
        e=item.get('evidence',{})
        if e:lines.append(f'|{item["case"]}/{item["trajectory"]}|{item["label"]}|{e["takeover_tick"]}|{e["remaining_ticks"]}|{e["final_phase"]}|{e["final_position_error"]:.4f}|{e["final_rotation_error"]:.4f}|{e["min_joint_margin"]:.4f}|{e["open_ticks"]}|')
    lines.extend(['','## 自然末端偏离负例','', '|轨迹|起始EEF距离m|最终距离m|最终角rad|阶段步数|','|---|---:|---:|---:|---|'])
    for item in natural_analysis:
        if item['case']=='natural_ee_offset':lines.append(f'|{item["trajectory"]}|{item["eef_distance_start"]:.3f}|{item["eef_distance_final"]:.3f}|{item["final_angle"]:.3f}|{json.dumps(item["phase_ticks"])}|')
    (ROOT/'docs/recovery_failure_analysis.md').write_text('\n'.join(lines)+'\n')
    (result/'all_failure_analysis.json').write_text(json.dumps(combined,indent=2,ensure_ascii=False))
    print(json.dumps(dict(qualification=qualification,natural_audit_issues=natural_audit,formal_label_counts={k:v for k,v in manifest.items() if 'labels' in k})),flush=True)


if __name__=='__main__':main()
