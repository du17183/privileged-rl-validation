"""Independent postprocessing of closed cases; no historical or worker writes.

One model seed: state intervals and paired state tests are exploratory. They
are not five-model-seed confidence intervals or formal generalization claims.
"""
import ast, csv, hashlib, json, math, time, fcntl
from collections import Counter, defaultdict
from pathlib import Path
import h5py
import numpy as np

R=Path.cwd(); E=R/'phase15_3/experiments'; O=R/'phase15_3/results/controlled_analysis'
O.mkdir(parents=True,exist_ok=True)
analysis_lock=(O/'.analysis.lock').open('a')
fcntl.flock(analysis_lock,fcntl.LOCK_EX)
A=json.loads((R/'phase15_3/results/phase15_2_baseline_audit.json').read_text())
assert A['passed']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def csvout(name,rows):
    if not rows:return
    p=O/name;tmp=p.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in rows for k in x)))
        w.writeheader();w.writerows(rows)
    tmp.replace(p)
def extract(path,names,scope):
    assert sha(path)==A['sha256'][str(path.relative_to(R))]
    t=ast.parse(path.read_text());nodes=[]
    for n in t.body:
        if isinstance(n,ast.FunctionDef) and n.name in names:nodes.append(n)
        elif isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id in names for v in n.targets):nodes.append(n)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'[read-only functions]','exec'),scope)
legacy={"np":np,"Counter":Counter,"math":math}
extract(R/'phase15_2/results/code_versions/recovery_funnel_v1/recovery_funnel.py',{'BINS','facts','ordered','relapse','bin_index'},legacy)
expanded={"np":np}
extract(R/'phase15_2/results/code_versions/final_delivery_v1/final_evidence_v2.py',{'ST','TAX','taxonomy'},expanded)
ST=expanded['ST']; input_sha={}; cases={}; episode_rows=[]; stage_rows=[]; motion_rows=[]

def wilson(k,n):
    if not n:return None,None
    z=1.959963984540054;c=(k/n+z*z/(2*n))/(1+z*z/n)
    h=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0,c-h),min(1,c+h)

def process(name,d,budget,motion_path,arm=None):
    p=d/'natural_severe.json'
    if not p.exists():return
    result=read(p);tele=read(d/'natural_severe.telemetry.json')
    assert result['episodes']==len(tele) and result['rl_updates']==0
    assert result['episode_budget_ticks']==budget
    for role,path in result['checkpoints'].items():
        rel=str(Path(path).relative_to(R));assert result['checkpoint_sha256'][role]==A['sha256'][rel]
    cohort=Path(result['cohort']);assert sha(cohort)==result['cohort_sha256']
    for f in [p,d/'natural_severe.telemetry.json',d/'natural_severe.stage_trace.h5',d/'natural_severe.initial.npz',cohort]:input_sha[str(f.relative_to(R))]=sha(f)
    with h5py.File(cohort) as h:
        attrs={k:dict(h[k].attrs) for k in h}
        assert all(x['natural_source_episode_failed'] and not x['training_labels'] for x in attrs.values())
    with h5py.File(d/'natural_severe.stage_trace.h5') as h:trace=h['steps'][:]
    grouped=defaultdict(list)
    for t in trace:grouped[int(t['clone']),int(t['episode'])].append(t)
    seen=set(); facts={}; rows=[]
    for t in tele:
        key=t['cohort_id'];assert key not in seen;seen.add(key)
        assert t['original_episode_ticks']<=budget
        samples=np.array(grouped[t['clone'],t['episode']],dtype=trace.dtype)
        assert len(samples)>0
        old=legacy['facts'](t); entries=t.get('normal_entries',[])
        takeover=any(np.any((samples['router']==0)&(samples['tick']>=n['tick']-1)) for n in entries) if entries else False
        trigger=any(x.get('route')=='reacquire' and x.get('switch_correct') for x in t['events'])
        vals=[True,trigger,*old[1:8],old[8],bool(takeover),old[9],old[10]]
        assert len(vals)==13
        primary,evidence=expanded['taxonomy'](t,samples);rc=legacy['relapse'](t)
        a=attrs[key];last=t['planner'][-1] if t['planner'] else {}
        row=dict(case=name,cohort_id=key,model_seed=result['model_seed'],distance_m=float(a['distance']),distance_bin=legacy['bin_index'](a['distance']),
            source_reset_seed=int(a['reset_seed']),source_clone=int(a['clone']),source_episode=int(a['source_episode']),
            total_budget_ticks=budget,original_episode_ticks=t['original_episode_ticks'],primary=primary,primary_evidence=evidence,
            attempts=len(t['planner']),last_trigger_tick=last.get('trigger_tick'),last_failure_stage=last.get('failure_stage'),
            first_normal_tick=entries[0]['tick'] if entries else None,last_normal_tick=entries[-1]['tick'] if entries else None,
            remaining_at_last_normal=budget-entries[-1]['tick'] if entries else None,final_angle_rad=t['final_angle_rad'],
            normal_5s_events=rc['events'],normal_5s_observed=rc['observed'],normal_5s_censored=rc['censored'],normal_5s_success_competing=rc['success_competing'],
            sampled_arm_joint_violation=bool(np.any(samples['joint_margin'][:,:7]<0)),
            strict_planner_to_normal=bool(any(p.get('valid_grasp') and p.get('probe_survived') and p.get('handback') for p in t['planner']) and entries),
            actual_arm_wrist_collision='unknown',state_interval_scope='exploratory state proportion; one model seed',five_model_seed_sd=None,five_model_seed_ci=None)
        row.update(zip(ST,vals));row.update(zip(['ordered_'+s for s in ST],legacy['ordered'](vals)));episode_rows.append(row);rows.append(row);facts[key]=row
    assert seen==set(attrs)
    for bucket in [-1,0,1,2,3,4]:
        sel=[x for x in rows if bucket<0 or x['distance_bin']==bucket]
        for mode in ['raw','ordered']:
            for i,s in enumerate(ST):
                field=('ordered_' if mode=='ordered' else '')+s;prev=('ordered_' if mode=='ordered' else '')+ST[max(0,i-1)]
                entered=[x for x in sel if i==0 or x[prev] is True];known=[x for x in entered if x[field] is not None]
                k=sum(x[field] is True for x in known);n=sum(x[field] is True for x in sel);lo,hi=wilson(n,len(sel));cl,ch=wilson(k,len(known))
                stage_rows.append(dict(case=name,distance_bin=bucket,mode=mode,stage=s,source_n=len(sel),entered_n=len(entered),conditional_known_n=len(known),success_joint_n=k,coverage_n=n,
                    unknown=sum(x[field] is None for x in sel),conditional_rate=k/len(known) if known else None,cumulative_rate=n/len(sel) if sel else None,
                    exploratory_state_wilson95_low=lo,exploratory_state_wilson95_high=hi,conditional_state_wilson95_low=cl,conditional_state_wilson95_high=ch,
                    model_seed_count=1,five_model_seed_sd=None,five_model_seed_ci=None))
    if motion_path.exists():
        # A closed arm can share a stream with the next active arm. Freeze only
        # complete newline records for the closed arm before deriving metrics.
        raw=motion_path.read_bytes();raw=raw[:raw.rfind(b'\n')+1]
        lines=[x for x in raw.splitlines() if x];motion=[json.loads(x) for x in lines]
        if arm is not None:
            selected=[b for b,x in zip(lines,motion) if x.get('arm')==arm]
            motion=[x for x in motion if x.get('arm')==arm]
        else:selected=lines
        saved=O/(name+'_closed_motion.jsonl');saved.write_bytes(b'\n'.join(selected)+b'\n')
        input_sha[str(saved.relative_to(R))]=sha(saved)
        # Cartesian command norm versus measured motion, not a claim that the
        # kinematic prediction equals the PD-controlled one-tick displacement.
        for phase in sorted(set(str(x['phase']) for x in motion)):
            mm=[x for x in motion if str(x['phase'])==phase];dq=[];tp=[];ta=[];cmd=[];actual=[];contacts=[]
            for x in mm:
                dq.append(max(abs(np.array(x['q_post'])[:7]-np.array(x['q_pre'])[:7])))
                cmd.append(float(np.linalg.norm(np.array(x['action'])[:3]*.05)))
                actual.append(float(np.linalg.norm(np.array(x['eef_post'])[:3]-np.array(x['eef_pre'])[:3])))
                contacts.append(bool(np.all(np.array(x['contact_post'])>.5)))
                if x.get('target_q') is not None:
                    tp.append(float(np.linalg.norm(np.array(x['target_q'])[:7]-np.array(x['q_pre'])[:7])))
                    ta.append(float(np.linalg.norm(np.array(x['target_q'])[:7]-np.array(x['q_post'])[:7])))
            motion_rows.append(dict(case=name,phase_after_control_before_physics=phase,count=len(mm),cartesian_command_norm_m_median=float(np.median(cmd)),
                actual_eef_step_m_median=float(np.median(actual)),arm_actual_step_linf_rad_p95=float(np.quantile(dq,.95)),
                joint_waypoint_error_before_median_rad=float(np.median(tp)) if tp else None,joint_waypoint_error_after_median_rad=float(np.median(ta)) if ta else None,
                joint_waypoint_step_reduction_median_rad=float(np.median(np.array(tp)-np.array(ta))) if tp else None,known_joint_target_samples=len(tp),
                bilateral_contact_post_count=sum(contacts),predicted_actuator_tracking_error='unknown; no Jacobian or PD target saved',actual_collision='unknown'))
    cases[name]=dict(results=d,facts=facts,initial=d/'natural_severe.initial.npz',cohort=cohort,result=result,motion_path=motion_path,arm=arm)

probe=E/'frozen600_seed1_first8_attempt003'
assert read(probe/'complete.json')['complete']
process('probe600',probe/'results',600,probe/'motion_steps.jsonl')
controlled=E/'controlled_dev16_seed1_v1'
for arm in ['frozen_E','GT_direct','GT_projected_retreat','GT_pose_pregrasp']:
    process(arm,controlled/arm/'results',600,controlled/'motion_steps.jsonl',arm)
timedir=E/'time600_900_1200_seed1_first8_v1'
for budget in [600,900,1200]:
    b=timedir/f'budget{budget}'
    if (b/'complete.json').exists():process('time'+str(budget),b/'results',budget,b/'motion_steps.jsonl')
confirmation=E/'confirmation_seed1_v1'
for arm in ['frozen_E','GT_direct','GT_projected_retreat','GT_pose_pregrasp']:
    process('confirm_'+arm,confirmation/arm/'results',600,confirmation/'motion_steps.jsonl',arm)

pairing=[];paired=[];trajectory=[]
for baseline,name in [('frozen_E','GT_direct'),('GT_direct','GT_projected_retreat'),('GT_direct','GT_pose_pregrasp'),('probe600','time600'),('time600','time900'),('time600','time1200'),
                      ('confirm_frozen_E','confirm_GT_direct'),('confirm_GT_direct','confirm_GT_projected_retreat'),('confirm_GT_direct','confirm_GT_pose_pregrasp')]:
    if baseline not in cases or name not in cases:continue
    ca,cb=cases[baseline],cases[name];za=np.load(ca['initial']);zb=np.load(cb['initial'])
    assert set(za['ids'])==set(zb['ids'])==set(ca['facts'])==set(cb['facts'])
    ids=list(za['ids']);order=[list(zb['ids']).index(x) for x in ids]
    robot=zb['robot'][order];tool=zb['state'][order];dr=abs(za['robot']-robot);dg=abs(za['state']-tool)
    exact=bool(np.array_equal(za['robot'],robot) and np.array_equal(za['state'],tool))
    contacts=bool(np.array_equal(za['state'][:,9:11]>.5,tool[:,9:11]>.5))
    clock=bool(np.array_equal(za['robot'][:,25],robot[:,25]))
    limits=dict(joint_linf_rad=.001,joint_velocity_linf_rad_s=.001,eef_position_m=.003,handle_position_m=.003,door_angle_rad=.0001,door_velocity_rad_s=.001)
    diffs=dict(joint_linf_rad=float(dr[:,:9].max()),joint_velocity_linf_rad_s=float(dr[:,9:18].max()),
        eef_position_m=float(np.linalg.norm(za['robot'][:,18:21]-robot[:,18:21],axis=1).max()),handle_position_m=float(np.linalg.norm(za['state'][:,2:5]-tool[:,2:5],axis=1).max()),
        door_angle_rad=float(dg[:,0].max()),door_velocity_rad_s=float(dg[:,1].max()))
    # No quaternion tolerance is needed to claim an exact match; for nonexact
    # starts we only report tolerances and require a separate pose review.
    passed=exact and contacts and clock
    record=dict(baseline=baseline,candidate=name,source_cohort_bytes_equal=sha(ca['cohort'])==sha(cb['cohort']),actual_initial_bit_exact=exact,contacts_equal=contacts,clock_equal=clock,
        measured_max_differences=diffs,diagnostic_tolerances=limits,passed=passed,contact_history_scope='same saved prefix and warmup; hidden PhysX contact cache not directly observed')
    pairing.append(record)
    if baseline.startswith(('probe','time')):
        def motions(c):
            rr=[json.loads(x) for x in c['motion_path'].read_text().splitlines()]
            return {(x['clone'],x['episode'],x['tick']):x for x in rr if x['tick']<600}
        ma,mb=motions(ca),motions(cb);common=set(ma)&set(mb)
        fields=['action','q_pre','q_post','eef_pre','eef_post','contact_pre','contact_post','door_post']
        dif={f:max(float(np.max(abs(np.array(ma[k][f])-np.array(mb[k][f])))) for k in common) if common else None for f in fields}
        divergence=min((k[2] for k in common if any(not np.array_equal(ma[k][f],mb[k][f]) for f in fields)),default=None)
        trajectory.append(dict(baseline=baseline,candidate=name,shared_pre600_motion_records=len(common),baseline_records=len(ma),candidate_records=len(mb),
            key_sets_equal=set(ma)==set(mb),max_absolute_difference_by_field=dif,first_observed_divergence_tick=divergence,
            scope='measured pre600 trajectory; extension may affect local available time and entry remaining-time check before600; only 600 replica is a no-change replication'))
    for endpoint in ['safe_pose','pregrasp_arrival','valid_grasp','handoff','door_progress','final_success']:
        pairs=[(ca['facts'][k][endpoint],cb['facts'][k][endpoint]) for k in ids]
        assert all(a is not None and b is not None for a,b in pairs)
        win=sum(not a and b for a,b in pairs);loss=sum(a and not b for a,b in pairs);nd=win+loss
        p=min(1.,2*sum(math.comb(nd,k) for k in range(min(win,loss)+1))/2**nd) if nd else 1.
        paired.append(dict(baseline=baseline,candidate=name,endpoint=endpoint,paired_states=len(ids),baseline_success=sum(a for a,b in pairs),candidate_success=sum(b for a,b in pairs),
            candidate_only=win,baseline_only=loss,delta_rate=(win-loss)/len(ids),exploratory_exact_mcnemar_p=p,actual_pairing_passed=passed,
            test_family='separate exploratory state tests, unadjusted; one model seed; no formal significance claim',total_budget_same=ca['result']['episode_budget_ticks']==cb['result']['episode_budget_ticks']))
csvout('episode_metrics.csv',episode_rows);csvout('stage_metrics.csv',stage_rows);csvout('motion_metrics.csv',motion_rows);csvout('paired_exploratory.csv',paired)
save(O/'initial_pairing.json',dict(pairs=pairing,closed_cases=list(cases),all_available_pairs_passed=all(x['passed'] for x in pairing),scope='only completed cases; absent comparisons remain pending'))
save(O/'time_trajectory_comparison.json',dict(comparisons=trajectory,source_sha256=sha(__file__)))
summary=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),closed_cases={k:dict(episodes=v['result']['episodes'],successes=sum(x['final_success'] for x in v['facts'].values()),budget=v['result']['episode_budget_ticks']) for k,v in cases.items()},
    controlled_complete=(controlled/'complete.json').exists(),time_complete=(timedir/'complete.json').exists(),confirmation_complete=(confirmation/'complete.json').exists(),pairing=pairing,input_sha256=input_sha,
    source_sha256=sha(__file__),model_seeds=1,exploratory=True,policy_training=False,GT_input_training_ablation='not performed',formal_statistics=False)
save(O/'analysis_scope.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ['input_sha256','pairing']},indent=2))
