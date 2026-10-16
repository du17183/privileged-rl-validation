"""Independent Phase15.3 completion audit. Never executes a physical worker.

Requires all scheduled controlled comparisons and original input revalidation.
Pending is explicitly distinct from passed. Does not certify a new Anchor.
"""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, math, sys, time
import h5py
import numpy as np

R=Path.cwd();P=R/'phase15_3';E=P/'experiments';O=P/'results'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proc(pid):
    p=Path('/proc')/str(pid)/'cmdline'
    return [x.decode() for x in p.read_bytes().split(b'\0') if x] if p.exists() else []
def save(name,x):
    p=O/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
names=['frozen600_seed1_first8_attempt003','new_natural_development_seed1','controlled_dev16_seed1_v1',
       'time600_900_1200_seed1_first8_v1','new_natural_confirmation_seed1','confirmation_seed1_v1']
supervisor_scripts=['run_probe.py','run_new_development.py','run_controlled.py','run_time.py','run_confirmation.py','run_confirmation_evaluation.py']
pending=[n for n in names if not (E/n/'complete.json').exists()]
if pending:
    result=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),passed=False,pending=pending,
        completion=False,reason='Physical comparisons have not all closed; no completion audit was claimed',training=False)
    save('completion_readiness.json',result);print(json.dumps(result,indent=2));sys.exit(0)

issues=[];refs={};exits=[]
def require(condition,description):
    if not condition:issues.append(description)
def reference(p):
    p=Path(p);refs[str(p.relative_to(R))]=sha(p);return refs[str(p.relative_to(R))]
baseline=read(O/'phase15_2_baseline_audit.json');require(baseline['passed'],'Historical baseline SHA audit failed')
require((time.time()-(O/'phase15_2_baseline_audit.json').stat().st_mtime)<1800,'Run original input revalidation after closing all physical jobs')
for role,expected in [('formal_tests',300),('formal_episodes',30500),('natural_validation',160),('natural_test',500),('natural_overlap',0)]:
    require(baseline[role]==expected,'Historical quota/split '+role)
for name,script in zip(names,supervisor_scripts):
    d=E/name;reference(d/'complete.json');l=read(d/'lifecycle.json');reference(d/'lifecycle.json')
    require(l['status']=='complete','Lifecycle incomplete '+name)
    for p in l['processes']:
        argv=proc(p['pid']);own=argv==p['command']
        require(not own,'Owned worker still live '+str(p['pid']));require(p['exit_code']==0,'Worker exit failure '+name)
        exits.append(dict(experiment=name,pid=p['pid'],role=p['role'],exit_code=p['exit_code'],owned_live=own,
            current_argv=argv,pid_reused=bool(argv and not own)))
    live=proc(l['supervisor_pid']);require(str(R/'phase15_3/experiments'/script) not in live,'Owned supervisor still live '+name)
source_sets={}
for split,reset in [('development',153301001),('confirmation',153401001)]:
    cohort=P/f'diagnostics/cohorts/{split}/seed1/natural_severe.h5';reference(cohort)
    identities=[]
    with h5py.File(cohort) as h:
        require(len(h)==16 and bool(h.attrs['evaluation_only']) and not bool(h.attrs['training_labels']),'New cohort scope '+split)
        for key in h:
            a=h[key].attrs
            require(bool(a['natural_source_episode_failed']) and not bool(a['training_labels']),'New source not a real failed evaluation episode')
            require(int(a['reset_seed'])==reset,'Locked reset stream changed '+split)
            require(.10<=float(a['distance'])<=.24 and 120<=int(a['handoff_tick'])<=420,'Natural source support changed '+split)
            identities.append(tuple(int(a[k]) for k in ['reference_model_seed','reset_seed','clone','source_episode']))
    require(len(set(identities))==16,'Duplicate new source identity '+split);source_sets[split]=set(identities)
require(not source_sets['development']&source_sets['confirmation'],'Development/confirmation overlap')
old=set()
for split in ['validation','test']:
    for seed in range(5):
        with h5py.File(R/f'phase15_2/results/cohorts/{split}/seed{seed}/natural_severe.h5') as h:
            for key in h:old.add(tuple(int(h[key].attrs[k]) for k in ['reference_model_seed','reset_seed','clone','source_episode']))
require(not old&(source_sets['development']|source_sets['confirmation']),'New sources overlap historical validation/test')

analysis=read(O/'controlled_analysis/analysis_scope.json');reference(O/'controlled_analysis/analysis_scope.json')
execution=read(E/'execution_manifest_v1.json');reference(E/'execution_manifest_v1.json')
for rel,want in execution['files'].items():require(sha(R/rel)==want['sha256'],'Frozen new execution input changed '+rel);refs[rel]=want['sha256']
require(analysis['controlled_complete'] and analysis['time_complete'] and analysis['confirmation_complete'],'Analysis scope incomplete')
require(analysis['model_seeds']==1 and analysis['exploratory'] and not analysis['formal_statistics'],'Exploratory seed/statistics scope changed')
for rel,want in analysis['input_sha256'].items():
    require(sha(R/rel)==want,'Closed evidence bytes changed '+rel);refs[rel]=want
initial=read(O/'controlled_analysis/initial_pairing.json');reference(O/'controlled_analysis/initial_pairing.json')
require(initial['all_available_pairs_passed'],'Actual paired initial observations failed')
ctrl=E/'controlled_dev16_seed1_v1';confirm=E/'confirmation_seed1_v1';lock=read(confirm/'candidate_lock.json');reference(confirm/'candidate_lock.json')
require(lock['confirmation_outcomes_observed'] is False and lock['formal_test_used_for_selection'] is False,'Selection used held-out outcomes')
require(lock['selection_protocol_sha256']=='45ab06a0c09e9ae682fdaf74e1fbdeb91aff2e0d920b1bee143e60a6f817175c','Selection protocol changed')
require(reference(E/'candidate_selection_protocol.json')==lock['selection_protocol_sha256'],'Protocol SHA differs from lock')
selected=lock['selected'];counts={}
for arm in ['GT_direct','GT_projected_retreat','GT_pose_pregrasp']:
    t=read(ctrl/arm/'results/natural_severe.telemetry.json')
    counts[arm]=[sum(bool(x['normal_entries']) for x in t),sum(x['chain_stage_first_tick'][5]>=0 for x in t),
        sum(any(p.get('reached_pregrasp') for p in x['planner']) for x in t),sum(any(p.get('safe_retreat') for p in x['planner']) for x in t)]
order=list(counts);expected=max(order,key=lambda a:tuple(counts[a])+(-order.index(a),))
require(selected==expected,'Candidate does not follow prespecified rank')
require((confirm/'candidate_lock.json').stat().st_mtime <= (confirm/'inference.log').stat().st_mtime,'Candidate lock not saved before confirmation inference started')

episode=list(csv.DictReader((O/'controlled_analysis/episode_metrics.csv').open()));stages=list(csv.DictReader((O/'controlled_analysis/stage_metrics.csv').open()))
for name,x in analysis['closed_cases'].items():
    rr=[r for r in episode if r['case']==name]
    require(len(rr)==x['episodes'] and len(set(r['cohort_id'] for r in rr))==len(rr),'Unique episode count '+name)
    require(sum(r['final_success']=='True' for r in rr)==x['successes'],'Final success count '+name)
    require(all(int(r['original_episode_ticks'])<=x['budget'] for r in rr),'Episode budget exceeded '+name)
    failure=Counter(r['primary'] for r in rr if r['final_success']=='False')
    require(sum(failure.values())+x['successes']==len(rr) and all(k in {'F'+str(i) for i in range(1,16)} for k in failure),'Exclusive primary partition '+name)
    for b in [-1,0,1,2,3,4]:
        subset=[r for r in rr if b<0 or int(r['distance_bin'])==b]
        for s in [v for v in stages if v['case']==name and int(v['distance_bin'])==b]:
            field=('ordered_' if s['mode']=='ordered' else '')+s['stage'];k=sum(r[field]=='True' for r in subset)
            require(int(s['source_n'])==len(subset) and int(s['coverage_n'])==k,'Independent stage/bin count '+name+' '+s['stage'])
            require(s['model_seed_count']=='1' and s['five_model_seed_sd']=='' and s['five_model_seed_ci']=='','One-seed result contains false five-seed variance')
            if not subset:require(s['cumulative_rate']=='','Empty distance bin represented as zero success')
    result_path=next(R/rel for rel in analysis['input_sha256'] if rel.endswith('/natural_severe.json') and
        ((name=='probe600' and 'frozen600_seed1_first8_attempt003' in rel) or
         (name.startswith('time') and '/budget'+name[4:]+'/' in rel) or
         (name.startswith('confirm_') and '/confirmation_seed1_v1/'+name[8:]+'/' in rel) or
         (not name.startswith(('time','confirm_')) and name!='probe600' and '/controlled_dev16_seed1_v1/'+name+'/' in rel)))
    result=read(result_path);require(result['rl_updates']==0 and result['diagnostic_only'] and result['phase15_3'],'New result namespace scope '+name)
    for role,path in result['checkpoints'].items():require(result['checkpoint_sha256'][role]==baseline['sha256'][str(Path(path).relative_to(R))],'Checkpoint mismatch '+name)

tc=read(O/'controlled_analysis/time_trajectory_comparison.json');reference(O/'controlled_analysis/time_trajectory_comparison.json')
require(len(tc['comparisons'])==3 and all(x['key_sets_equal'] and all(v==0 for v in x['max_absolute_difference_by_field'].values()) for x in tc['comparisons']),'Measured pre600 time trajectories differ')
h=read(O/'handoff_audit/scope.json');reference(O/'handoff_audit/scope.json')
expert_dev=h['summaries'][selected];expert_confirm=h['summaries']['confirm_'+selected]
expert_gate=expert_dev['observed_strict_first_handoff']>=13 and expert_confirm['observed_strict_first_handoff']>=13
observed_joints=[r for r in episode if r['case'] in [selected,'confirm_'+selected] and r['sampled_arm_joint_violation']=='True']
expert_gate=expert_gate and not observed_joints
decision=dict(selected=selected,development_handoff=expert_dev['observed_strict_first_handoff'],confirmation_handoff=expert_confirm['observed_strict_first_handoff'],
    gate_13_of16_each=expert_gate,observed_joint_violations=len(observed_joints),
    pure_planner_direct_dev=expert_dev['direct_planner_transition_normal'],pure_planner_direct_confirm=expert_confirm['direct_planner_transition_normal'],
    labels_or_training_authorized_by_this_gate=False,reason='Even a passed exploratory prerequisite still requires a separate data/action/expert-label chain audit; failed gate forbids labels/training')
save('expert_usability_audit.json',decision)
for p in [O/'controlled_analysis/stage_metrics.csv',O/'controlled_analysis/episode_metrics.csv',O/'controlled_analysis/paired_exploratory.csv',O/'handoff_audit/per_episode.csv',R/'docs/phase15_3_reacquire_report.md']:reference(p)
result=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),passed=not issues,issues=issues,
    physical_controlled_comparisons_complete=True,exploratory_model_seeds=1,new_development=16,new_confirmation=16,
    formal_phase15_2_unchanged=True,expert_usability=decision,exit_evidence=exits,input_sha256=refs,
    source_sha256=sha(Path(__file__)),new_anchor=False,phase16_allowed=False,online_rl=False,lwd_divl=False,
    scope='Completed controlled diagnostic experiment audit. Does not prove repaired severe recovery, learning/GT input gains, or a reliable Anchor.')
save('final_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','exit_evidence']},indent=2))
assert not issues
