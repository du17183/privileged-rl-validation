"""Read formal natural sparse physics and planner histories with explicit limits."""
import csv,json,hashlib
from pathlib import Path
import numpy as np,h5py
R=Path.cwd();P=R/'phase15_2/results';O=R/'phase15_3/results/retrospective'
facts=list(csv.DictReader((P/'metrics/funnel_v2_episode.csv').open()));index={(x['method'],int(x['seed']),x['cohort_id']):x for x in facts if x['population']=='natural'}
out=[];attempts=[];shas={}
for method in 'ABCDE':
 for seed in range(5):
  base=P/f'test/{method}_seed{seed}';tp=base/'natural_severe.telemetry.json';hp=base/'natural_severe.stage_trace.h5'
  shas[str(tp.relative_to(R))]=hashlib.sha256(tp.read_bytes()).hexdigest();shas[str(hp.relative_to(R))]=hashlib.sha256(hp.read_bytes()).hexdigest()
  rows=json.loads(tp.read_text())
  with h5py.File(hp) as h:samples=h['steps'][:]
  groups={}
  for row in samples:groups.setdefault((int(row['clone']),int(row['episode'])),[]).append(row)
  for row in rows:
   key=(int(row['clone']),int(row['episode']));ss=np.array(groups.get(key,[]),dtype=samples.dtype)
   f=index[method,seed,row['cohort_id']];hist=row.get('planner',[])
   trigger=hist[-1]['trigger_tick'] if hist else None
   last=ss[ss['tick']>=trigger] if trigger is not None else ss
   for ai,m in enumerate(hist):
    checks=m.get('retreat_checks',[]);travel=[x['normal_travel_m'] for x in checks if 'normal_travel_m' in x]
    attempts.append(dict(method=method,seed=seed,cohort_id=row['cohort_id'],attempt=ai,trigger_tick=m['trigger_tick'],remaining_at_trigger=600-m['trigger_tick'],
      safe_pose=bool(m.get('safe_retreat')),plan=bool(m.get('planning_success')),arrival=bool(m.get('reached_pregrasp')),valid=bool(m.get('regrasp')),handoff=bool(m.get('handback')),
      failure_stage=m.get('failure_stage'),planner_failure=m.get('failure'),retreat_checks=len(checks),
      candidate_observed=sum(bool(x.get('escape_candidate_found')) for x in checks),model_clear_observed=sum(bool(x.get('collision_free_model')) for x in checks),
      normal_travel_first_m=travel[0] if travel else None,normal_travel_last_m=travel[-1] if travel else None,normal_travel_max_m=max(travel) if travel else None,
      retreat_tick=m.get('retreat_tick'),pregrasp_tick=m.get('pregrasp_tick'),handback_tick=m.get('handback_tick'),
      planning_not_attempted=m.get('planning_not_attempted',False),finite_ik_success=m.get('ik_success') if not m.get('planning_not_attempted',False) else None))
   def minfinite(a):
    a=np.asarray(a);a=a[np.isfinite(a)];return float(a.min()) if len(a) else None
   assessed=last['model_collision_assessed'] if len(last) else np.array([],bool)
   out.append(dict(method=method,seed=seed,cohort_id=row['cohort_id'],distance_m=f['distance_m'],distance_bin=f['distance_bin'],primary=f['primary'],
      success=int(row['success']),original_ticks=row['original_episode_ticks'],last_attempt_trigger_tick=trigger,remaining_at_last_trigger=600-trigger if trigger is not None else None,
      sparse_samples=len(last),model_assessed_samples=int(assessed.sum()),model_rejected_samples=int(last['model_collision_rejected'][assessed].sum()) if len(last) else 0,
      arm_margin_min_rad=minfinite(last['joint_margin'][:,:7]) if len(last) else None,finger_margin_min_m=minfinite(last['joint_margin'][:,7:]) if len(last) else None,
      target_position_error_min_m=minfinite(last['target_position_error']) if len(last) else None,target_orientation_error_min_rad=minfinite(last['target_orientation_error']) if len(last) else None,
      zero_arm_action_samples=int((np.max(np.abs(last['action'][:,:6]),axis=1)<1e-8).sum()) if len(last) else 0,
      actual_collision='unknown',measurement_scope='sparse pre-action state and post-action stage/terminal flags; waypoint may change; no per-step predicted-vs-actual residual recorded in formal data'))
assert len(out)==2500
for name,rr in [('mechanism_per_episode.csv',out),('planner_attempts.csv',attempts)]:
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
summary={}
for primary in ['F2','F6','F7','F11','F12','F15','success']:
 rr=[x for x in out if x['method']=='E' and x['primary']==primary];d=dict(n=len(rr),episodes_any_sampled_model_reject=sum(x['model_rejected_samples']>0 for x in rr),episodes_sampled_arm_margin_negative=sum(x['arm_margin_min_rad'] is not None and x['arm_margin_min_rad']<0 for x in rr))
 for field in ['remaining_at_last_trigger','target_position_error_min_m','target_orientation_error_min_rad']:
  v=[x[field] for x in rr if x[field] is not None];d[field]=dict(n=len(v),median=float(np.median(v)) if v else None,p95=float(np.percentile(v,95)) if v else None)
 summary[primary]=d
(O/'mechanism_scope.json').write_text(json.dumps(dict(formal_source_sha256=shas,E_summary=summary,training=False,actual_collision='unknown',finite_IK_failure_not_global_unreachability=True,sparse_target_error_not_direct_execution_residual=True),indent=2));print(json.dumps(summary,indent=2))
