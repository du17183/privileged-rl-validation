"""Retrospective simulated-tick budgets. CPU wall time is a separate clock."""
from pathlib import Path
import json,csv,hashlib,collections
import h5py,numpy as np
R=Path.cwd();P=R/'phase15_2/results';O=R/'phase15_3/results/retrospective';A=json.loads((R/'phase15_3/results/phase15_2_baseline_audit.json').read_text())
assert A['passed'];rows=[];attempts=[];sources={}
def verify(p):
 s=hashlib.sha256(p.read_bytes()).hexdigest();assert s==A['sha256'][str(p.relative_to(R))];sources[str(p.relative_to(R))]=s
for seed in range(5):
 cohort=P/f'cohorts/test/seed{seed}/natural_severe.h5';tele=P/f'test/E_seed{seed}/natural_severe.telemetry.json';verify(cohort);verify(tele)
 with h5py.File(cohort) as h:attrs={k:dict(h[k].attrs) for k in h}
 rr=json.loads(tele.read_text());assert len(rr)==100
 for r in rr:
  a=attrs[r['cohort_id']];entries=r['normal_entries'];first=entries[0]['tick'] if entries else None;last=entries[-1]['tick'] if entries else None
  rows.append(dict(model_seed=seed,cohort_id=r['cohort_id'],source_age_ticks=int(a['handoff_tick']),source_distance_m=float(a['distance']),
   original_budget=600,available_at_source_ticks=600-int(a['handoff_tick']),attempts=len(r['planner']),success=int(r['success']),
   first_normal_tick=first,last_normal_tick=last,remaining_at_first_normal=600-first if first is not None else None,
   remaining_at_last_normal=600-last if last is not None else None,source_to_first_normal_ticks=first-int(a['handoff_tick']) if first is not None else None,
   total_original_ticks=r['original_episode_ticks'],phase_ticks_normal=r['phase_ticks']['normal'],phase_ticks_reacquire=r['phase_ticks']['reacquire']))
  for j,p in enumerate(r['planner']):
   trigger=p['trigger_tick'];safe=p.get('retreat_tick');pre=p.get('pregrasp_tick');back=p.get('handback_tick')
   attempts.append(dict(model_seed=seed,cohort_id=r['cohort_id'],attempt=j,trigger_tick=trigger,remaining_at_trigger=600-trigger,
     safe_tick=safe,pregrasp_tick=pre,handback_tick=back,release_and_retreat_ticks=safe-trigger if safe is not None else None,
     safe_to_pregrasp_ticks=pre-safe if safe is not None and pre is not None else None,
     pregrasp_to_last_handback_ticks=back-pre if pre is not None and back is not None else None,
     last_handback_scope='may include subsequent Recovery/Normal re-entry; not pure grasp duration',failure_stage=p.get('failure_stage'),
     planning_success=p.get('planning_success'),safe_retreat=p.get('safe_retreat'),reached_pregrasp=p.get('reached_pregrasp'),success=r['success']))
def csvout(name,x):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
csvout('temporal_per_episode.csv',rows);csvout('temporal_per_attempt.csv',attempts)
bins=[]
for seed in range(5):
 for lo,hi in [(0,60),(60,120),(120,180),(180,240),(240,601)]:
  x=[r for r in rows if r['model_seed']==seed and r['remaining_at_first_normal'] is not None and lo<=r['remaining_at_first_normal']<hi]
  bins.append(dict(model_seed=seed,remaining_bin=f'[{lo},{hi})',normal_entry_episodes=len(x),final_successes=sum(r['success'] for r in x),
   success_rate=sum(r['success'] for r in x)/len(x) if x else None,scope='observational entry-time stratum, confounded by source difficulty and recovery history; not causal'))
csvout('temporal_entry_bins_per_seed.csv',bins)
summary=dict(episodes=len(rows),normal_entry_episodes=sum(r['first_normal_tick'] is not None for r in rows),source_sha256=sources,
 source_age_median=float(np.median([r['source_age_ticks'] for r in rows])),source_available_ticks_median=float(np.median([r['available_at_source_ticks'] for r in rows])),
 remaining_at_first_normal_median=float(np.median([r['remaining_at_first_normal'] for r in rows if r['remaining_at_first_normal'] is not None])),
 source_to_first_normal_median=float(np.median([r['source_to_first_normal_ticks'] for r in rows if r['source_to_first_normal_ticks'] is not None])),
 first_entry_less_than120_remaining=sum(r['remaining_at_first_normal'] is not None and r['remaining_at_first_normal']<120 for r in rows),
 remaining_at_last_normal_median=float(np.median([r['remaining_at_last_normal'] for r in rows if r['remaining_at_last_normal'] is not None])),
 causal_evidence='historical timing is descriptive; same-state600/900 intervention assessed separately',
 clock_warning='600tick is simulated episode time; CPU geometry wall-clock latency does not consume extra simulation ticks and cannot by itself explain a600tick terminal',
 model_seeds=5,history_modified=False,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(O/'temporal_scope.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='source_sha256'},indent=2))
