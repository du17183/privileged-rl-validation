"""Observe the actual strict planner-to-Normal sequence in closed motion files.

This is an evidence audit, not a controller change or a training-label exporter.
Contact runs are measured contact observations, not full force/collision safety.
"""
from pathlib import Path
from collections import defaultdict
import csv, json, hashlib, time
import h5py
import numpy as np

R=Path.cwd();A=R/'phase15_3/results/controlled_analysis';O=R/'phase15_3/results/handoff_audit'
O.mkdir(exist_ok=True);scope=json.loads((A/'analysis_scope.json').read_text())
rows=list(csv.DictReader((A/'episode_metrics.csv').open()));out=[];sha={};summaries={}
for case in scope['closed_cases']:
    motion=A/(case+'_closed_motion.jsonl')
    if not motion.exists():continue
    sha[str(motion.relative_to(R))]=hashlib.sha256(motion.read_bytes()).hexdigest()
    paths=[Path(p) for p in scope['input_sha256'] if p.endswith('/natural_severe.telemetry.json')]
    # Use the source result associated with a case via its exact frozen motion
    # and cohort facts; names are only output labels, not episode identities.
    if case=='probe600':root=R/'phase15_3/experiments/frozen600_seed1_first8_attempt003/results'
    elif case.startswith('time'):root=R/f'phase15_3/experiments/time600_900_1200_seed1_first8_v1/budget{case[4:]}/results'
    elif case.startswith('confirmation_'):root=R/'phase15_3/experiments/confirmation_seed1_v1'/case[len('confirmation_'):]/'results'
    else:root=R/'phase15_3/experiments/controlled_dev16_seed1_v1'/case/'results'
    tele_path=root/'natural_severe.telemetry.json';result=json.loads((root/'natural_severe.json').read_text());tele=json.loads(tele_path.read_text())
    assert hashlib.sha256(tele_path.read_bytes()).hexdigest()==scope['input_sha256'][str(tele_path.relative_to(R))]
    sha[str(tele_path.relative_to(R))]=hashlib.sha256(tele_path.read_bytes()).hexdigest()
    with h5py.File(result['cohort']) as h:source_age={k:int(h[k].attrs['handoff_tick']) for k in h}
    grouped=defaultdict(list)
    for line in motion.read_text().splitlines():
        x=json.loads(line);grouped[x['clone'],x['episode']].append(x)
    case_rows=[]
    for t in tele:
        mm=sorted(grouped[t['clone'],t['episode']],key=lambda x:x['tick']);by_tick={x['tick']:x for x in mm};entries=t['normal_entries'];first=entries[0] if entries else None
        record=dict(case=case,cohort_id=t['cohort_id'],source_age_ticks=source_age[t['cohort_id']],total_budget_ticks=result['episode_budget_ticks'],
            original_terminal_ticks=t['original_episode_ticks'],replay_active_sim_ticks=t['original_episode_ticks']-source_age[t['cohort_id']],
            first_normal_tick=first['tick'] if first else None,first_normal_remaining=result['episode_budget_ticks']-first['tick'] if first else None,
            source_to_first_normal_ticks=first['tick']-source_age[t['cohort_id']] if first else None,
            first_normal_action_observed=False,strict_ready_probe_sequence_observed=False,probe_bilateral_run_before_ready=None,
            normal_entry_handle_distance_m=None,first15_post_steps_observed=0,first15_contact_loss_observed=None,first6_net_door_angle_rad=None,
            success=bool(t['success']),actual_arm_wrist_collision='unknown',model_seeds=1)
        if first:
            entry=by_tick.get(first['tick']);record['first_normal_action_observed']=bool(entry and entry['router']==0)
            if entry:
                record['normal_entry_handle_distance_m']=float(np.linalg.norm(np.array(entry['eef_pre'])[:3]-np.array(entry['handle_pre'])[:3]))
                ready=[x for x in mm if x['tick']<first['tick'] and x['router']==2 and x['phase']=='ready']
                if ready:
                    z=ready[-1]['tick'];run=0;cursor=z
                    while cursor in by_tick and by_tick[cursor]['phase'] in ['probe','ready'] and np.all(np.array(by_tick[cursor]['contact_pre'])>.5):run+=1;cursor-=1
                    record['probe_bilateral_run_before_ready']=run
                    transition=[by_tick.get(k) for k in range(z+1,first['tick'])]
                    record['strict_ready_probe_sequence_observed']=bool(run>=12 and len(transition)==15 and all(x and x['router']==3 for x in transition)
                        and record['first_normal_action_observed'] and record['normal_entry_handle_distance_m']<.060
                        and any(p.get('valid_grasp') and p.get('probe_survived') for p in t['planner']))
                after=[by_tick[k] for k in range(first['tick'],first['tick']+15) if k in by_tick]
                record['first15_post_steps_observed']=len(after)
                record['first15_contact_loss_observed']=bool(any(not np.all(np.array(x['contact_post'])>.5) for x in after)) if after else None
                if first['tick']-1 in by_tick and first['tick']+5 in by_tick:
                    record['first6_net_door_angle_rad']=by_tick[first['tick']+5]['door_post']-by_tick[first['tick']-1]['door_post']
        out.append(record);case_rows.append(record)
    entered=[x for x in case_rows if x['first_normal_tick'] is not None]
    full15=[x for x in entered if x['first15_post_steps_observed']==15]
    summaries[case]=dict(episodes=len(case_rows),normal_entry_episodes=len(entered),observed_strict_first_handoff=sum(x['strict_ready_probe_sequence_observed'] for x in case_rows),
        first15_fully_observed=len(full15),first15_contact_loss=sum(x['first15_contact_loss_observed'] for x in full15),
        mean_active_sim_ticks=float(np.mean([x['replay_active_sim_ticks'] for x in case_rows])),
        mean_original_terminal_ticks=float(np.mean([x['original_terminal_ticks'] for x in case_rows])),
        median_source_to_first_normal_ticks=float(np.median([x['source_to_first_normal_ticks'] for x in entered])) if entered else None,
        mean_first_normal_remaining=float(np.mean([x['first_normal_remaining'] for x in entered])) if entered else None,
        total_simulated_clone_tick_interactions=json.loads((root/'natural_severe.json').read_text())['interactions'],
        per_episode_wall_seconds='unknown; shared32 control costs are not per-episode CPU time',model_seeds=1,five_seed_variance=None)
with (O/'per_episode.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(O/'scope.json').write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),summaries=summaries,input_sha256=sha,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_labels=False,
    sequence_definition='12 measured probe/ready bilateral pre-action observations,15 observed TRANSITION ticks, actual Normal action and handle distance<60mm; telemetry strict probe success',
    limitation='observational audit of closed logs; no force stability/complete physical collision sensor; first15 contact loss denominator excludes censored intervals; not a new policy or Anchor gate'),indent=2,allow_nan=False))
print(json.dumps(summaries,indent=2))
