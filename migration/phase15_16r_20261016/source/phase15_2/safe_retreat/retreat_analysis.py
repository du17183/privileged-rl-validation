"""Episode-level retreat and terminal-failure classification."""
import numpy as np

STAGES=['failure_trigger','safe_retreat','pregrasp','reapproach','contact','valid_grasp','handoff','progress_resumed','final_success']

def analyze(row):
    plans=row['planner'];stages=row.get('chain_stage_first_tick',[-1]*9)
    events=row['events'];entries=row.get('normal_entries',[])
    failed=not row['success'];last=plans[-1] if plans else {}
    handbacks=[e for e in events if e.get('returned_normal')]
    if not failed:category='success'
    elif handbacks and any(e['tick']>handbacks[-1]['return_tick'] for e in events):category='H_Normal_refailure'
    elif stages[7]>=0:category='J_episode_timeout_after_resumed_progress'
    elif handbacks:category='I_progress_not_resumed'
    elif last.get('failure_stage')=='time_budget':category='J_episode_timeout'
    elif str(last.get('failure_stage') or '').startswith('safe_retreat'):category='A_retreat_failed'
    elif last.get('start_collision') or 'collision' in str(last.get('failure_stage')):category='B_model_collision_rejection_physical_unknown'
    elif last and not last.get('safe_retreat'):category='A_retreat_failed'
    elif last and not last.get('ik_success') and not last.get('planning_not_attempted'):category='C_IK_solution_not_found'
    elif stages[2]<0:category='D_pregrasp_failed'
    elif stages[4]<0:category='E_recontact_failed'
    elif stages[5]<0:category='F_grasp_invalid'
    else:category='G_handoff_failed'
    relapses=at_risk=censored=success_competing=0
    for entry in entries:
        tick=entry['tick'];again=next((e['tick'] for e in events if e['tick']>tick),None)
        terminal=row['original_episode_ticks']
        if again is not None and again-tick<=300:relapses+=1;at_risk+=1
        elif terminal-tick>=300:at_risk+=1
        elif row['success']:success_competing+=1
        else:censored+=1
    # Precise false positives require a counterfactual no-intervention rollout.
    mild_trigger=sum(e['initial_distance']<.06 and e['trigger'] in ['eef_deviation','severe_deviation'] for e in events)
    return dict(category=category,stages=[int(v>=0) for v in stages],switches=row['switches'],
        reacquire_repeats=max(0,row['reacquire_triggers']-1),trigger_rule_violation_proxy=mild_trigger,
        handoff_5s_relapses=relapses,handoff_5s_observed=at_risk,handoff_5s_timeout_censored=censored,
        handoff_5s_success_competing=success_competing,
        retreat_attempts=len(plans),retreat_plans=sum(bool(p.get('safe_retreat')) for p in plans),
        retreat_candidate_found=sum(any(c.get('escape_candidate_found') for c in p.get('retreat_checks',[])) for p in plans),
        sampled_retreat_checks=sum(len(p.get('retreat_checks',[])) for p in plans),
        sampled_joint_safe=sum(not any(v>1e-5 for v in c.get('margin_violations',[])) for p in plans for c in p.get('retreat_checks',[])),
        sampled_collision_free=sum(not c.get('model_penetrations') for p in plans for c in p.get('retreat_checks',[])),
        pregrasp_plans=sum(bool(p.get('planning_success')) for p in plans),
        checked_joint_safe=sum(bool(p.get('safe_retreat')) for p in plans),
        collision_free_model_safe_pose=sum(bool(p.get('safe_retreat')) for p in plans))
