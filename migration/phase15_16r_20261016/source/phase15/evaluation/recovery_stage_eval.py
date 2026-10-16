"""Augment unchanged historical tests with event-level module diagnostics."""
import numpy as np


def summarize(rows):
    events = [e for r in rows for e in r['events']]
    plans = [p for r in rows for p in r['planner']]
    result = dict(episodes_with_failure=sum(bool(r['events']) for r in rows),
        failure_detection_fraction=float(np.mean([bool(r['events']) for r in rows])) if rows else None,
        switch_per_episode_fraction=float(np.mean([any(e['switch_correct'] for e in r['events']) for r in rows])) if rows else None,
        failure_events=len(events), recovery_triggers=sum(r['recovery_triggers'] for r in rows),
        reacquire_triggers=sum(r['reacquire_triggers'] for r in rows),
        switches_mean=float(np.mean([r['switches'] for r in rows])) if rows else 0.)
    for k, name in enumerate(['reapproach', 'recontact', 'regrasp', 'progress_resumed']):
        result[name + '_per_detected_event'] = float(np.mean([e['flags'][k] for e in events])) if events else None
    for name in ['switch_correct', 'returned_normal', 'final_success']:
        result[name + '_per_detected_event'] = float(np.mean([e.get(name, False) for e in events])) if events else None
    result['reacquire'] = dict(events=len(plans))
    for name in ['ik_success', 'planning_success', 'start_collision', 'reached_pregrasp', 'regrasp', 'handback']:
        result['reacquire'][name] = float(np.mean([p.get(name, False) for p in plans])) if plans else None
    result['reacquire']['failure_stages'] = {name: sum(p.get('failure_stage') == name for p in plans)
        for name in sorted(set(str(p.get('failure_stage')) for p in plans)) if name != 'None'}
    result['reacquire']['planning_latency_mean_s'] = float(np.mean([p['plan_s'] for p in plans])) if plans else None
    result['reacquire']['final_after_handback'] = float(np.mean([p['final_task_success'] for p in plans if p.get('handback')])) if any(p.get('handback') for p in plans) else None
    result['mean_episode_inference_s'] = float(np.mean([sum(r['costs_s'].values()) +
        sum(p['plan_s'] for p in r['planner']) for r in rows])) if rows else None
    return result
