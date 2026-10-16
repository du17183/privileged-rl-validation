"""Count actual closed physical traces, including discarded technical attempts."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import h5py,numpy as np
from experiments.phase16_r_residual_sac.common import ROOT,RES,save,sha,utc

def main():
    rows=[];issues=[];pending=[]
    for path in sorted(RES.rglob('*.h5')):
        if any(part in ['report_revisions','source_versions'] for part in path.parts):continue
        relative=path.relative_to(RES);attempt=next((p for p in relative.parts if p.startswith('attempt')),None)
        category='technical_failed_attempts' if attempt else 'engineering' if relative.parts[0]=='engineering' else 'training' if relative.parts[0]=='train' else 'evaluation'
        if category=='training':
            p=path.parent;metapath=next((p/name for name in ['complete.json','stopped.json','failed.json'] if (p/name).exists()),None)
        else:metapath=path.with_suffix('.json') if path.with_suffix('.json').exists() else None
        if metapath is None:pending.append(str(relative));continue
        meta=json.loads(metapath.read_text())
        try:
            with h5py.File(path) as h:
                if 'executed' not in h:issues.append('Unrecognized physical trace '+str(relative));continue
                nt,n=h['executed'].shape[:2];count=nt*n;resets=int(h['done'][:].sum())
                completed=int(h['done'][:].sum());success=int(((h['next_gt'][:,:,0]>1)&h['done'][:].astype(bool)).sum())
                scored=int((h['done'][:].astype(bool)&h['scored'][:].astype(bool)).sum()) if 'scored' in h else completed
        except OSError:pending.append(str(relative));continue
        wanted=meta.get('physical_interactions',meta.get('training_interactions'))
        if wanted is not None and count!=wanted:issues.append('Interaction count differs '+str(relative))
        wanted_reset=meta.get('automatic_resets',meta.get('training_automatic_resets'))
        if wanted_reset is not None and resets!=wanted_reset:issues.append('Reset count differs '+str(relative))
        actual_sha=sha(path)
        if meta.get('trace_sha256') and meta['trace_sha256']!=actual_sha:issues.append('Trace SHA differs '+str(relative))
        rows.append(dict(path=str(path.relative_to(ROOT)),category=category,attempt=attempt,vector_ticks=nt,envs=n,physical_interactions=count,automatic_resets=resets,all_completed_episodes=completed,all_successes=success,scored_completed_episodes=scored,unscored_completions=completed-scored,wall_s=meta.get('elapsed_s'),sha256=actual_sha))
    probe=RES/'reset_probe003/complete.json'
    if probe.exists():
        x=json.loads(probe.read_text());rows.append(dict(path=str(probe.relative_to(ROOT)),category='technical_reset_probe',attempt='probe004',vector_ticks=3,envs=16,physical_interactions=x['physical_interactions'],automatic_resets=0,all_completed_episodes=0,all_successes=0,scored_completed_episodes=0,unscored_completions=0,wall_s=None,sha256=sha(probe)))
    totals={}
    for category in sorted({r['category'] for r in rows}):
        values=[r for r in rows if r['category']==category]
        totals[category]={k:sum(r[k] for r in values) for k in ['physical_interactions','automatic_resets','all_completed_episodes','all_successes','scored_completed_episodes','unscored_completions']}
        totals[category]['rollouts']=len(values)
    total={k:sum(r[k] for r in rows) for k in ['physical_interactions','automatic_resets','all_completed_episodes','all_successes']}
    with (RES/'physical_cost_ledger.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(RES/'physical_cost_ledger.json',dict(utc=utc(),passed=not issues and not pending,issues=issues,pending_unclosed_traces=pending,categories=totals,total=total,source_sha256=sha(Path(__file__)),counting='each distinct physical execution path counted once; identical trace SHA across repeated physical attempts still represents separate interactions; archived copies excluded',scope='task env.step transitions, including unscored clones; physics engine initialization/reset-forward/startup are not measured robot task transitions; automatic resets counted from actual terminal flags; explicit startup resets separate from automatic episode resets'))
    print(json.dumps(dict(issues=issues,pending=pending,totals=totals,total=total)))
if __name__=='__main__':main()
