"""Read-only current progress snapshot; incomplete arms have no final claims."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from experiments.phase9_safe_online.analyze import summarize, paired
from experiments.phase9_safe_online.protocol import ARMS
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase9_safe_online'

def main():
    result={'utc':datetime.now(timezone.utc).isoformat(),'arms':{}}
    for arm in ARMS:
        rows=[]
        for seed in range(5):
            path=OUT/f'eval_P9{arm}_seed{seed}_policy.csv'
            with path.open() as stream:
                rows.append(list(csv.DictReader(stream)))
        entry=dict(steps=[int(v[-1]['env_steps']) for v in rows],
            online_successes=sum(int(v[-1]['online_successes']) for v in rows),
            online_episodes=sum(int(v[-1]['online_episodes']) for v in rows))
        entry['training_complete']=entry['steps']==[300000]*5
        if entry['training_complete']:
            entry['policy_auc']=summarize([np.trapz([float(r['success']) for r in v],[int(r['env_steps']) for r in v])/300000 for v in rows])
            entry['curve_policy_final']=summarize([float(v[-1]['success']) for v in rows])
            entry['rejections']=summarize([int(v[-1]['rollback_events']) for v in rows])
        paths=[OUT/'heldout'/f'{arm}_seed{s}.json' for s in range(5)]
        entry['independent_seeds_complete']=sum(p.exists() for p in paths)
        if all(p.exists() for p in paths):
            tests=[json.loads(p.read_text()) for p in paths]
            for mode in ('policy','deterministic'):
                for selection in ('best','final'):
                    entry[f'{mode}_{selection}']=summarize([next(r['success'] for r in test if r['checkpoint']==selection and r['condition']=='nominal' and r['mode']==mode) for test in tests])
                entry[f'{mode}_gap']=summarize(np.array(entry[f'{mode}_best']['per_seed'])-entry[f'{mode}_final']['per_seed'])
            entry['perturbed_policy_final']={condition:summarize([next(r['success'] for r in test if r['checkpoint']=='final' and r['condition']==condition and r['mode']=='policy') for test in tests]) for condition in ('angle_2p5','angle_5','handle_y_minus_1cm','handle_y_plus_1cm')} if arm in ('A','B','C','D') else {}
        result['arms'][arm]=entry
    if all('policy_final' in result['arms'][a] for a in ('A','B','C','D')):
        result['primary_paired_final']={f'{a}-{b}':paired(result['arms'][a]['policy_final']['per_seed'],result['arms'][b]['policy_final']['per_seed']) for a,b in [('B','A'),('C','B'),('D','C')]}
    (OUT/'current_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
