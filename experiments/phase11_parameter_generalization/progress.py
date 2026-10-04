import csv
import json
from datetime import datetime, timezone
from experiments.phase11_parameter_generalization.protocol import OUT, CKPT, ARMS


def main():
    rows=[]
    for arm in ARMS:
        for seed in range(5):
            run=f'P11{arm}_seed{seed}';marker=CKPT/run/'completed.json'
            row=dict(arm=arm,seed=seed,steps=0,status='starting')
            if marker.exists():
                data=json.loads(marker.read_text());row.update(steps=data['steps'],status='trained',
                    online_successes=data['online_successes'],online_episodes=data['online_episodes'],level=data['final_level'],rejections=data['rejections'])
            else:
                path=OUT/f'updates_{run}.csv'
                if path.exists():
                    with path.open() as f:records=list(csv.DictReader(f))
                    if records:
                        last=records[-1];row.update(steps=int(last['env_steps']),status='training',level=int(last['level']),rejections=int(last['rejections']))
            path=OUT/f'eval_{run}_level2.csv'
            if path.exists():
                with path.open() as f:records=list(csv.DictReader(f))
                if records:row.update(random_success=float(records[-1]['success']),online_successes=int(records[-1]['online_successes']),online_episodes=int(records[-1]['online_episodes']))
            rows.append(row)
    heldout=list((OUT/'heldout').glob('*.json')) if (OUT/'heldout').exists() else []
    complete=[p for p in heldout if not p.name.endswith('.partial.json') and not p.name.startswith('error_')]
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(),completed_runs=sum(r['status']=='trained' for r in rows),total_runs=20,
        recorded_training_steps=sum(r['steps'] for r in rows),budget=6000000,
        heldout_jobs_completed=len(complete),heldout_jobs_expected=25,
        errors=[p.name for p in OUT.glob('error_*')],runs=rows)
    (OUT/'current_progress.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
