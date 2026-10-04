import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/phase9_safe_online'
records=[]
for path in sorted(OUT.glob('eval_P9*_policy.csv')):
    if 'smoke' in path.name:
        continue
    with path.open() as f:
        rows=list(csv.DictReader(f))
    if rows:
        r=rows[-1]
        records.append((path.stem.removeprefix('eval_').removesuffix('_policy'),int(r['env_steps']),
            round(float(r['success']),3),int(r['online_successes']),int(r['online_episodes']),int(r['rollback_events'])))
print('run,steps,accepted_stochastic_success,online_successes,episodes,rejections')
for row in records:
    print(*row,sep=',')
print('formal_completed',sum(1 for p in (ROOT/'checkpoints/phase9_safe_online').glob('*/completed.json') if 'smoke' not in str(p)))
print('observed_total_training_steps',sum(r[1] for r in records))
