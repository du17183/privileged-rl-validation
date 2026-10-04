import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase9_safe_online'
groups={}
for p in OUT.glob('eval_P9*_policy.csv'):
    if 'smoke' in p.name:continue
    with p.open() as f:rows=list(csv.DictReader(f))
    if not rows:continue
    arm=p.name.split('_seed')[0].removeprefix('eval_P9')
    groups.setdefault(arm,[]).append(rows[-1])
total=0
for arm,rows in sorted(groups.items()):
    steps=[int(r['env_steps']) for r in rows];total+=sum(steps)
    print(arm,'seeds=',len(rows),'steps=',[min(steps),max(steps)],'new_success=',sum(int(r['online_successes']) for r in rows),
          'episodes=',sum(int(r['online_episodes']) for r in rows),'current_sample_success=',round(sum(float(r['success']) for r in rows)/len(rows),3))
print('training_interactions_observed',total)
print('completed_runs',sum(1 for p in (ROOT/'checkpoints/phase9_safe_online').glob('*/completed.json') if 'smoke' not in str(p)))
print('heldout_completed',len(list((OUT/'heldout').glob('*_seed?.json'))))
errors=list(OUT.glob('error_*.txt'))+list((OUT/'heldout').glob('*error.txt'))+list((OUT/'heldout_pilot').glob('*error.txt'))
print('errors',[str(p.relative_to(ROOT)) for p in errors])
