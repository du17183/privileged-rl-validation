"""Compact read-only wall-time progress, with no incomplete seed inference."""
import csv
import json
from pathlib import Path
from experiments.phase10_quality_lwd.protocol import ARMS
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'
records=[]
for arm in ARMS:
    rows=[]
    for seed in range(5):
        p=OUT/f'eval_P10{arm}_seed{seed}_policy.csv'
        if p.exists():
            with p.open() as f:r=list(csv.DictReader(f))[-1]
            rows.append(dict(seed=seed,steps=int(r['env_steps']),success=float(r['success']),
                online_successes=int(r['online_successes']),online_episodes=int(r['online_episodes']),
                wall_time_s=round(float(r['wall_time_s']))))
    records.append(dict(arm=arm,runs_started=len(rows),runs_completed=sum((ROOT/'checkpoints/phase10_quality_lwd'/f'P10{arm}_seed{s}'/'completed.json').exists() for s in range(5)),
        evaluated_steps=sum(r['steps'] for r in rows),online_successes=sum(r['online_successes'] for r in rows),records=rows))
print(json.dumps(dict(completed=sum(r['runs_completed'] for r in records),evaluated_training_steps=sum(r['evaluated_steps'] for r in records),
    target_steps=10500000,arms=records,errors=[p.name for p in OUT.glob('error*')],
    pipeline_error=(OUT/'pipeline_error.txt').exists()),indent=2))
