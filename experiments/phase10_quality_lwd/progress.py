"""Read-only complete progress snapshot; no summaries from unfinished seeds."""
import csv
import json
import time
from pathlib import Path
from experiments.phase10_quality_lwd.protocol import ARMS
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'
records=[]
for arm in ARMS:
    for seed in range(5):
        run=f'P10{arm}_seed{seed}'
        path=OUT/f'eval_{run}_policy.csv'
        rows=[]
        if path.exists():
            with path.open() as f:rows=list(csv.DictReader(f))
        marker=ROOT/'checkpoints/phase10_quality_lwd'/run/'completed.json'
        record=dict(arm=arm,seed=seed,steps=int(rows[-1]['env_steps']) if rows else 0,complete=marker.exists())
        if rows:
            r=rows[-1]
            record.update(success=float(r['success']),online_successes=int(r['online_successes']),online_episodes=int(r['online_episodes']),
                          rollback_events=int(r['rollback_events']),wall_time_s=float(r['wall_time_s']))
        records.append(record)
errors=[str(p.relative_to(ROOT)) for p in OUT.glob('error_*.txt')]
print(json.dumps(dict(timestamp=time.time(),training_complete=sum(r['complete'] for r in records),
    formal_evaluated_training_steps=sum(r['steps'] for r in records),target_steps=10500000,
    records=records,errors=errors),indent=2))
