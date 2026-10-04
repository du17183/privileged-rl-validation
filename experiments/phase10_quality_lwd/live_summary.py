"""Short read-only progress and throughput snapshots, never partial inference."""
import csv
import json
import time
from pathlib import Path
from experiments.phase10_quality_lwd.protocol import ARMS
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'
rows=[]
for arm in ARMS:
    stages=[];successes=episodes=complete=0
    for seed in range(5):
        path=OUT/f'eval_P10{arm}_seed{seed}_policy.csv'
        if path.exists():
            with path.open() as f:data=list(csv.DictReader(f))
            last=data[-1];stages.append(int(last['env_steps']))
            successes+=int(last['online_successes']);episodes+=int(last['online_episodes'])
        complete+=(ROOT/'checkpoints/phase10_quality_lwd'/f'P10{arm}_seed{seed}'/'completed.json').exists()
    rows.append(dict(arm=arm,started=len(stages),complete=complete,steps=sum(stages),
                     range=[min(stages),max(stages)] if stages else None,online_successes=successes,online_episodes=episodes))
queue=json.loads((OUT/'queue_state.json').read_text())
result=dict(timestamp=time.time(),complete=sum(r['complete'] for r in rows),formal_steps=sum(r['steps'] for r in rows),
            target=10500000,percent=round(100*sum(r['steps'] for r in rows)/10500000,2),
            active=len(queue['running']),pending=len(queue['pending']),arms=rows,
            errors=[p.name for p in OUT.glob('error*')],pipeline_error=(OUT/'pipeline_error.txt').exists())
print(json.dumps(result,separators=(',',':')))
with (OUT/'throughput_snapshots.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
