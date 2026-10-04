"""Verify unchanged scientific artifacts; Python caches are excluded."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase10_quality_lwd'
parser=argparse.ArgumentParser()
parser.add_argument('--verify',action='store_true')
args=parser.parse_args()
folders=['door_env','door_dataset','assets','progress_rl','algorithms','offline_rl','auxiliary_learning','stability','regularization','configs','docs','experiments','checkpoints','results','datasets','logs']
current={}
for folder in folders:
    for p in (ROOT/folder).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or 'phase10_quality_lwd' in p.parts or p.name=='phase10_quality_lwd_report.md' or 'trajectory_quality' in p.parts or 'lwd' in p.parts or p.name=='quality_replay.py':
            continue
        s=p.stat()
        item=dict(size=s.st_size,mtime_ns=s.st_mtime_ns)
        if p.suffix in ('.py','.md','.yaml','.toml','.usd','.json') and s.st_size<10000000:
            item['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        current[str(p.relative_to(ROOT))]=item
path=OUT/'baseline_inventory.json'
if args.verify:
    old=json.loads(path.read_text())
    changed=[p for p in old if current.get(p)!=old[p]]
    result=dict(artifacts=len(old),changed=changed,unchanged=len(old)-len(changed))
    (OUT/'baseline_preservation.json').write_text(json.dumps(result,indent=2))
    if changed:
        raise RuntimeError(f'Changed prior artifacts: {changed[:10]}')
    print(json.dumps(result))
else:
    with path.open('x') as f:
        json.dump(current,f,indent=2)
    print('Recorded',len(current),'prior artifacts')
