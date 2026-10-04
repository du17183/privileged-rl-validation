"""Guard preexisting source roots omitted from earlier inherited inventories."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'
p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
paths=[]
for folder in ('replay','safe_online','envs','evaluation','planners','train','diagnostics',
               'privileged_variants','checkpoint_manager','value_analysis'):
    paths+=list((ROOT/folder).rglob('*'))
current={}
for path in paths:
    if not path.is_file() or '__pycache__' in path.parts or path.name=='quality_replay.py':continue
    st=path.stat();entry=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
    if path.suffix in ('.py','.md','.json','.yaml') and st.st_size<10000000:
        entry['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    current[str(path.relative_to(ROOT))]=entry
source=OUT/'supplemental_baseline_inventory.json'
if not args.verify:
    with source.open('x') as f:json.dump(current,f,indent=2)
    print('Additional preexisting source artifacts recorded',len(current))
else:
    old=json.loads(source.read_text());changed=[k for k in old if current.get(k)!=old[k]]
    result=dict(artifacts=len(old),unchanged=len(old)-len(changed),changed=changed)
    (OUT/'supplemental_baseline_preservation.json').write_text(json.dumps(result,indent=2))
    if changed:raise RuntimeError(f'Prior sources changed: {changed}')
    print(json.dumps(result))
