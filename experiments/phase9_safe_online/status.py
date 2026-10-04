import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in sorted((ROOT/'results/phase9_safe_online').glob('anchor_seed*.json')):
    r = json.loads(p.read_text())
    print(p.name, {k:round(v['success'],4) for k,v in r['metrics'].items()}, flush=True)
for p in sorted((ROOT/'checkpoints/phase9_safe_online').glob('*/completed.json')):
    print(p.parent.name, p.read_text(), flush=True)
