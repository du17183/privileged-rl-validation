"""Phase16-R files are confined to the new experiment namespaces."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments/phase16_r_residual_sac'
RES = ROOT / 'results/phase16_r'
AUTH = b'phase16_r_local'

def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)

def protocol():
    p = EXP / 'protocol.json'
    lock = json.loads((EXP / 'protocol_lock.json').read_text())
    if sha(p) != lock['sha256']:
        raise RuntimeError('Preregistered protocol changed')
    return json.loads(p.read_text())

def checkpoint(seed):
    manifest = json.loads((RES / 'historical_inputs.json').read_text())
    row = manifest['normal'][str(seed)]
    path = ROOT / row['checkpoint']
    if sha(path) != row['sha256']:
        raise RuntimeError('Frozen Normal checkpoint changed')
    return path
