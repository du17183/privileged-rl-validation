import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'phase15/results'
CHECKPOINTS = ROOT / 'checkpoints/phase15'
LOGS = ROOT / 'logs/phase15'
DATA = ROOT / 'datasets/phase15/recovery_chunks_v1'
AUTH = b'phase15_local'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)


def historical(arm, seed):
    selection = json.loads((ROOT / 'results/phase14_3_pi05_bc/selection.json').read_text())
    step = selection[f'{arm}_seed{seed}']['step']
    return ROOT / f'checkpoints/phase14_3_pi05_bc/{arm}_seed{seed}_step{step}.pt'
