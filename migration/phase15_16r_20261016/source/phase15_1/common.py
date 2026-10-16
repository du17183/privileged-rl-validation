import json
from pathlib import Path
from phase15.common import ROOT, AUTH, save_json, sha256

RESULTS = ROOT / 'phase15_1/results'
LOGS = ROOT / 'logs/phase15_1'


def frozen(seed):
    selection = json.loads((ROOT / 'phase15/results/selection.json').read_text())
    variant = selection['selected_variant']
    step = selection['candidates'][variant]['seeds'][str(seed)]['step']
    normal = json.loads((ROOT / 'results/phase14_3_pi05_bc/selection.json').read_text())
    from phase15.common import historical, CHECKPOINTS
    source = historical('C', seed)
    return dict(normal=str(CHECKPOINTS / 'normal_policy' / source.name),
                recovery=str(CHECKPOINTS / 'recovery_policy/formal' / f'{variant}_seed{seed}_step{step}.pt'))


def isolated(path):
    path = Path(path).resolve()
    if not path.is_relative_to(RESULTS.resolve()):
        raise RuntimeError('New results must stay within Phase15.1')
    return path
