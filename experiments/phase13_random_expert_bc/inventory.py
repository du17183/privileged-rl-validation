"""Freeze source hashes and prior artifact size/mtime before Phase 13."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/phase13_random_expert_bc'


def old(path):
    text = path.relative_to(ROOT).as_posix()
    return ('phase13' not in text and not text.startswith('datasets/random_door_expert/')
            and '__pycache__' not in text and '.git/' not in text)


def snapshot():
    sources = {}
    for path in ROOT.rglob('*.py'):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(('.venv/', 'third_party/')) or not old(path): continue
        sources[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    artifacts = {}
    for directory in ['results','checkpoints','datasets','door_dataset','docs']:
        base = ROOT/directory
        if not base.exists(): continue
        for path in base.rglob('*'):
            if not path.is_file() or not old(path): continue
            stat = path.stat()
            artifacts[path.relative_to(ROOT).as_posix()] = [stat.st_size, stat.st_mtime_ns]
    return dict(sources=sources,artifacts=artifacts)


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'baseline_inventory.json'
    value=snapshot()
    if not target.exists():
        with target.open('x') as f: json.dump(value,f)
        print('FROZEN',len(value['sources']),len(value['artifacts']))
    else:
        prior=json.loads(target.read_text())
        changed={key:[name for name,oldvalue in prior[key].items() if value[key].get(name)!=oldvalue]
                 for key in ['sources','artifacts']}
        report=dict(changed=changed,source_count=len(prior['sources']),artifact_count=len(prior['artifacts']))
        (OUT/'preservation_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report))
        if any(changed.values()): raise RuntimeError('Previous artifacts changed')
