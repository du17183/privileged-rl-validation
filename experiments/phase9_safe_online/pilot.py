import json
import os
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
jobs = []
for arm, gpu in [('A',5), ('C',6), ('D',7)]:
    log = (ROOT/'logs'/'phase9_safe_online'/f'pilot_{arm}.log').open('x')
    process = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-u',
        'experiments/phase9_safe_online/train.py', '--arm', arm, '--seed', '0',
        '--steps', '10080', '--smoke', '--device', 'cuda:0'], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
    jobs.append((arm,process,log))
for arm, process, log in jobs:
    code = process.wait()
    log.close()
    marker = ROOT/'checkpoints'/'phase9_safe_online'/f'P9{arm}_seed0_smoke'/'completed.json'
    if code or not marker.exists():
        raise RuntimeError(f'Pilot {arm} failed or incomplete, exit={code}')
    result = json.loads(marker.read_text())
    if result['steps'] != 10080 or result['optimizer_steps'] != 1260:
        raise RuntimeError('Pilot budget mismatch')
    print(arm, json.dumps(result), flush=True)
(ROOT/'results'/'phase9_safe_online'/'pilot_completed.json').write_text(json.dumps(dict(arms=['A','C','D'], steps=10080)))
