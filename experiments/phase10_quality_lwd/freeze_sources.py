"""Record exact new source and unchanged dependency provenance for review."""
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'
destination=OUT/'source_snapshot';destination.mkdir(exist_ok=False)
files=list((ROOT/'experiments/phase10_quality_lwd').glob('*.py'))
files+=list((ROOT/'trajectory_quality').glob('*.py'))+list((ROOT/'lwd').glob('*.py'))
files+=[ROOT/'replay/quality_replay.py']
manifest={}
for p in files:
    rel=p.relative_to(ROOT);target=destination/rel
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    manifest[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
data=dict(captured_at=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),torch=torch.__version__,
    cuda=torch.version.cuda,new_source=manifest,gpu_state=subprocess.check_output(['nvidia-smi',
    '--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),
    baseline_inventory='Existing phase1-9 scientific artifacts preserved separately; no dependency source edited',
    archive_note='Core learner and replay code unchanged after queue release; analysis/report files captured after implementation, before results.')
(OUT/'source_manifest.json').write_text(json.dumps(data,indent=2))
print('Recorded source snapshot',len(files))
