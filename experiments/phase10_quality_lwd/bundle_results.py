"""Review-sized final artifact; full datasets/checkpoints stay on server."""
import json
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'
if not (OUT/'phase10_completed.json').exists():raise RuntimeError('No final bundle before completion')
destination=OUT/'review_artifacts.zip'
paths=[ROOT/'docs/phase10_quality_lwd_report.md']
for folder in (OUT,ROOT/'experiments/phase10_quality_lwd',ROOT/'trajectory_quality',ROOT/'lwd'):
    paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.jsonl','.csv','.png','.txt')
                 and 'ipc' not in p.parts and '__pycache__' not in p.parts and 'initial_upload_staging' not in p.parts)
paths.append(ROOT/'replay/quality_replay.py')
paths.extend((ROOT/'checkpoints/phase10_quality_lwd').glob('*/completed.json'))
with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(paths)):z.write(p,str(p.relative_to(ROOT)))
(OUT/'review_bundle.json').write_text(json.dumps(dict(path=str(destination),size_bytes=destination.stat().st_size,
    files=len(set(paths)),excluded='Full HDF5 datasets, binary checkpoints, TensorBoard and simulator logs remain under project paths'),indent=2))
print('Final review bundle',destination.stat().st_size,'bytes')
