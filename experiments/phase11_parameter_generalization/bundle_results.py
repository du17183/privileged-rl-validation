import hashlib
import json
import zipfile
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, CKPT, LOG
archive=OUT/'phase11_review_artifacts.zip'
folders=['environment_state','randomized_env','curriculum','experiments/phase11_parameter_generalization']
files=[]
for name in folders:files += [p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [ROOT/'evaluation/stratified_generalization.py',ROOT/'evaluation/parameter_sensitivity.py',ROOT/'docs/phase11_parameter_generalization_report.md',ROOT/'docs/phase11_parameter_generalization_protocol.md']
files += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.png','.py','.md') and 'ipc' not in p.parts and p.name not in ('review_bundle.json',)]
files += list(CKPT.glob('*/completed.json'))
files += [p for p in LOG.glob('*.log') if p.stat().st_size<2000000]
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
    for p in sorted(set(files)):stream.write(p,p.relative_to(ROOT))
manifest=dict(archive=str(archive),size=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              full_checkpoints_datasets='Remain on b300-2; review archive contains code, numerical results, figures, audits and lightweight logs')
(OUT/'review_bundle.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
