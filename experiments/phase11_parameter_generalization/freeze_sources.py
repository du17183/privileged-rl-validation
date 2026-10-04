import hashlib
import json
import shutil
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT
paths = []
for folder in ('environment_state', 'randomized_env', 'curriculum', 'experiments/phase11_parameter_generalization'):
    paths.extend((ROOT/folder).glob('*.py'))
paths += [ROOT/'evaluation/stratified_generalization.py', ROOT/'evaluation/parameter_sensitivity.py', ROOT/'docs/phase11_parameter_generalization_protocol.md']
manifest = {}
snapshot = OUT/'training_source';snapshot.mkdir(exist_ok=False)
for path in paths:
    target = snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(path, target)
    manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
with (OUT/'training_source_manifest.json').open('x') as stream: json.dump(manifest, stream, indent=2)
print('Captured', len(manifest), 'sources')
