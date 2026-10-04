"""Only decide curriculum before releasing formal jobs; no hyperparameter search."""
import csv
import json
from experiments.phase11_parameter_generalization.protocol import OUT, CKPT
marker = CKPT/'P11D_seed0_pilot/completed.json'
if not marker.exists(): raise RuntimeError('Pilot incomplete')
rows = list(csv.DictReader((OUT/'eval_P11D_seed0_pilot_level2.csv').open()))
random_success = float(rows[-1]['success'])
updates = list(csv.DictReader((OUT/'updates_P11D_seed0_pilot.csv').open()))
enabled = random_success < .8 or int(updates[-1]['rejections']) > 0
path = OUT/'protocol.json';protocol = json.loads(path.read_text())
if protocol['curriculum_enabled'] is not None: raise RuntimeError('Decision already frozen')
protocol['curriculum_enabled'] = enabled
protocol['pilot_decision'] = dict(steps=10016, arm='D', seed=0, level2_success=random_success,
    reason='Direct Level2 pilot below 80% or rejected update' if enabled else 'Direct Level2 pilot stable >=80%',
    rule='Same controller every arm: start Level0, advance one level only after two disjoint 64-episode policy evaluations both >=80%; retain episode boundaries',
    limitation='Adaptive realized exposure can differ; identical random seeds and controller; fixed Level2 heldout is primary. Any effect includes state-mediated curriculum progression.')
path.write_text(json.dumps(protocol, indent=2))
(OUT/'pilot_decision.json').write_text(json.dumps(protocol['pilot_decision'], indent=2))
print(json.dumps(protocol['pilot_decision']))
