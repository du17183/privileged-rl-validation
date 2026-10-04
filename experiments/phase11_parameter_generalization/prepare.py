"""Read-only historical inventory and immutable initialization protocol."""
import hashlib
import json
from datetime import datetime, timezone
import h5py
import numpy as np
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, LOG, CKPT, CONFIG
for folder in (OUT, LOG, CKPT): folder.mkdir(parents=True, exist_ok=True)
inventory = {}
for name in ('assets', 'door_env', 'door_dataset', 'progress_rl', 'safe_online', 'algorithms', 'offline_rl',
             'auxiliary_learning', 'stability', 'regularization', 'configs', 'docs', 'experiments', 'replay',
             'trajectory_quality', 'lwd', 'checkpoints', 'results', 'datasets', 'logs', 'environment_state', 'randomized_env', 'curriculum', 'evaluation'):
    for p in (ROOT/name).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or 'phase11_parameter_generalization' in p.parts or p.name.startswith('phase11_parameter_generalization') or name in ('environment_state', 'randomized_env', 'curriculum') or p.name in ('stratified_generalization.py', 'parameter_sensitivity.py'):
            continue
        stat = p.stat();entry = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if stat.st_size < 10000000 and p.suffix in ('.py', '.md', '.usd', '.json', '.yaml', '.toml', '.h5'):
            entry['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        inventory[str(p.relative_to(ROOT))] = entry
with (OUT/'baseline_inventory.json').open('x') as stream: json.dump(inventory, stream, indent=2)
dataset = ROOT/'door_dataset/door_expert_1000.h5'
with h5py.File(dataset, 'r') as h5:
    poses = np.stack([h5[k]['state'][0, 2:5] for k in h5 if k.startswith('traj_')])
    count = len(poses)
assert count == 1000
source_hashes = {}
for seed in range(5):
    for label, path in (('source', ROOT/'checkpoints/phase9_safe_online'/f'P9C_seed{seed}'/'step_300000.pt'),
                        ('anchor', ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{seed}'/'best.pt')):
        source_hashes[f'{label}_{seed}'] = hashlib.sha256(path.read_bytes()).hexdigest()
protocol = dict(**CONFIG, created_at=datetime.now(timezone.utc).isoformat(), handle_center=poses.mean(0).tolist(),
    expert_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(), source_hashes=source_hashes,
    orientation='Excluded: hardware reliability unconfirmed; separate future ablation only',
    deployability='Angle encoder, reset reference, calibrated fixture position, finger contact assumed measurable; hardware verification still required',
    fixture_randomization='Rigid cabinet XYZ translation, not independent reshaping of the handle',
    curriculum_enabled=None, primary_contrasts=['B-A', 'C-B', 'D-C', 'D-A'],
    inference='5 seed paired t confidence intervals and exact sign-flip tests; Holm family correction; no episode pseudoreplication',
    evaluation='Fixed Level2 AUC every 10k, separate nominal preservation; independent heldout 64 episodes/mode/condition at best/final',
    conditional_tests='Sensor noise and input masking only if D improves randomized success over A and preserves nominal >=90%',
    randomization='Independent per-environment RNG, common seed streams; actual exposures may differ under identical adaptive curriculum',
    prior_costs='Additional interactions only; all inherited data and Phase9 training costs disclosed separately')
with (OUT/'protocol.json').open('x') as stream: json.dump(protocol, stream, indent=2)
print(json.dumps(dict(prior_artifacts=len(inventory), handle_center=protocol['handle_center'], expert_trajectories=count)))
