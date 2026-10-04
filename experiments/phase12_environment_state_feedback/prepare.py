"""Freeze prior artifacts and preregister before any Phase12 learning."""
import hashlib,json
from datetime import datetime,timezone
import h5py,numpy as np
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,LOG,CKPT,CONFIG,TAG
for p in (OUT,LOG,CKPT):p.mkdir(parents=True,exist_ok=True)
inventory={}
names=('assets','door_env','door_dataset','progress_rl','safe_online','algorithms','offline_rl','auxiliary_learning',
 'stability','regularization','configs','docs','experiments','replay','trajectory_quality','lwd','checkpoints',
 'results','datasets','logs','environment_state','randomized_env','curriculum','evaluation')
for name in names:
 for p in (ROOT/name).rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts or TAG in p.parts or p.name.startswith(TAG):continue
  s=p.stat();entry=dict(size=s.st_size,mtime_ns=s.st_mtime_ns)
  if s.st_size<10000000 and p.suffix in ('.py','.md','.usd','.json','.yaml','.toml','.h5'):
   entry['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  inventory[str(p.relative_to(ROOT))]=entry
with (OUT/'baseline_inventory.json').open('x') as f:json.dump(inventory,f,indent=2)
dataset=ROOT/'door_dataset/door_expert_1000.h5'
with h5py.File(dataset,'r') as h:
 poses=np.stack([h[k]['state'][0,2:5] for k in h if k.startswith('traj_')]);assert len(poses)==1000
source_hashes={}
for s in range(5):
 for label,p in (('source',ROOT/'checkpoints/phase9_safe_online'/f'P9C_seed{s}'/'step_300000.pt'),('anchor',ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{s}'/'best.pt')):
  source_hashes[f'{label}_{s}']=hashlib.sha256(p.read_bytes()).hexdigest()
protocol=dict(**CONFIG,created_at=datetime.now(timezone.utc).isoformat(),handle_center=poses.mean(0).tolist(),
 expert_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),source_hashes=source_hashes,
 robot_observation='Existing 26D q9/qd9/EE XYZ3/quaternion4/clock1 retained in all arms; gripper joints included',
 contact='Two independent finger filtered contact force >0.5N; measured binary signals',
 handle_pose='Rigid cabinet translation; workspace handle XYZ; orientation excluded',
 parameter_diagnostics='Individual masking cannot isolate redundant angle/progress/remaining; coherent family masking additionally required; not retrained feature ablation',
 phase11_difference='Phase11 trained adaptively;19/20 stopped atLevel1. Phase12 uses fullLevel2 always and adds velocity/remaining.',
 deployment='Inputs assumed available from encoders/contact/fixture calibration; handle sensing needs real hardware validation',
 costs='300k additional per run, inherited expert and source-policy training separately disclosed')
with (OUT/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
print(json.dumps(dict(prior_artifacts=len(inventory),expert_trajectories=len(poses),handle_center=protocol['handle_center'])))
