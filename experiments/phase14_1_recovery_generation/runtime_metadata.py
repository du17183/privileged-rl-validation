import json,sys,subprocess,importlib.metadata
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[2]
versions={}
for package in ['torch','numpy','h5py','scipy','matplotlib','isaaclab','isaacsim','pin']:
    try:versions[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:versions[package]='distribution metadata unavailable'
value=dict(python=sys.version,executable=sys.executable,versions=versions,torch_cuda=torch.version.cuda,
           gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv,noheader'],text=True).splitlines(),
           robot='Franka Panda',task='existing randomized Panda Door',num_envs_per_process=32,
           training_updates=0,bc_updates=0,rl_updates=0)
(root/'results/phase14_1_recovery_generation/formal_v1/runtime_environment.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value))
