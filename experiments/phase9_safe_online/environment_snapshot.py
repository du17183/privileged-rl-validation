import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase9_safe_online'
packages={}
for name in ['isaacsim','isaaclab','torch','numpy','h5py','tensorboard','gymnasium','scipy','matplotlib','nvidia-cuda-nvrtc-cu12']:
    try:packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:packages[name]=None
result=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages,
    torch_cuda=torch.version.cuda,isaaclab_commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/IsaacLab'),'rev-parse','HEAD'],text=True).strip(),
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv'],text=True),
    ld_library_path=os.environ.get('LD_LIBRARY_PATH'),project_realpath=str(ROOT))
library=ROOT/'.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
result['nvrtc_resolved_library']=str(library.resolve()) if library.exists() else None
(OUT/'environment_snapshot.json').write_text(json.dumps(result,indent=2))
(OUT/'environment_pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
print(json.dumps(result,indent=2))
