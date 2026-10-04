import hashlib,json,shutil
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,TAG
files=[]
for name in ('environment_feedback',f'experiments/{TAG}','door_env','progress_rl','safe_online','algorithms','regularization','environment_state','randomized_env','offline_rl'):
 files += [p for p in (ROOT/name).rglob('*.py') if '__pycache__' not in p.parts]
manifest={}
for p in files:
 name=str(p.relative_to(ROOT));manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 target=OUT/'training_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
with (OUT/'training_source_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print('Frozen files',len(manifest))
