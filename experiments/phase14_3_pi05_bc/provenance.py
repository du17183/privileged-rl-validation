"""Read-only runtime, official source and pretrained weight provenance."""
import argparse,hashlib,importlib.metadata,json,platform,subprocess,sys,time
from pathlib import Path
from runtime_paths import OPENPI_SOURCE, PI05_BASE, dependency_commit, dependency_status
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_3_pi05_bc'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def main(a):
    versions={}
    for name in ['torch','numpy','transformers','safetensors','sentencepiece','h5py','scipy','isaacsim','isaaclab','jax','openpi']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    d=dict(time=time.time(),python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions)
    if a.full:
        source=OPENPI_SOURCE
        d['openpi_commit']=dependency_commit(source)
        d['openpi_worktree_status']=dependency_status(source)
        d['openpi_used_sources']={str(p.relative_to(source)):sha(p) for part in ['src/openpi/models_pytorch','src/openpi/models/pi0_config.py','src/openpi/models/tokenizer.py'] for p in ((source/part).rglob('*.py') if (source/part).is_dir() else [source/part])}
        weights=PI05_BASE
        d['pretrained_base']=dict(path=str(weights),bytes=weights.stat().st_size,sha256=sha(weights),scope='task-independent pretrained pi05 base; no peg task fine-tuned weights')
        from transformers import modeling_utils
        tf=Path(modeling_utils.__file__).parent
        d['transformers_used_sources']={str(p.relative_to(tf)):sha(p) for p in [tf/'models/siglip/modeling_siglip.py',tf/'models/gemma/modeling_gemma.py']}
        d['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv,noheader'],text=True)
        used=['pi05','experiments/phase14_3_pi05_bc','evaluation/pi05_closed_loop_eval.py','evaluation/multimodal_action_analysis.py',
            'evaluation/recovery_policy_eval.py','experiments/phase14_2_recovery_bc/model.py','experiments/phase13_random_expert_bc/model.py',
            'randomized_env/door_randomization.py','environment_state','recovery_expert/kinematic_compensation.py','configs/runtime_env.sh']
        d['project_used_sources']={str(p.relative_to(ROOT)):sha(p) for part in used for p in ((ROOT/part).rglob('*.py') if (ROOT/part).is_dir() else [ROOT/part])}
    (R/(a.name+'_provenance.json')).write_text(json.dumps(d,indent=2));print(json.dumps({'runtime':a.name,'packages':versions,'full':a.full}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--full',action='store_true');main(p.parse_args())
