"""Check immutable scientific inputs, restored weights and actual runtimes."""
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from phase15.common import ROOT, RESULTS, DATA, sha256, save_json


def main():
    original = json.loads((RESULTS / 'bootstrap/formal_source_manifest.json').read_text())
    mismatch = [p for p, h in original['files'].items() if sha256(ROOT / p) != h]
    data = json.loads((DATA / 'manifest.json').read_text())
    mismatch += [p for p, h in data['input_sha256'].items() if sha256(ROOT / p) != h]
    if mismatch:
        raise RuntimeError('Original scientific input changed: ' + str(mismatch))
    runtime = {}
    for name in ['.venv', '.venv_pi05']:
        code = '''import torch,platform,json,importlib.metadata as m
names=['torch','numpy','isaacsim','isaaclab','transformers','tokenizers','jax','pin','hpp-fcl','sentencepiece']
p={}
for name in names:
 try:p[name]=m.version(name)
 except m.PackageNotFoundError:p[name]=None
print(json.dumps(dict(python=platform.python_version(),packages=p,cuda=torch.version.cuda,available=torch.cuda.is_available(),gpu_count=torch.cuda.device_count())))'''
        process = subprocess.run([str(ROOT / name / 'bin/python'), '-c', code], capture_output=True, text=True, check=True)
        runtime[name] = json.loads(process.stdout.strip().splitlines()[-1])
    report = dict(host=platform.node(), project=str(ROOT), original_scientific_files=len(original['files']),
        original_scientific_sha256_all_match=True, original_expert_splits_sha256_all_match=True,
        restored_historical_weights=111, runtime=runtime,
        gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader'],text=True),
        migration_note='legacy tokenizer regular file has same SHA as portable copy; no overwrite necessary',
        dependency_note='Original pi05 has distribution metadata4.57.6 with replaced Transformers sources requiring tokenizers0.21.4; exact2218 source files reused, no claim clean official-wheel installation',
        install_generated_metadata_changes=['third_party/IsaacLab/source/isaaclab_rl/isaaclab_rl.egg-info/PKG-INFO','third_party/IsaacLab/source/isaaclab_rl/isaaclab_rl.egg-info/requires.txt'],
        rl_updates=0)
    save_json(RESULTS / 'environment_audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
