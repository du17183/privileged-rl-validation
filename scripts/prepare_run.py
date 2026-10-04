#!/usr/bin/env python3
"""Create a clean rerun directory without old completion markers or weights."""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'datasets','checkpoints','results','logs','third_party','external_weights','_release_cache','_weight_release_cache','runs','reproducibility','.git'}


def main(args):
    if not args.name or Path(args.name).name != args.name: raise ValueError('Use one directory name')
    run = ROOT / 'runs' / args.name
    if run.exists(): raise FileExistsError('Do not reuse a historical experiment directory')
    prepared = ROOT / 'datasets/phase14_2/prepared_v1/manifest.json'
    cohorts = ROOT / 'results/phase14_3_pi05_bc/cohorts'
    if not prepared.exists() or not cohorts.exists(): raise RuntimeError('Restore datasets and results first')
    if args.dry_run:
        print(json.dumps({'run':str(run),'uses_saved_pressure_cohorts':True,'training_launched':False})); return
    run.mkdir(parents=True)
    for item in ROOT.iterdir():
        if item.name in SKIP or item.name.startswith('.venv') or item.name == '__pycache__': continue
        if item.is_dir(): shutil.copytree(item, run/item.name, ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.h5','*.npz'))
        elif item.suffix in {'.py','.sh','.md','.toml','.json','.txt'}: shutil.copy2(item, run/item.name)
    for name in ['datasets','third_party','external_weights','.venv','.venv_pi05']:
        (run/name).symlink_to(ROOT/name, target_is_directory=True)
    for name in ['logs','checkpoints','results','docs']: (run/name).mkdir(exist_ok=True)
    shutil.copytree(ROOT/'reproducibility', run/'reproducibility')
    destination = run / 'results/phase14_3_pi05_bc/cohorts'
    shutil.copytree(cohorts, destination)
    print('Clean run created:', run)
    print('No training was launched. Activate configs/runtime_env.sh in this directory before running experiments.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--name',required=True); parser.add_argument('--dry-run',action='store_true')
    main(parser.parse_args())
