#!/usr/bin/env python3
"""CPU-only checks: Python syntax and declared data/model integrity."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS={'.pt','.pth','.ckpt','.safetensors','.onnx'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main(args):
    errors=[]; syntax=0
    for path in ROOT.rglob('*.py'):
        if any(x in path.parts for x in ['.git','_release_cache','runs','results']) or any(x.startswith('.venv') for x in path.parts):continue
        try: ast.parse(path.read_text(encoding='utf-8'),filename=str(path));syntax+=1
        except (ValueError,SyntaxError,UnicodeError) as exc: errors.append(str(path.relative_to(ROOT))+': '+str(exc))
    catalog_path=ROOT/'reproducibility/selected_weights_catalog.jsonl'
    models=[json.loads(line) for line in catalog_path.read_text().splitlines() if line] if catalog_path.exists() else []
    declared={item['path'] for item in models}
    weights=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in WEIGHTS and str(p.relative_to(ROOT)) not in declared and not any(x in p.parts for x in ['external_weights','runs','.git','_release_cache','_weight_release_cache']) and not any(x.startswith('.venv') for x in p.parts)]
    if weights:errors.append('Unexpected policy weight files: '+repr(weights))
    checked=missing=0
    if args.data:
        manifest=json.loads((ROOT/'reproducibility/original_files.json').read_text())
        for item in manifest['files']:
            path=ROOT/item['path']
            if path.parts[len(ROOT.parts)] not in {'datasets','results','logs','checkpoints','third_party','door_dataset'} and not path.name.endswith(('.tar','.tar.gz','.zip')):continue
            if item['path'].startswith('door_dataset/') and path.suffix=='.py':continue
            if not path.exists():missing+=1;continue
            if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:errors.append('Integrity mismatch: '+item['path'])
            checked+=1
        if missing:errors.append(f'{missing} expected archived files are missing; restore all groups first')
    model_checked=0
    if args.weights:
        if not models:errors.append('Selected weight catalog is missing')
        for item in models:
            path=ROOT/item['path']
            if not path.is_file():errors.append('Missing selected model: '+item['path']);continue
            if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:errors.append('Model integrity mismatch: '+item['path'])
            model_checked+=1
    result={'python_files_checked':syntax,'restored_files_checked':checked,'selected_models_checked':model_checked,'errors':errors}
    print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',action='store_true');parser.add_argument('--weights',action='store_true');main(parser.parse_args())
