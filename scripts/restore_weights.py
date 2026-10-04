#!/usr/bin/env python3
"""Download, SHA256-check and restore selected task models and shared pi05 base."""
import argparse
import json
import shutil
from pathlib import Path
from restore_release import download, extract, digest

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    manifest = json.loads((ROOT/'reproducibility/selected_weights_manifest.json').read_text())
    records = [json.loads(line) for line in (ROOT/'reproducibility/selected_weights_catalog.jsonl').read_text().splitlines() if line]
    available = manifest['groups']
    groups = list(available) if args.groups == 'all' else args.groups.split(',')
    if any(g not in available for g in groups): raise ValueError('Valid groups: '+','.join(available))
    destination = Path(args.destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    cache = ROOT/'_weight_release_cache'; cache.mkdir(exist_ok=True)
    base = f"https://github.com/{manifest['owner_repo']}/releases/download/{manifest['tag']}/"
    for group in groups:
        parts = available[group]['parts']; paths = []
        for item in parts:
            path = cache/item['name']; paths.append(path)
            print('DOWNLOAD/VERIFY', item['name'], item['bytes'], flush=True)
            download(base+item['name'], path, item)
        if args.download_only: continue
        print('RESTORE', group, available[group]['file_count'], 'models', flush=True)
        extract(paths, destination, args.overwrite)
        selected = [r for r in records if r['group'] == group]
        assert len(selected) == available[group]['file_count']
        for item in selected:
            path = destination/item['path']
            if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise RuntimeError('Restored model mismatch: '+item['path'])
        if group == 'pi05_base':
            folder = destination/'external_weights/pi05_base'
            for source, name in [('docs/WEIGHT_NOTICE.txt','NOTICE.txt'),
                                 ('third_party/openpi/LICENSE','LICENSE_OPENPI.txt'),
                                 ('third_party/openpi/LICENSE_GEMMA.txt','LICENSE_GEMMA.txt')]:
                path = folder/name; payload = (ROOT/source).read_bytes()
                if path.exists() and path.read_bytes() != payload and not args.overwrite:
                    raise FileExistsError('Different existing notice: '+str(path))
                path.write_bytes(payload)
        print('MODEL SHA256 VERIFIED', group, len(selected), flush=True)
    print('SELECTED WEIGHTS RESTORED; original experiment configurations apply.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', default='all')
    parser.add_argument('--destination', default=str(ROOT))
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    main(parser.parse_args())
