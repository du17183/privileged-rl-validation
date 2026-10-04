#!/usr/bin/env python3
"""Restore all non-weight experiment data, metrics and logs from a public Release."""
import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER_REPO = 'du17183/privileged-rl-validation'


def headers():
    result = {'User-Agent': 'privileged-rl-validation-restore', 'Accept': 'application/vnd.github+json'}
    token = os.environ.get('GITHUB_TOKEN')
    if token: result['Authorization'] = 'Bearer ' + token
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): value.update(chunk)
    return value.hexdigest()


class JoinedReader(io.RawIOBase):
    def __init__(self, paths): self.paths = iter(paths); self.current = None
    def readable(self): return True
    def read(self, size=-1):
        if size < 0: raise ValueError('Streaming archive reads must be bounded')
        output = bytearray()
        while len(output) < size:
            if self.current is None:
                try: self.current = next(self.paths).open('rb')
                except StopIteration: break
            chunk = self.current.read(size - len(output))
            if chunk: output.extend(chunk)
            else: self.current.close(); self.current = None
        return bytes(output)
    def close(self):
        if self.current: self.current.close()
        super().close()


def download(url, path, expected):
    if path.exists() and path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256']:
        return
    temporary = path.with_name(path.name + '.download')
    request = urllib.request.Request(url, headers=headers())
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
        shutil.copyfileobj(response, output, length=8 << 20)
    if temporary.stat().st_size != expected['bytes'] or digest(temporary) != expected['sha256']:
        raise RuntimeError('Size/SHA256 mismatch: ' + path.name)
    os.replace(temporary, path)


def extract(paths, destination, overwrite):
    with JoinedReader(paths) as stream, tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for entry in archive:
            relative = Path(entry.name)
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe archive path')
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()): raise ValueError('Archive escapes destination')
            if entry.issym() or entry.islnk():
                raise ValueError('Unexpected link in experiment archive: ' + entry.name)
            if entry.isdir(): target.mkdir(parents=True, exist_ok=True); continue
            if not entry.isfile(): raise ValueError('Unexpected archive member type')
            payload = archive.extractfile(entry)
            if target.exists() and not overwrite:
                with target.open('rb') as current:
                    while True:
                        block = payload.read(8 << 20)
                        if not block: break
                        if current.read(len(block)) != block: raise FileExistsError('Different existing file; use --overwrite intentionally: ' + str(target))
                    if current.read(1): raise FileExistsError(str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('wb') as output: shutil.copyfileobj(payload, output, length=8 << 20)
            target.chmod(entry.mode & 0o777)


def main(args):
    manifest = json.loads((ROOT / 'reproducibility/release_manifest.json').read_text())
    available = manifest['groups']
    groups = list(available) if args.groups == 'all' else args.groups.split(',')
    if any(group not in available for group in groups): raise ValueError('Unknown group; valid groups: ' + ','.join(available))
    api = f"https://api.github.com/repos/{OWNER_REPO}/releases/tags/{manifest['tag']}"
    with urllib.request.urlopen(urllib.request.Request(api, headers=headers()), timeout=60) as response:
        release = json.load(response)
    assets = {item['name']:item['browser_download_url'] for item in release['assets']}
    cache = ROOT / '_release_cache'; cache.mkdir(exist_ok=True)
    destination = Path(args.destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    for group in groups:
        parts = available[group]['parts']; paths = []
        for part in parts:
            path = cache / part['name']; paths.append(path)
            print('DOWNLOAD/VERIFY', part['name'], part['bytes'], flush=True)
            download(assets[part['name']], path, part)
        if not args.download_only:
            print('RESTORE', group, available[group]['file_count'], 'files', flush=True)
            extract(paths, destination, args.overwrite)
    print('RESTORE COMPLETE; no policy weights are present.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', default='all')
    parser.add_argument('--destination', default=str(ROOT))
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    main(parser.parse_args())
