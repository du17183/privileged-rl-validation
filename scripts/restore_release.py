#!/usr/bin/env python3
"""Restore all non-weight experiment data, metrics and logs from a public Release."""
import argparse
import hashlib
import http.client
import io
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER_REPO = 'du17183/privileged-rl-validation'


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
    for attempt in range(3):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > expected['bytes']: offset = 0
        if offset == expected['bytes']:
            if digest(temporary) != expected['sha256']:
                raise RuntimeError('SHA256 mismatch in completed partial file: '+temporary.name)
            os.replace(temporary, path); return
        # Public assets redirect to a signed CDN URL; never forward a PAT.
        headers = {'User-Agent':'privileged-rl-validation-restore'}
        if offset: headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status == 206:
                    content_range = response.headers.get('Content-Range','')
                    if not content_range.startswith(f'bytes {offset}-'):
                        raise RuntimeError('Unexpected byte range: '+path.name)
                elif response.status == 200:
                    offset = 0
                else: raise RuntimeError('Unexpected download status: '+str(response.status))
                mode = 'ab' if offset else 'wb'; last = time.monotonic()
                with temporary.open(mode) as output:
                    read = getattr(response, 'read1', response.read)
                    while True:
                        block = read(64 << 10)
                        if not block: break
                        output.write(block); offset += len(block)
                        if time.monotonic()-last >= 20:
                            output.flush()
                            print('DOWNLOAD PROGRESS',path.name,offset,'/',expected['bytes'],flush=True)
                            last = time.monotonic()
            if temporary.stat().st_size != expected['bytes']:
                raise http.client.IncompleteRead(b'', expected['bytes']-temporary.stat().st_size)
            if digest(temporary) != expected['sha256']:
                raise RuntimeError('SHA256 mismatch: '+path.name)
            os.replace(temporary, path); return
        except (TimeoutError, ConnectionError, urllib.error.URLError, http.client.IncompleteRead):
            if attempt == 2: raise
            print('DOWNLOAD RETRY',path.name,attempt+1,flush=True)
            time.sleep(2*(attempt+1))


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
    # The committed manifest already names each public asset. Avoid anonymous
    # GitHub API limits by using the stable browser download URLs directly.
    base = f"https://github.com/{OWNER_REPO}/releases/download/{manifest['tag']}/"
    cache = ROOT / '_release_cache'; cache.mkdir(exist_ok=True)
    destination = Path(args.destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    for group in groups:
        parts = available[group]['parts']; paths = []
        for part in parts:
            path = cache / part['name']; paths.append(path)
            print('DOWNLOAD/VERIFY', part['name'], part['bytes'], flush=True)
            download(base + part['name'], path, part)
        if not args.download_only:
            print('RESTORE', group, available[group]['file_count'], 'files', flush=True)
            extract(paths, destination, args.overwrite)
    print('DATA RELEASE RESTORED; selected models are restored by restore_weights.py.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', default='all')
    parser.add_argument('--destination', default=str(ROOT))
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    main(parser.parse_args())
