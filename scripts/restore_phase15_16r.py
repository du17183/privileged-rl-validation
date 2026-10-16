#!/usr/bin/env python3
"""Restore a SHA-verified, paused historical snapshot. Never run an experiment."""
import argparse, hashlib, io, json, os, pathlib, shutil, subprocess, sys, tarfile, time, urllib.request
TAG='migration-phase15-16r-20261016'
BASE='https://github.com/du17183/privileged-rl-validation/releases/download/'+TAG+'/'
MANIFEST_SHA256='10939fd5bfff9620354cef8dcfb124e09e8e22b389076523faf125903bb22567'
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def download(name,p,expected):
    if p.exists() and digest(p)==expected:return
    temp=p.with_name(p.name+'.partial')
    for attempt in range(6):
        try:
            offset=temp.stat().st_size if temp.exists() else 0
            req=urllib.request.Request(BASE+name,headers={'User-Agent':'phase15-16r-restore',**({'Range':f'bytes={offset}-'} if offset else {})})
            with urllib.request.urlopen(req,timeout=90) as response:
                resume=offset and response.status==206
                with temp.open('ab' if resume else 'wb') as f:shutil.copyfileobj(response,f,4<<20)
            if digest(temp)!=expected:
                temp.unlink();raise ValueError('SHA256 mismatch: '+name)
            os.replace(temp,p);return
        except Exception:
            if attempt==5:raise
            time.sleep(min(10,attempt+1))
class Joined(io.RawIOBase):
    def __init__(self,parts):self.parts=iter(parts);self.f=None
    def readable(self):return True
    def read(self,n=-1):
        if n<0:raise ValueError('Unbounded archive read disallowed')
        out=bytearray()
        while len(out)<n:
            if self.f is None:
                p=next(self.parts,None)
                if p is None:break
                self.f=p.open('rb')
            b=self.f.read(n-len(out))
            if b:out.extend(b)
            else:self.f.close();self.f=None
        return bytes(out)
    def close(self):
        if self.f:self.f.close()
        super().close()
def target(root,name):
    rel=pathlib.PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name:raise ValueError('Unsafe archive path')
    p=root.joinpath(*rel.parts)
    if not p.resolve().is_relative_to(root):raise ValueError('Path escapes destination')
    return p
def verify_files(root,manifest):
    for name,r in manifest['files'].items():
        p=target(root,name)
        if not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('Restored file mismatch: '+name)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destination',required=True,type=pathlib.Path)
    ap.add_argument('--cache',type=pathlib.Path)
    ap.add_argument('--with-dependencies',action='store_true',help='Also restore the original shared pi05 base and pinned Isaac Lab source from the earlier unchanged releases; no installation or GPU runs')
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args();root=args.destination.resolve()
    marker=root/'.phase15_16r_migration.json'
    if root.exists() and any(root.iterdir()) and not marker.exists():raise FileExistsError('Use a new empty destination; historical outputs must not be overwritten')
    root.mkdir(parents=True,exist_ok=True)
    if marker.exists() and json.loads(marker.read_text())['manifest_sha256']!=MANIFEST_SHA256:raise ValueError('Destination belongs to another snapshot')
    cache=(args.cache or root.parent/(root.name+'_download_cache')).resolve();cache.mkdir(parents=True,exist_ok=True)
    download('migration_manifest.json',cache/'migration_manifest.json',MANIFEST_SHA256)
    m=json.loads((cache/'migration_manifest.json').read_text());assert m['tag']==TAG
    marker.write_text(json.dumps({'manifest_sha256':MANIFEST_SHA256,'state':'restoring','experiments_started':False},indent=2))
    if not args.verify_only:
        for group,g in m['groups'].items():
            parts=[]
            for part in g['parts']:
                p=cache/part['name'];print('DOWNLOAD/VERIFY',part['name'],flush=True)
                download(part['name'],p,part['sha256'])
                if p.stat().st_size!=part['bytes']:raise ValueError('Archive size mismatch')
                parts.append(p)
            seen=set()
            with Joined(parts) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tar:
                for member in tar:
                    r=m['files'].get(member.name)
                    if not member.isfile() or r is None or r['group']!=group or member.name in m['weight_aliases'] or member.name in seen:raise ValueError('Unexpected archive member: '+member.name)
                    if member.size!=r['bytes']:raise ValueError('Archive member size mismatch')
                    seen.add(member.name);p=target(root,member.name)
                    if p.exists():
                        if not p.is_file() or digest(p)!=r['sha256']:raise FileExistsError('Conflicting existing file: '+member.name)
                        continue
                    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.restoring')
                    h=hashlib.sha256()
                    with tar.extractfile(member) as source,temp.open('wb') as dest:
                        for b in iter(lambda:source.read(4<<20),b''):h.update(b);dest.write(b)
                    if h.hexdigest()!=r['sha256']:temp.unlink();raise ValueError('Member SHA mismatch: '+member.name)
                    os.replace(temp,p);p.chmod(r['mode'])
            expected={n for n,r in m['files'].items() if r['group']==group and n not in m['weight_aliases']}
            if seen!=expected:raise ValueError('Missing archive members: '+group)
        for alias,source in m['weight_aliases'].items():
            src=target(root,source);dest=target(root,alias)
            if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        if args.with_dependencies:
            subprocess.run([sys.executable,str(root/'scripts/restore_weights.py'),'--groups','pi05_base','--destination',str(root)],check=True)
            subprocess.run([sys.executable,str(root/'scripts/restore_release.py'),'--groups','third_party','--destination',str(root)],check=True)
    verify_files(root,m)
    external=[]
    for d in m['external_dependencies']:
        p=target(root,d['path']);row={'path':d['path'],'present':p.exists()}
        if d.get('sha256'):
            row['sha256_verified']=p.is_file() and digest(p)==d['sha256']
            if args.with_dependencies and not row['sha256_verified']:raise ValueError('External base SHA mismatch')
        if args.with_dependencies and not p.exists():raise ValueError('External source dependency absent: '+d['path'])
        external.append(row)
    report={'complete':True,'files_sha256_verified':len(m['files']),'manifest_sha256':MANIFEST_SHA256,'external_dependencies':external,'experiments_started':False,'runtime_installation_performed':False,'historical_pid_socket_lock_are_not_resume_state':True}
    marker.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
