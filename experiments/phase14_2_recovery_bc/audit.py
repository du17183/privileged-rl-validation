"""Protect preexisting sources/artifacts, including Phase14.1 raw data."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase14_2_recovery_bc/preservation_before.json'

def main(check):
    protected=[]
    for directory in ['envs','door_env','progress_rl','randomized_env','environment_state','configs','algorithms',
                      'experiments','recovery_expert','bc','datasets','results','checkpoints','docs']:
        for file in (ROOT/directory).rglob('*'):
            if not file.is_file() or '__pycache__' in file.parts:continue
            rel=str(file.relative_to(ROOT))
            if 'phase14_2' in rel or file.suffix in ['.pyc','.tmp']:continue
            protected.append(file)
    if not check:
        OUT.parent.mkdir(parents=True,exist_ok=True)
        if OUT.exists():raise FileExistsError(OUT)
        values={}
        for file in protected:
            st=file.stat();v=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
            if file.suffix in ['.py','.json','.yaml','.yml','.sh','.md'] or ('phase14_1' in str(file) and file.suffix=='.h5'):
                h=hashlib.sha256()
                with file.open('rb') as f:
                    for c in iter(lambda:f.read(1<<20),b''):h.update(c)
                v['sha256']=h.hexdigest()
            values[str(file.relative_to(ROOT))]=v
        OUT.write_text(json.dumps(values));print('protected',len(values),flush=True)
    else:
        before=json.loads(OUT.read_text());changed=[]
        for rel,v in before.items():
            f=ROOT/rel
            if not f.exists() or f.stat().st_size!=v['size'] or f.stat().st_mtime_ns!=v['mtime_ns']:changed.append(rel);continue
            if 'sha256' in v:
                h=hashlib.sha256()
                with f.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
                if h.hexdigest()!=v['sha256']:changed.append(rel)
        result=dict(checked=len(before),changed=changed)
        (OUT.parent/'preservation_after.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
        if changed:raise RuntimeError('Protected files changed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');main(p.parse_args().check)
