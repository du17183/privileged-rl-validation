"""Protect all preceding phases, including Phase14.2, without modifying them."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_3_pi05_bc'
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main(check):
    path=R/'preservation_before.json';R.mkdir(exist_ok=True)
    if not check:
        assert not path.exists();old={}
        for directory in ['envs','door_env','progress_rl','randomized_env','environment_state','configs','algorithms','experiments','recovery_expert','bc','datasets','results','checkpoints','docs']:
            for f in (ROOT/directory).rglob('*'):
                rel=str(f.relative_to(ROOT))
                if not f.is_file() or '__pycache__' in f.parts or 'phase14_3' in rel or f.suffix in ['.pyc','.tmp']:continue
                st=f.stat();old[rel]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
                if f.suffix in ['.py','.json','.yaml','.yml','.sh','.md'] or (f.suffix=='.h5' and 'phase14_1' in rel):old[rel]['sha256']=sha(f)
        path.write_text(json.dumps(old));print('PROTECTED',len(old),flush=True)
    else:
        before=json.loads(path.read_text());bad=[]
        for rel,v in before.items():
            f=ROOT/rel
            if not f.exists() or f.stat().st_size!=v['bytes'] or f.stat().st_mtime_ns!=v['mtime_ns'] or ('sha256' in v and sha(f)!=v['sha256']):bad.append(rel)
        (R/'preservation_after.json').write_text(json.dumps({'checked':len(before),'changed':bad},indent=2))
        if bad:raise RuntimeError(bad[:15])
        print('PRESERVED',len(before),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');main(p.parse_args().check)
