"""Read-only identity and progress snapshot; never launches a worker."""
import json,os,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,sha,save,utc,protocol

def proc(pid):
    p=Path(f'/proc/{pid}')
    if not p.exists():return dict(pid=pid,alive=False)
    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,alive=True,ppid=int(fields[1]),start_ticks=int(fields[19]),argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())

def main():
    protocol();issues=[]
    manifest=json.loads((RES/'source_manifest.json').read_text())
    for rel,expected in manifest['files'].items():
        if sha(ROOT/rel)!=expected:issues.append('source SHA '+rel)
    hist=json.loads((RES/'historical_inputs.json').read_text())
    for row in hist['normal'].values():
        if sha(ROOT/row['checkpoint'])!=row['sha256']:issues.append('frozen Normal changed '+row['checkpoint'])
    lifecycle=json.loads((RES/'lifecycle.json').read_text()) if (RES/'lifecycle.json').exists() else {}
    identities=[]
    for old in [lifecycle.get('supervisor',{}),*lifecycle.get('launched',[])]:
        if not old:continue
        now=proc(old['pid']);now['expected']=old
        if now['alive'] and (now['start_ticks']!=old['start_ticks'] or now['argv']!=old['argv']):
            issues.append('PID reused/identity mismatch '+str(old['pid']))
        identities.append(now)
    progress=[]
    for p in (RES/'train').glob('*/progress.json'):
        progress.append(json.loads(p.read_text()))
    failures=[dict(path=str(p.relative_to(ROOT)),content=json.loads(p.read_text())) for p in RES.rglob('failed.json')]
    complete=[str(p.relative_to(RES)) for p in RES.rglob('worker_complete.json')]
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
    snapshot=dict(utc=utc(),passed=not issues,issues=issues,processes=identities,training_progress=progress,failed=failures,completed_workers=complete,frozen_baseline_ready=(RES/'Frozen_pi05_baseline.json').exists(),first_round_complete=(RES/'first_round_complete.json').exists(),pipeline_failed=(RES/'pipeline_failed.json').exists(),gpu=gpu,compute_apps=apps,cpu_load=os.getloadavg())
    dest=RES/'monitor'/f'{snapshot["utc"].replace(":","")}.json';save(dest,snapshot)
    print(json.dumps({k:v for k,v in snapshot.items() if k!='processes'},indent=2));print('IDENTITIES',json.dumps(identities,indent=2))

if __name__=='__main__':main()
