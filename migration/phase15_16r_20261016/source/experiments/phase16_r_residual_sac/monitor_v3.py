"""Read-only source, frozen checkpoint, parent and fresh-child identity monitor."""
import sys,json,subprocess,os
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,save,sha,utc,protocol
from experiments.phase16_r_residual_sac.monitor import proc as basic_proc

def proc(pid):
    try:
        result=basic_proc(pid)
        if result.get('alive'):
            state=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0]
            result['state']=state
            if state in ['Z','X']:
                result.update(alive=False,exited_pending_reap=True)
        return result
    except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,alive=False,exited_during_snapshot=True)

def main():
    protocol();issues=[];identities=[]
    m=json.loads((RES/'source_manifest.json').read_text())
    for rel,wanted in m['files'].items():
        if sha(ROOT/rel)!=wanted:issues.append('Source SHA drift '+rel)
    h=json.loads((RES/'historical_inputs.json').read_text())
    for seed,row in h['normal'].items():
        if sha(ROOT/row['checkpoint'])!=row['sha256']:issues.append('Frozen Normal SHA drift '+seed)
    life=json.loads((RES/'lifecycle.json').read_text());expected=[life['supervisor'],*life.get('launched',[])]
    for path in RES.rglob('physics_children.json'):
        if any(part.startswith('attempt') for part in path.parts):continue
        x=json.loads(path.read_text())
        for row in x['children']:
            if sha(ROOT/row['job'])!=row['job_sha256']:issues.append('Child job SHA drift '+row['job'])
            expected.append(row['process'])
    for old in expected:
        now=proc(old['pid']);now['expected']=old
        if now['alive'] and any(now.get(k)!=old.get(k) for k in ['argv','ppid','start_ticks']):
            now['pid_reused']=True
            if now.get('start_ticks')==old.get('start_ticks'):issues.append('Live identity mismatch '+str(old['pid']))
        if now['alive'] and not now.get('pid_reused'):
            path=Path('/proc',str(old['pid']));s=(path/'stat').read_text().rsplit(')',1)[1].split()
            now['cpu_user_ticks']=int(s[11]);now['cpu_system_ticks']=int(s[12]);now['wchan']=(path/'wchan').read_text().strip()
        identities.append(now)
    active=[x for x in identities if x['alive'] and not x.get('pid_reused')]
    progress=[json.loads(p.read_text()) for p in (RES/'train').glob('*/progress.json')]
    baseline=[]
    for p in (RES/'baseline').glob('seed*/A_random.json'):
        x=json.loads(p.read_text());baseline.append(dict(seed=x['model_seed'],episodes=x['episodes'],success=x['success']))
    engineering={}
    for name in ['A','zero','small','medium','physical_gate','sac_real_gate','failed']:
        path=RES/'engineering'/f'{name}.json'
        if path.exists():
            x=json.loads(path.read_text());engineering[name]={k:v for k,v in x.items() if k not in ['calls','error']}
            if name=='failed':engineering[name]['error_tail']=x['error'][-1000:]
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
    snapshot=dict(utc=utc(),passed=not issues,issues=issues,monitor_source_sha256=sha(Path(__file__)),protocol_sha256=sha(EXP/'protocol.json'),source_manifest_sha256=sha(RES/'source_manifest.json'),processes=identities,engineering=engineering,baseline_rows=baseline,training_progress=progress,compute_apps=apps,gpu=gpu,cpu_load=os.getloadavg(),Frozen_pi05_baseline_ready=(RES/'Frozen_pi05_baseline.json').exists(),first_round_complete=(RES/'first_round_complete.json').exists(),pipeline_failed=(RES/'pipeline_failed.json').exists())
    save(RES/'monitor'/f'{utc().replace(":","")}_v3.json',snapshot)
    print(json.dumps({k:v for k,v in snapshot.items() if k not in ['processes','engineering']},indent=2));print('ACTIVE_IDENTITIES',json.dumps(active));print('ENGINEERING',json.dumps(engineering))
if __name__=='__main__':main()
