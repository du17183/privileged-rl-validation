"""Wait for existing physics jobs; archive only affected strict-entry pilots."""
import os
import json
import time
import subprocess
import shutil
import traceback
from pathlib import Path
from phase15_2.common import ROOT,RESULTS,LOGS,save_json,read,sha256
from phase15_1.run import identity

def run():
    source=read(RESULTS/'launch_v3.json');pid=source['supervisor_pid']
    expected='.venv/bin/python -u -m phase15_2.run --replicas-per-gpu 4 '
    while identity(pid):
        if identity(pid)!=expected:raise RuntimeError('Supervisor identity changed')
        save_json(RESULTS/'resume_entry_gate.json',dict(status='waiting_existing_pilots',manager_pid=os.getpid(),
            supervisor_pid=pid,no_worker_interruption=True,rl_updates=0))
        time.sleep(10)
    if (RESULTS/'pilot/gate.json').exists() and read(RESULTS/'pilot/gate.json')['passed']:
        raise RuntimeError('Unexpected old gate passed; must audit before proceeding')
    marker=read(RESULTS/'entry_gate_requires_rerun.json')
    if marker['affected']!=['C_R2','D_R2','E_R2']:raise RuntimeError('Unexpected rerun scope')
    lifecycle=read(RESULTS/'lifecycle.json')
    for row in lifecycle['processes'].values():
        if identity(row['pid']):raise RuntimeError('Old child still active')
    v=RESULTS/'code_versions/v4_entry_gate';v.mkdir(parents=True,exist_ok=True)
    reuse={}
    for label in ['A_R0','B_R1','B_R2','B_R3']:
        for p in (RESULTS/f'pilot/{label}_seed0').glob('*'):
            if p.is_file():reuse[str(p.relative_to(ROOT))]=sha256(p)
    save_json(v/'reuse_manifest.json',dict(files=reuse,reason='same physics/controller; entry boundary only affects CDE'))
    for label in marker['affected']:
        p=RESULTS/f'pilot/{label}_seed0'
        if p.exists():shutil.move(str(p),str(v/p.name))
    for name in ['pipeline_failed.json','entry_gate_requires_rerun.json','lifecycle.json','progress.json']:
        if (RESULTS/name).exists():shutil.move(str(RESULTS/name),str(v/name))
    p=RESULTS/'jobs/pilot_v3'
    if p.exists():shutil.move(str(p),str(v/'jobs_v3'))
    shutil.copytree(LOGS,v/'logs_before',dirs_exist_ok=True)
    for name,digest in reuse.items():
        if sha256(ROOT/name)!=digest:raise RuntimeError('Unchanged pilot evidence altered')
    result=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','phase15_2.diagnostics.unit_gate'],cwd=ROOT,check=True)
    log=LOGS/'supervisor_v4.log'
    with log.open('ab',buffering=0) as f:
        proc=subprocess.Popen(['bash','-lc','source configs/runtime_env.sh && exec .venv/bin/python -u -m phase15_2.run --replicas-per-gpu 4'],
            cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save_json(RESULTS/'launch_v4.json',dict(supervisor_pid=proc.pid,command='.venv/bin/python -u -m phase15_2.run --replicas-per-gpu 4',
        launched=time.time(),log=str(log),rl_updates=0))
    save_json(RESULTS/'resume_entry_gate.json',dict(status='restarted_scoped_pilot',supervisor_pid=proc.pid,
        archived_affected=marker['affected'],preserved_unaffected=4,no_worker_interruption=True,rl_updates=0))

if __name__=='__main__':
    try:run()
    except BaseException:
        save_json(RESULTS/'resume_entry_gate.json',dict(status='blocked',error=traceback.format_exc(),manager_pid=os.getpid(),rl_updates=0));raise
