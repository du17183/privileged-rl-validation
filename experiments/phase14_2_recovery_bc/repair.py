"""Complete the first jobs whose preliminary pressure files were archived."""
import os,sys,subprocess,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_2_recovery_bc';L=ROOT/'logs/phase14_2_recovery_bc'

def run(slot):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(slot+4),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with (L/f'repair_slot{slot}.log').open('a') as f:
        child=subprocess.Popen(['nice','-n','10',sys.executable,'-u','-m','experiments.phase14_2_recovery_bc.evaluate',
           '--jobs',str(R/'repair_jobs.json'),'--slot',str(slot),'--slots','4','--device','cuda:0'],cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        rc=child.wait()
    if rc:raise RuntimeError(slot)

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in [pool.submit(run,s) for s in range(4)]:f.result()
    jobs=json.loads((R/'repair_jobs.json').read_text())
    missing=[str(ROOT/j['output']/(t+'.json')) for j in jobs for t in j['tests'] if not (ROOT/j['output']/(t+'.json')).exists()]
    if missing:raise RuntimeError(missing)
    (R/'repair_complete.json').write_text(json.dumps(dict(completed=True,jobs=len(jobs),tests=sum(len(j['tests']) for j in jobs))))
