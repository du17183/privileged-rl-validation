import argparse,json,os,subprocess,sys,time,hashlib,shutil,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
CASES=['ee_offset','contact_loss','door_regression','stagnation']
SOURCES=['recovery_expert/collect_recovery.py','recovery_expert/perturbation_generator.py',
         'recovery_expert/failure_classifier.py','recovery_expert/phase14_1/recovery_planner.py',
         'experiments/phase14_1_recovery_generation/run.py','experiments/phase14_1_recovery_generation/analyze.py',
         'recovery_expert/recovery_planner.py','recovery_expert/kinematic_compensation.py',
         'experiments/phase13_random_expert_bc/model.py','experiments/phase13_random_expert_bc/state.py',
         'randomized_env/door_randomization.py','door_env/door.py','door_env/isaac_env.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--stage',choices=['pilot','formal'],required=True);p.add_argument('--gpus',default='2,3')
    p.add_argument('--expert',choices=['local','historical'],default='local');a=p.parse_args()
    if not a.run_id.replace('_','').replace('-','').isalnum():raise ValueError(a.run_id)
    result=ROOT/'results/phase14_1_recovery_generation'/a.run_id
    log=ROOT/'logs/phase14_1_recovery_generation'/a.run_id
    if result.exists():raise FileExistsError(result)
    result.mkdir(parents=True);log.mkdir(parents=True)
    recipe={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    for name in SOURCES:
        dst=result/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    (result/'source_hashes.json').write_text(json.dumps(recipe,indent=2))
    sizes=[36,16,16,16] if a.stage=='pilot' else [200,100,100,100]
    protocol=dict(stage=a.stage,run_id=a.run_id,expert=a.expert,
                  targets=dict(zip(CASES,sizes)),seeds=dict(zip(CASES,[141110,141111,141112,141113] if a.stage=='pilot' else [141201,141202,141203,141204])),
                  prefix='frozen Phase13 selected BC+GT, deterministic, no expert prefix substitution',
                  perturbation='EEF +/-5/10/20mm x/y/z via existing Cartesian actions; gripper opening; explicit door30deg->20deg; physical stagnation',
                  envs=32,workers=len(a.gpus.split(',')),gpus=a.gpus.split(','),randomization='existing Level2',episode_seconds=10,
                  bc=False,rl=False,reward_change=False,
                  gate='overall>0.90 AND ee_offset>0.90; invalid injection outcomes separately retained')
    (result/'protocol.json').write_text(json.dumps(protocol,indent=2))
    state={};lock=threading.Lock();gpu_ids=a.gpus.split(',')
    def mark(case,**values):
        with lock:
            state[case]=dict(utc=datetime.now(timezone.utc).isoformat(),**values)
            temporary=result/'jobs.tmp';temporary.write_text(json.dumps(state,indent=2));temporary.replace(result/'jobs.json')
        print(json.dumps(dict(case=case,**values)),flush=True)
    def run_case(index):
        case=CASES[index];gpu=gpu_ids[index%len(gpu_ids)]
        # Thread groups own GPUs; no simultaneous processes on a shared slot.
        environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']=gpu
        environment['OMP_NUM_THREADS']='4';environment['MKL_NUM_THREADS']='4'
        data=ROOT/'datasets/recovery_expert/phase14_1'/a.run_id/case
        command=['nice','-n','10',sys.executable,'-u','-m','recovery_expert.collect_recovery',
                 '--case',case,'--episodes',str(sizes[index]),'--seed',str(protocol['seeds'][case]),
                 '--max-attempts',str(max(320,sizes[index]*12)),'--expert',a.expert,
                 '--checkpoint',str(ROOT/'checkpoints/phase13_random_expert_bc/randomized_gt_candidate_anchor.pt'),
                 '--output',str(data),'--device','cuda:0']
        with (log/(case+'.log')).open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            mark(case,status='running',pid=child.pid,gpu=gpu,target=sizes[index],data=str(data))
            try:rc=child.wait(timeout=4200)
            except subprocess.TimeoutExpired:
                # Only the process created above is terminated; datasets/logs retained.
                child.terminate();mark(case,status='timeout',pid=child.pid);raise
        if rc:mark(case,status='failed',exit_code=rc);raise RuntimeError(case)
        current={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
        if current!=recipe:raise RuntimeError('Source changed during experiment')
        mark(case,status='complete',summary=json.loads((data/'summary.json').read_text()))
    def group(slot):
        for index in range(slot,len(CASES),len(gpu_ids)):run_case(index)
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures=[pool.submit(group,slot) for slot in range(len(gpu_ids))]
        for future in futures:future.result()
    subprocess.run([sys.executable,'-m','experiments.phase14_1_recovery_generation.analyze','--run-id',a.run_id],cwd=ROOT,check=True)


if __name__=='__main__':main()
