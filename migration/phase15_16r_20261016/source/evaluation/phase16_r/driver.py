"""Fresh physics process per paired rollout, persistent frozen inference."""
import argparse,sys,os,json,time,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np,h5py,torch
from multiprocessing.connection import Client
from experiments.phase16_r_residual_sac.common import ROOT,EXP,RES,AUTH,save,sha,utc,protocol,checkpoint
from algorithms.residual_sac.gates import numerical_gate

def proc(pid):
    p=Path('/proc',str(pid))
    if not p.exists():return dict(pid=pid,alive=False)
    s=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,alive=True,ppid=int(s[1]),start_ticks=int(s[19]),argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())

def gate(out,P):
    errors={}
    with h5py.File(out/'A.h5') as a,h5py.File(out/'zero.h5') as b:
        for k in ['base','executed','raw','next_robot','next_gt','reward_original','reward_signed','done','cursor','chunk']:
            assert a[k].shape==b[k].shape,k
            d=float(np.abs(a[k][:].astype(float)-b[k][:].astype(float)).max());errors[k]=d
        save(out/'zero_equivalence_measured.json',dict(errors=errors,tolerance=1e-6,fresh_physics_processes=True,strict_actor_mean_composition=True))
        if any(v>1e-6 for v in errors.values()):raise RuntimeError('Zero equivalence failed '+str(errors))
        ar=json.loads((out/'A.json').read_text());br=json.loads((out/'zero.json').read_text());assert ar['success']==br['success']
        for name in ['small','medium']:
            with h5py.File(out/(name+'.h5')) as c:
                assert not np.array_equal(c['executed'][0],a['executed'][0])
                d=float(np.abs(c['next_robot'][0,:,:7]-a['next_robot'][0,:,:7]).max());assert d>1e-8
                assert np.array_equal(c['executed'][:,:,6],c['base'][:,:,6]);errors[name+'_actual_joint_delta_tick1_max']=d
        meta=torch.load(checkpoint(0),map_location='cpu',weights_only=False);mean=np.asarray(meta['mean'],np.float32);std=np.asarray(meta['std'],np.float32)
        raw=a['raw'][:];base=a['base'][:];chunk=a['chunk'][:];cur=a['cursor'][:]
        context=np.concatenate((2/np.pi*np.arctan((raw-mean)/std),base,chunk.reshape(*raw.shape[:2],70),np.eye(5,dtype=np.float32)[cur]),-1).astype(np.float32)
        done=a['done'][:-1].reshape(-1,1);nextctx=context[1:].reshape(-1,121).copy();nextbase=base[1:].reshape(-1,7).copy();nextctx[done[:,0]]=0;nextbase[done[:,0]]=0
        ids=np.r_[np.random.default_rng(160123).integers(0,len(done),240),np.flatnonzero(done[:,0])[:16]]
        batch=dict(context=context[:-1].reshape(-1,121)[ids],base=base[:-1].reshape(-1,7)[ids],next_context=nextctx[ids],next_base=nextbase[ids],delta=a['delta'][:-1].reshape(-1,6)[ids],executed=a['executed'][:-1].reshape(-1,7)[ids],reward=a['reward_original'][:-1].reshape(-1,1)[ids],done=done[ids].astype(np.float32))
        result=numerical_gate({k:torch.as_tensor(v) for k,v in batch.items()},'cpu')
        save(out/'sac_real_gate.json',dict(utc=utc(),**result,real_source='closed frozen-A physical trace; next actual base and chunk from following tick',source_trace_sha256=sha(out/'A.h5'),main_trial_updates=0))
    save(out/'physical_gate.json',dict(utc=utc(),passed=True,errors=errors,zero_success=ar['success'],fresh_process_per_rollout=True,small_medium_not_performance_selection=True))

def main(args):
    P=protocol();torch.set_num_threads(4);job=json.loads(Path(args.job).read_text());out=ROOT/job['output'];cases=[];children=[]
    def add(name,arm,resetseed,n,level=2,det=False,max_ticks=None):
        cases.append(dict(mode='eval_rollout',seed=job['seed'],arm=arm,output=job['output'],casefile=name,resetseed=resetseed,nscored=n,level=level,det=det,max_ticks=max_ticks,keep_inference=True,**{k:job[k] for k in ['checkpoint','checkpoint_sha256'] if k in job}))
    if job['mode']=='engineering':
        for arm in ['A','zero','small','medium']:add(arm,arm,P['seeds']['engineering_reset'],16,max_ticks=60 if arm in ['small','medium'] else None)
    elif job['mode']=='baseline':
        for arm in ['A','B']:
            for scope,n,level,seed in [('random',128,2,P['seeds']['test_reset']),('validation',64,2,P['seeds']['validation_reset']),('fixed',32,0,P['seeds']['fixed_reset'])]:add(f'{arm}_{scope}',arm,seed,n,level)
    elif job['mode']=='evalnode':
        for det in [False,True]:add('validation_'+('mean' if det else 'sampled'),job['arm'],P['seeds']['validation_reset'],64,det=det)
        if job['step']==50000 or job.get('final_test'):
            for scope,n,level,seed in [('random',128,2,P['seeds']['test_reset']),('fixed',32,0,P['seeds']['fixed_reset'])]:
                for det in [False,True]:add(scope+'_'+('mean' if det else 'sampled'),job['arm'],seed,n,level,det)
    else:raise ValueError(job['mode'])
    try:
        for case in cases:
            casepath=EXP/'jobs'/(Path(args.job).stem+'_'+case['casefile']+'.json');save(casepath,case)
            command=[str(ROOT/'.venv/bin/python'),'-u','-m','evaluation.phase16_r.runner','--job',str(casepath),'--socket',args.socket,'--device',args.device]
            log=ROOT/'logs/phase16_r'/(casepath.stem+'_physics.log')
            with log.open('x') as f:
                process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                time.sleep(.1);record=dict(casefile=case['casefile'],job=str(casepath.relative_to(ROOT)),job_sha256=sha(casepath),process=proc(process.pid));children.append(record);save(out/'physics_children.json',dict(utc=utc(),driver=proc(os.getpid()),children=children))
                print('PHYSICS_CASE_LAUNCHED '+json.dumps(record),flush=True);rc=process.wait()
                record.update(exit_code=rc,exit_utc=utc(),after=proc(process.pid));save(out/'physics_children.json',dict(utc=utc(),driver=proc(os.getpid()),children=children))
            if rc:raise RuntimeError('Affected physics case failed '+case['casefile']+' '+str(rc))
        if job['mode']=='engineering':gate(out,P)
        save(out/'complete.json',dict(utc=utc(),passed=True,seed=job['seed'],mode=job['mode'],physical_processes=len(children),checkpoint_sha256=job.get('checkpoint_sha256'),rl_updates=0 if job['mode']!='evalnode' else None))
        save(out/'worker_complete.json',dict(utc=utc(),passed=True,pid=os.getpid(),fresh_physics_processes=len(children)))
    finally:
        try:
            with Client(args.socket,family='AF_UNIX',authkey=AUTH) as conn:conn.send(dict(op='stop'));conn.recv()
        except (FileNotFoundError,ConnectionRefusedError,EOFError):pass

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--socket',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    try:main(a)
    except BaseException:
        error=traceback.format_exc();print(error,flush=True);job=json.loads(Path(a.job).read_text());save(ROOT/job['output']/'failed.json',dict(utc=utc(),error=error,pid=os.getpid()));sys.exit(1)
