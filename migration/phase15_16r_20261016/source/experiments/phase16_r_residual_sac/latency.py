"""Exclusive batch1 inference and residual microbench; no physical episode."""
import sys,os,json,time,subprocess,traceback,csv
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np,h5py,torch
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,AUTH,sha,save,utc,protocol,checkpoint
from experiments.phase16_r_residual_sac.monitor import proc
from evaluation.phase16_r.chunk import ChunkClient
from algorithms.residual_sac.sac import SAC

def main():
    P=protocol();out=RES/'latency';out.mkdir(parents=True,exist_ok=True)
    if (out/'launch.json').exists():raise RuntimeError('Latency already launched; inspect original identity before any retry')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);torch.set_num_threads(4)
    def snapshot():
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
        gpus=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
        uuid={int(line.split(',')[0]):line.split(',')[1].strip() for line in gpus.strip().splitlines()}[gpu]
        occupied=[line for line in apps.splitlines() if line.split(',')[0].strip()==uuid]
        return dict(utc=utc(),gpu=gpu,compute_apps=apps,gpus=gpus,selected_gpu_compute=occupied)
    before=snapshot();assert not before['selected_gpu_compute'],before
    sock=RES/'lat.sock';log=ROOT/'logs/phase16_r/latency001_inference.log'
    cmd=[str(ROOT/'.venv_pi05/bin/python'),'-u','-m','evaluation.phase16_r.inference_worker','--socket',str(sock),'--backend','graph','--persistent']
    save(out/'launch.json',dict(utc=utc(),process=proc(os.getpid()),before=before,source_sha256=sha(Path(__file__)),argv=cmd))
    start=time.perf_counter();samples=[];rows=[];status=[]
    with log.open('x') as logstream:
        inf=subprocess.Popen(cmd,cwd=ROOT,stdout=logstream,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(.1);save(out/'inference_launch.json',dict(utc=utc(),process=proc(inf.pid)))
        client=None
        try:
            client=ChunkClient(sock,1,161617)
            startup=time.perf_counter()-start
            with h5py.File(RES/'engineering/A.h5') as h:raw=h['raw'][:7].reshape(-1,39)[:100].copy()
            assert len(raw)==100
            for seed in range(5):
                load=time.perf_counter();client.load(checkpoint(seed));load_s=time.perf_counter()-load
                for i in range(10):client.rpc(dict(op='infer',module='normal',raw=raw[i:i+1],noise_seeds=[16160000+i],horizon=10))
                chunks=[]
                for i in range(100):
                    begin=time.perf_counter();r=client.rpc(dict(op='infer',module='normal',raw=raw[i:i+1],noise_seeds=[16160000+i],horizon=10));rpc=time.perf_counter()-begin
                    chunks.append(r['actions'][0]);samples.append(dict(method='Normal',seed=seed,call=i,batch=1,forward_s=r['latency_s'],service_s=r['service_s'],rpc_s=rpc,load_s=load_s,startup_s=startup))
                status.append(dict(seed=seed,status=client.rpc(dict(op='status'))))
                arr=[x for x in samples if x['method']=='Normal' and x['seed']==seed]
                for clock in ['forward_s','service_s','rpc_s']:
                    a=np.asarray([x[clock] for x in arr]);rows.append(dict(method='Normal',seed=seed,distribution='pi_flow',clock=clock,n=len(a),p50_s=float(np.quantile(a,.5)),p95_s=float(np.quantile(a,.95)),p99_s=float(np.quantile(a,.99)),scope='GPU exclusive batch1 H10 query; executes5 at60Hz; not robot end-to-end'))
                if seed<3:
                    chunks=np.asarray(chunks);base=chunks[:,0,:];ctx=np.concatenate((2/np.pi*np.arctan((raw-client.mean)/client.std),base,chunks.reshape(100,70),np.tile(np.eye(5,dtype=np.float32)[0],(100,1))),-1).astype(np.float32)
                    for arm in ['C','D']:
                        path=ROOT/f'checkpoints/phase16_r/{arm}_seed{seed}/step50000.pt'
                        if not path.exists():continue
                        s=SAC(seed,P['residual']['small_bound'],'cuda:0',P['sac']);s.restore(torch.load(path,map_location='cuda:0',weights_only=False))
                        for det in [False,True]:
                            c=torch.as_tensor(ctx,device='cuda:0');b=torch.as_tensor(base,device='cuda:0')
                            with torch.no_grad():
                                for i in range(10):s.policy(c[i:i+1],b[i:i+1],deterministic=det)
                                torch.cuda.synchronize()
                                for i in range(100):
                                    event_a=torch.cuda.Event(enable_timing=True);event_b=torch.cuda.Event(enable_timing=True)
                                    wall=time.perf_counter();cpu=time.thread_time();event_a.record();s.policy(c[i:i+1],b[i:i+1],deterministic=det);event_b.record();torch.cuda.synchronize()
                                    samples.append(dict(method=arm,seed=seed,call=i,batch=1,distribution='mean' if det else 'sampled',wall_s=time.perf_counter()-wall,thread_s=time.thread_time()-cpu,cuda_event_s=event_a.elapsed_time(event_b)/1000))
                            arr=[x for x in samples if x['method']==arm and x['seed']==seed and x['distribution']==('mean' if det else 'sampled')]
                            for clock in ['wall_s','thread_s','cuda_event_s']:
                                a=np.asarray([x[clock] for x in arr]);rows.append(dict(method=arm,seed=seed,distribution='mean' if det else 'sampled',clock=clock,n=100,p50_s=float(np.quantile(a,.5)),p95_s=float(np.quantile(a,.95)),p99_s=float(np.quantile(a,.99)),scope='batch1 residual actor and composition only; no physics, input IPC or Normal included'))
                        del s
            after=snapshot();known={str(os.getpid()),str(inf.pid)}
            assert all(line.split(',')[1].strip() in known for line in after['selected_gpu_compute']),after
            client.stop();client=None;rc=inf.wait(timeout=90);assert rc==0
            keys=sorted(set().union(*(row.keys() for row in samples)))
            with (out/'calls.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(samples)
            with (out/'quantiles.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            save(out/'complete.json',dict(utc=utc(),passed=True,source_sha256=sha(Path(__file__)),physical_interactions=0,normal_samples=500,warmup_calls=50,rows=rows,inference_status=status,resource_before=before,resource_after=after,inference_exit_code=rc,inference_after=proc(inf.pid),total_wall_s=time.perf_counter()-start,calls_sha256=sha(out/'calls.csv'),quantiles_sha256=sha(out/'quantiles.csv'),not_robot_end_to_end=True,startup_and_load_excluded_from_quantiles=True))
            print(json.dumps(dict(passed=True,rows=rows)),flush=True)
        except BaseException:
            save(out/'failed.json',dict(utc=utc(),error=traceback.format_exc(),inference=proc(inf.pid),signals_sent=[]));raise
        finally:
            if client is not None:
                try:client.stop()
                except (EOFError,ConnectionRefusedError):pass
if __name__=='__main__':main()
