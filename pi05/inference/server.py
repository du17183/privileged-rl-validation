"""Local Unix-socket inference service isolates OpenPI/Isaac Python runtimes."""
import argparse,json,os,time,traceback
from multiprocessing.connection import Listener
from pathlib import Path
import numpy as np,torch
from pi05.model import build,observation,restore
from pi05.state_adapter import StateAdapter
from pi05.action_adapter import to_environment

def main(a):
    torch.set_num_threads(4);sock=Path(a.socket).resolve();root=Path(__file__).resolve().parents[2]/'results/phase14_3_pi05_bc'
    assert sock.parent==root.resolve() and not sock.exists()
    net=build();net.eval();listener=Listener(str(sock),family='AF_UNIX',authkey=b'phase14_3_local');os.chmod(sock,0o600)
    adapter=None;timings=[];meta=None
    print('SERVER READY',flush=True)
    try:
        with listener.accept() as conn:
            while True:
                request=conn.recv();op=request['op']
                if op=='stop':conn.send({'done':True});break
                try:
                    if op=='load':
                        meta=restore(net,request['checkpoint']);adapter=StateAdapter(meta['mean'],meta['std']);timings=[];conn.send({'meta':{k:v for k,v in meta.items() if k!='learned'}})
                    elif op=='infer':
                        raw=request['raw'];tokens,mask=adapter.encode(raw);obs=observation(raw,tokens,mask,'cuda:0')
                        noise=np.stack([np.random.default_rng(int(s)).normal(size=(10,32)).astype(np.float32) for s in request['noise_seeds']])
                        torch.cuda.synchronize();start=time.monotonic()
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):chunk=net.sample_actions('cuda:0',obs,noise=torch.tensor(noise,device='cuda:0'),num_steps=10)
                        torch.cuda.synchronize();elapsed=time.monotonic()-start;timings.append(elapsed)
                        action,clip=to_environment(chunk.float().cpu().numpy());conn.send({'actions':action,'latency_s':elapsed,'clip_fraction':clip})
                    elif op=='stats':conn.send({'calls':len(timings),'mean_s':float(np.mean(timings)) if timings else None,'p95_s':float(np.quantile(timings,.95)) if timings else None,'peak_gpu_GB':torch.cuda.max_memory_allocated()/1e9})
                    else:raise ValueError(op)
                except Exception:conn.send({'error':traceback.format_exc()});raise
    finally:
        listener.close()
        if sock.exists():sock.unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);main(p.parse_args())
