"""Historical inference implementation, with a new confined socket namespace."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
import os,time,traceback,numpy as np
from multiprocessing.connection import Listener
import phase15_1.inference_opt.worker as original
from experiments.phase16_r_residual_sac.common import RES, AUTH

def persistent(a):
    # Exact historical build/restore/Engine/adapter; only allow new physics
    # processes to disconnect and reconnect to this experiment-owned service.
    torch.set_num_threads(4);socket=isolated(a.socket)
    if socket.exists():raise RuntimeError('Existing socket')
    net=original.build();engine=None;meta=None;loaded=None
    listener=Listener(str(socket),family='AF_UNIX',authkey=AUTH);os.chmod(socket,0o600)
    print('SERVER READY',flush=True);stop=False
    try:
        while not stop:
            with listener.accept() as conn:
                while True:
                    request=conn.recv();op=request['op']
                    if op in ['stop','disconnect']:
                        conn.send(dict(done=True));stop=op=='stop';break
                    try:
                        if op=='load':
                            paths=request['checkpoints'];assert set(paths)=={'normal'}
                            if paths!=loaded:
                                engine=None;torch.cuda.empty_cache()
                                for name,param in net.named_parameters():param.requires_grad_('.A' in name or '.B' in name or name.startswith(('action_in_proj.','action_out_proj.','time_mlp_in.','time_mlp_out.')))
                                meta=original.restore(net,paths['normal']);net.eval().requires_grad_(False)
                                adapter=original.StateAdapter(meta['mean'],meta['std']);engine=original.Engine(net,adapter,a.backend);loaded=paths.copy()
                            conn.send(dict(meta={'normal':{k:v for k,v in meta.items() if k!='learned'}}))
                        elif op=='infer':
                            assert request['module']=='normal';start=time.perf_counter()
                            output,latency=engine.run(request['raw'],request['noise_seeds'],request.get('horizon',10));chunk=output.float().cpu().numpy()
                            if not np.isfinite(chunk).all():raise RuntimeError('Nonfinite action')
                            active=chunk[:,:,:7];conn.send(dict(actions=np.clip(active,-1,1).astype('float32'),latency_s=latency,service_s=time.perf_counter()-start,clip_fraction=float((abs(active)>1).mean()),engineering=engine.status()))
                        elif op=='status':conn.send({'normal':engine.status()} if engine else {})
                        else:raise ValueError(op)
                    except BaseException:
                        conn.send(dict(error=traceback.format_exc()));raise
    finally:
        listener.close()
        if socket.exists():socket.unlink()

def isolated(path):
    p = Path(path).resolve()
    if not p.is_relative_to(RES.resolve()) or len(str(p).encode()) >= 108:
        raise RuntimeError('Invalid Phase16-R Unix socket path')
    return p

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--socket', required=True)
    p.add_argument('--backend', default='graph')
    p.add_argument('--persistent',action='store_true')
    a = p.parse_args()
    original.isolated = isolated
    original.AUTH = AUTH
    if a.persistent:persistent(a)
    else:original.main(a)
