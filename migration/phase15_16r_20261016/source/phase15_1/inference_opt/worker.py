import os
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
import torch
from pi05.model import build,restore
from pi05.action_adapter import to_environment
from pi05.state_adapter import StateAdapter
from phase15.recovery_policy.model import build_recovery
from phase15.recovery_policy.state import RecoveryAdapter
from phase15_1.common import AUTH,isolated
from .engine import Engine


def main(a):
    torch.set_num_threads(4)
    socket=isolated(a.socket)
    if socket.exists():raise RuntimeError('Socket belongs to an existing run')
    socket.parent.mkdir(parents=True,exist_ok=True)
    nets={'normal':build()};engines={}
    listener=Listener(str(socket),family='AF_UNIX',authkey=AUTH);os.chmod(socket,0o600)
    print('SERVER READY',flush=True)
    try:
        with listener.accept() as conn:
            while True:
                request=conn.recv();op=request['op']
                if op=='stop':conn.send(dict(done=True));break
                try:
                    if op=='load':
                        engines.clear();torch.cuda.empty_cache();metas={}
                        for module,path in request['checkpoints'].items():
                            if module not in nets:nets[module]=build_recovery()
                            net=nets[module]
                            # Restore exact historical trainable schema, then freeze all parameters.
                            for name,p in net.named_parameters():
                                p.requires_grad_('.A' in name or '.B' in name or name.startswith(
                                    ('action_in_proj.','action_out_proj.','time_mlp_in.','time_mlp_out.')))
                            meta=restore(net,path);net.eval().requires_grad_(False)
                            adapter=(RecoveryAdapter if module=='recovery' else StateAdapter)(meta['mean'],meta['std'])
                            engines[module]=Engine(net,adapter,a.backend)
                            metas[module]={k:v for k,v in meta.items() if k!='learned'}
                        conn.send(dict(meta=metas))
                    elif op=='infer':
                        tick=time.perf_counter();engine=engines[request['module']]
                        output,latency=engine.run(request['raw'],request['noise_seeds'],request.get('horizon',10))
                        # Original adapter accepts variable horizon without repeating actions.
                        chunk=output.float().cpu().numpy()
                        if not __import__('numpy').isfinite(chunk).all():raise RuntimeError('Nonfinite action')
                        active=chunk[:,:,:7]
                        actions=__import__('numpy').clip(active,-1,1).astype('float32')
                        clipping=float((abs(active)>1).mean())
                        conn.send(dict(actions=actions,latency_s=latency,service_s=time.perf_counter()-tick,
                            clip_fraction=clipping,engineering=engine.status()))
                    elif op=='status':conn.send({k:v.status() for k,v in engines.items()})
                    else:raise ValueError(op)
                except BaseException:
                    conn.send(dict(error=traceback.format_exc()));raise
    finally:
        listener.close()
        if socket.exists():socket.unlink()
