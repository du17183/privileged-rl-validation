"""Two frozen inference instances; separate simulator and OpenPI runtimes."""
import argparse
import os
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
import numpy as np
import torch
from pi05.model import build, restore, observation
from pi05.state_adapter import StateAdapter
from pi05.action_adapter import to_environment
from phase15.common import RESULTS, AUTH
from phase15.recovery_policy.model import build_recovery
from phase15.recovery_policy.state import RecoveryAdapter


def main(a):
    torch.set_num_threads(4)
    socket = Path(a.socket).resolve()
    if not socket.is_relative_to(RESULTS.resolve()) or socket.exists():
        raise RuntimeError('Socket must be a fresh Phase15 artifact')
    socket.parent.mkdir(parents=True, exist_ok=True)
    normal = build()
    normal.eval().requires_grad_(False)
    recovery = None
    nets = {'normal': normal}
    adapters = {}
    listener = Listener(str(socket), family='AF_UNIX', authkey=AUTH)
    os.chmod(socket, 0o600)
    print('SERVER READY', flush=True)
    try:
        with listener.accept() as conn:
            while True:
                request = conn.recv()
                if request['op'] == 'stop':
                    conn.send({'done': True})
                    break
                try:
                    if request['op'] == 'load':
                        metas = {}
                        for module, path in request['checkpoints'].items():
                            if module == 'recovery' and recovery is None:
                                recovery = build_recovery()
                                recovery.eval()
                                # restore uses the original trainable parameter schema.
                                nets['recovery'] = recovery
                            net = nets[module]
                            # Keep the LoRA requires_grad schema for strict restore;
                            # inference has no gradients and no optimizer.
                            for name, p in net.named_parameters():
                                p.requires_grad_('.A' in name or '.B' in name or name.startswith(
                                    ('action_in_proj.', 'action_out_proj.', 'time_mlp_in.', 'time_mlp_out.')))
                            meta = restore(net, path)
                            adapters[module] = (RecoveryAdapter if module == 'recovery' else StateAdapter)(meta['mean'], meta['std'])
                            metas[module] = {k: v for k, v in meta.items() if k != 'learned'}
                        conn.send({'meta': metas})
                    elif request['op'] == 'infer':
                        module = request['module']
                        wall = time.monotonic()
                        raw = request['raw']
                        tokens, mask = adapters[module].encode(raw)
                        obs = observation(raw, tokens, mask, 'cuda:0')
                        noise = np.stack([np.random.default_rng(int(s)).normal(size=(10, 32)).astype(np.float32)
                                          for s in request['noise_seeds']])
                        torch.cuda.synchronize()
                        tick = time.monotonic()
                        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                            chunk = nets[module].sample_actions('cuda:0', obs,
                                noise=torch.tensor(noise, device='cuda:0'), num_steps=10)
                        torch.cuda.synchronize()
                        elapsed = time.monotonic() - tick
                        actions, clipping = to_environment(chunk.float().cpu().numpy())
                        conn.send({'actions': actions, 'latency_s': elapsed,
                                   'service_s': time.monotonic() - wall, 'clip_fraction': clipping})
                    else:
                        raise ValueError(request['op'])
                except Exception:
                    conn.send({'error': traceback.format_exc()})
                    raise
    finally:
        listener.close()
        if socket.exists():
            socket.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--socket', required=True)
    main(p.parse_args())
