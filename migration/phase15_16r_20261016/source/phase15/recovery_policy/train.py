"""Matched-budget recovery LoRA fits, exact cumulative R:S ratios."""
import argparse
import csv
import json
import math
import time
import torch
import numpy as np
from pi05.model import observation, learned_state, BASE
from phase15.common import DATA, CHECKPOINTS, historical, sha256, save_json
from .model import build_recovery


def main(a):
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    ratio = {'R0': 1., 'R1': .8, 'R2': .5}[a.variant]
    output = CHECKPOINTS / 'recovery_policy' / ('pilot' if a.pilot else 'formal')
    output.mkdir(parents=True, exist_ok=True)
    stem = f'{a.variant}_seed{a.seed}'
    complete = output / (stem + '_complete.json')
    if complete.exists():
        return
    m = json.loads((DATA / 'manifest.json').read_text())
    ds = {}
    for source in ['base', 'recovery']:
        with np.load(DATA / f'{source}_train.npz') as d:
            ds[source] = {k: torch.tensor(d[k], device='cuda:0') for k in
                          ['raw', 'tokens', 'token_mask', 'actions', 'action_mask']}
    initial = historical('D', a.seed)
    start = time.monotonic()
    net = build_recovery(seed=a.seed, initial=initial)
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(.9, .95), weight_decay=.01)
    gen = torch.Generator(device='cuda:0').manual_seed(150400 + a.seed)
    torch.manual_seed(150450 + a.seed)
    checkpoints = []
    grid = sorted(set([250, 1250, 2500, 3750, 5000, a.updates]))
    total_recovery = 0
    with (output / (stem + '_training.csv')).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['update', 'flow_loss', 'gradient_norm', 'lr',
                                             'recovery_draws', 'elapsed_s', 'step_s'])
        writer.writeheader()
        for step in range(1, a.updates + 1):
            torch.cuda.synchronize()
            tick = time.monotonic()
            nr = int(math.floor(step * a.batch * ratio + 1e-7) -
                     math.floor((step - 1) * a.batch * ratio + 1e-7))
            total_recovery += nr
            nb = a.batch - nr
            ri = torch.randint(len(ds['recovery']['raw']), (nr,), generator=gen, device='cuda:0')
            si = torch.randint(len(ds['base']['raw']), (nb,), generator=gen, device='cuda:0')
            batch = {k: torch.cat((ds['recovery'][k][ri], ds['base'][k][si])) for k in ds['base']}
            obs = observation(batch['raw'], batch['tokens'], batch['token_mask'], 'cuda:0')
            y = torch.zeros(a.batch, 10, 32, device='cuda:0')
            y[:, :, :7] = batch['actions']
            warm = min(250, max(1, a.updates // 10)) if a.pilot else 250
            lr = 1e-4 * min(1, step / warm) * (.1 + .9 * .5 *
                (1 + math.cos(math.pi * max(0, step - warm) / max(1, a.updates - warm))))
            for group in opt.param_groups:
                group['lr'] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                error = net(obs, y)
                loss = (error * batch['action_mask'][:, :, None]).sum() / (32 * batch['action_mask'].sum())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
            torch.cuda.synchronize()
            if step == 1 or step % 20 == 0 or step == a.updates:
                row = dict(update=step, flow_loss=float(loss.detach()), gradient_norm=float(norm), lr=lr,
                    recovery_draws=total_recovery, elapsed_s=time.monotonic() - start,
                    step_s=time.monotonic() - tick)
                writer.writerow(row)
                f.flush()
                print(json.dumps(row), flush=True)
            if step in grid:
                checkpoint = output / f'{stem}_step{step}.pt'
                if checkpoint.exists():
                    raise RuntimeError('Checkpoint overwrite forbidden')
                payload = dict(learned=learned_state(net), variant=a.variant, seed=a.seed, update=step,
                    mode='full', input_dim=47, mean=m['mean'], std=m['std'], images_used=False,
                    chunk_horizon=10, execute_steps=5, flow_steps=10, rl_updates=0,
                    model='official_openpi_pi05_state_only_lora16_recovery',
                    initial_checkpoint=str(initial), initial_sha256=sha256(initial),
                    initial_recipe='validation-selected same-seed Phase14.3 D; identical R0/R1/R2',
                    data_manifest_sha256=sha256(DATA / 'manifest.json'), recovery_ratio=ratio,
                    padding_recipe='full32_zero_target_v2', base_checkpoint=str(BASE), target_angle=1.,
                    trainable=sum(p.numel() for p in params), batch=a.batch, extra_optimizer_updates=step,
                    selected_by_test=False, pilot=a.pilot)
                tmp = checkpoint.with_suffix('.tmp')
                torch.save(payload, tmp)
                tmp.replace(checkpoint)
                save_json(checkpoint.with_suffix('.ready.json'), dict(complete=True, update=step))
                checkpoints.append(str(checkpoint))
    save_json(complete, dict(seed=a.seed, variant=a.variant, updates=a.updates, batch=a.batch,
        recovery_ratio=ratio, recovery_draws=total_recovery, total_draws=a.updates * a.batch,
        elapsed_s=time.monotonic() - start, peak_gpu_GB=torch.cuda.max_memory_allocated() / 1e9,
        checkpoints=checkpoints, rl_updates=0, pilot=a.pilot))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['R0', 'R1', 'R2'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=5000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--pilot', action='store_true')
    main(p.parse_args())
