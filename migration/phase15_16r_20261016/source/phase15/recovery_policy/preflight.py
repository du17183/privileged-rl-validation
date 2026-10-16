"""Restore a real pilot checkpoint and test relative-geometry sensitivity."""
import argparse
import json
import time
import numpy as np
import torch
from pi05.model import restore, observation
from phase15.common import DATA, RESULTS, save_json
from .model import build_recovery
from .state import RecoveryAdapter


def main(a):
    torch.set_num_threads(4)
    net = build_recovery()
    meta = restore(net, a.checkpoint)
    net.eval()
    with np.load(DATA / 'recovery_validation.npz') as d:
        raw = d['raw'][:4].copy()
    adapter = RecoveryAdapter(meta['mean'], meta['std'])
    ids, masks = adapter.encode(raw)
    obs = observation(raw, ids, masks, 'cuda:0')
    noise = torch.tensor(np.random.default_rng(150000).normal(size=(4, 10, 32)).astype(np.float32), device='cuda:0')
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        original = net.sample_actions('cuda:0', obs, noise=noise, num_steps=10)
        altered = raw.copy()
        altered[:, 40:43] += .01
        ids2, masks2 = adapter.encode(altered)
        changed = net.sample_actions('cuda:0', observation(altered, ids2, masks2, 'cuda:0'), noise=noise, num_steps=10)
    torch.cuda.synchronize()
    difference = float((original[:, :, :7] - changed[:, :, :7]).abs().mean())
    if not torch.isfinite(original).all() or not torch.isfinite(changed).all() or difference <= 0 or np.array_equal(ids, ids2):
        raise RuntimeError('Relative geometry ignored or invalid action')
    save_json(RESULTS / 'pilot/model_gate.json', dict(checkpoint_restore=True, input_dim=47,
        pretrained_backbone=True, finite_actions=True, geometry_token_changes=int((ids != ids2).sum()),
        relative_geometry_action_difference=difference, elapsed_s=time.monotonic() - start,
        pilot_updates=meta['update'], rl_updates=0, no_test_data_used=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    main(p.parse_args())
