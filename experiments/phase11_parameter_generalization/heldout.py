"""Independent conditions/strata on saved Best/Final; no checkpoint selection."""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser();p.add_argument('--arm', choices=('A', 'B', 'C', 'D'), required=True)
p.add_argument('--seed', type=int, required=True);p.add_argument('--frozen', action='store_true')
p.add_argument('--conditional', action='store_true');AppLauncher.add_app_launcher_args(p);args = p.parse_args()
app = AppLauncher(headless=True).app
import json
import os
from pathlib import Path
import torch
import isaaclab_tasks
from randomized_env.door_randomization import create
from randomized_env.sensor_noise import SensorNoise
from experiments.phase11_parameter_generalization.agent import MeasuredSAC
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, CKPT, prepared
from experiments.phase11_parameter_generalization.metrics import evaluate
from evaluation.stratified_generalization import CONDITIONS


def atomic(path, result):
    temp = path.with_suffix('.tmp');temp.write_text(json.dumps(result));os.replace(temp, path)


def main():
    protocol = prepared()
    output = OUT/('conditional' if args.conditional else 'heldout');output.mkdir(parents=True, exist_ok=True)
    name = f'{args.arm}_seed{args.seed}'+('_frozen' if args.frozen else '')
    path = output/f'{name}.json'
    if path.exists(): raise RuntimeError('Refuse overwriting independent tests')
    initial = torch.load(ROOT/'checkpoints/phase9_safe_online'/f'P9C_seed{args.seed}'/'step_300000.pt', map_location=args.device or 'cuda:0', weights_only=False)
    anchor = torch.load(ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{args.seed}'/'best.pt', map_location=args.device or 'cuda:0', weights_only=False)
    agent = MeasuredSAC(initial, anchor, args.arm, args.device or 'cuda:0')
    env = create(32, args.device or 'cuda:0', 210000+args.seed)
    reference_material = env.nominal_material[:, :, :2].mean(1).cpu()
    results = []
    endpoints = [('initial', None)] if args.frozen else [('best', CKPT/f'P11{args.arm}_seed{args.seed}'/'best.pt'),
                                                         ('final', CKPT/f'P11{args.arm}_seed{args.seed}'/'step_300000.pt')]
    try:
        for endpoint, checkpoint in endpoints:
            if checkpoint: agent.load_state(torch.load(checkpoint, map_location=env.device, weights_only=False))
            agent.actor.eval();agent.actor.cap = .01
            if args.conditional:
                tests = [('level2', 'policy', ablation, False) for ablation in ('angle', 'angle_family', 'contact', 'handle')]
                tests += [('nominal', 'policy', None, True), ('level2', 'policy', None, True)]
            else:
                tests = [(condition, 'policy', None, False) for condition in CONDITIONS]
                tests += [('nominal', 'deterministic', None, False), ('level2', 'deterministic', None, False)]
            for condition, mode, ablation, sensor in tests:
                seed = (310000 if args.conditional else 210000)+args.seed
                noise = SensorNoise(env.device, 410000+args.seed, True) if sensor else None
                rounds = 4 if condition == 'level2' and mode == 'policy' else 2
                row = evaluate(agent.actor, agent.anchor, env, args.arm, protocol['handle_center'], seed,
                               condition, mode, rounds, ablation, noise)
                for record in row['records']:
                    # Reset adapter asserts actual PhysX material readback equals
                    # this immutable per-clone reference times sampled scale.
                    material = reference_material[record['env_index']]*record['friction_scale']
                    record.update(verified_static_friction_mean=float(material[0]), verified_dynamic_friction_mean=float(material[1]))
                row.update(endpoint=endpoint, arm=args.arm, seed=args.seed, ablation=ablation, sensor_noise=sensor,
                    checkpoint_step=0 if checkpoint is None else torch.load(checkpoint, map_location='cpu', weights_only=False)['env_steps'])
                results.append(row);atomic(output/f'{name}.partial.json', dict(results=results, completed=False))
                print(name, endpoint, condition, mode, row['metrics']['success'], flush=True)
        atomic(path, dict(arm=args.arm, seed=args.seed, frozen=args.frozen, completed=True,
            nominal_material_static_dynamic_means=reference_material.tolist(),
            total_episodes=sum(r['metrics']['episodes'] for r in results),
            total_interactions=sum(r['metrics']['eval_env_steps'] for r in results), results=results))
    finally: env.close()

if __name__ == '__main__':
    try: main()
    except BaseException:
        import traceback
        folder = OUT/('conditional' if args.conditional else 'heldout');folder.mkdir(parents=True, exist_ok=True)
        (folder/f'error_{args.arm}_seed{args.seed}.txt').write_text(traceback.format_exc());raise
    finally: app.close()
