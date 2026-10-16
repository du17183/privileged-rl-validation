"""Real simulator worker: frozen single policies or modular systems.

Historical common cohorts retained exactly. The prospective full-loop test
uses fresh randomized resets and no injected perturbation/state reconstruction.
"""
import argparse
import os
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--jobs', required=True)
p.add_argument('--slot', type=int, default=0)
p.add_argument('--slots', type=int, default=1)
p.add_argument('--socket', required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(headless=True).app

import fcntl
import json
import time
import torch
import isaaclab_tasks
from randomized_env.door_randomization import create
from door_env.isaac_env import transition_after_step
from experiments.phase14_3_pi05_bc.ordinary_eval import ordinary
from evaluation.recovery_policy_eval import pressure
from phase15.common import ROOT, RESULTS, save_json, sha256
from phase15.policy_client import ModularPolicy
from phase15.evaluation.recovery_stage_eval import summarize


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    env = create(32, a.device or 'cuda:0', 143501, level=2)
    client = ModularPolicy(a.socket, env)
    original_step = env.step

    def observed_step(action):
        result = original_step(action)
        robot, gt, done = transition_after_step(env, result[2], result[3])
        client.observe(robot, gt, done)
        return result

    env.step = observed_step
    try:
        jobs = json.loads(open(a.jobs).read())
        for job in jobs[a.slot::a.slots]:
            out = ROOT / job['output']
            if not out.resolve().is_relative_to(RESULTS.resolve()):
                raise RuntimeError('Evaluation output outside isolated Phase15')
            out.mkdir(parents=True, exist_ok=True)
            with (out / '.evaluation.lock').open('a+') as guard:
                fcntl.flock(guard, fcntl.LOCK_EX)
                if all((out / (t + '.json')).exists() for t in job['tests']):
                    continue
                checkpoints = {module: str(ROOT / path) for module, path in job['checkpoints'].items()}
                metas = client.load(checkpoints, job['arm'])
                meta = dict(metas['normal'], variant=job['label'], seed=job['seed'])
                mean = torch.tensor(meta['mean'], device=env.device)
                std = torch.tensor(meta['std'], device=env.device)
                for test in job['tests']:
                    path = out / (test + '.json')
                    if path.exists():
                        continue
                    tick = time.monotonic()
                    client.seed = job['evaluation_seed'] + 300
                    if test in ['fixed', 'random', 'full_loop']:
                        client.configure()
                        level = 0 if test == 'fixed' else 2
                        n = job['fixed_n'] if test == 'fixed' else job['random_n']
                        eseed = job['full_loop_seed'] if test == 'full_loop' else job['evaluation_seed']
                        client.seed = eseed + 300
                        result = ordinary(env, client, mean, std, meta, level, n, path, eseed)
                    else:
                        cohort = ROOT / job['cohorts'] / (test + '.h5')
                        env.set_randomization(2)
                        client.configure(cohort)
                        result = pressure(env, client, mean, std, meta, cohort, path)
                        result['cohort_sha256'] = sha256(cohort)
                    # Only episodes counted by original test are summarized.
                    rows = client.telemetry
                    if test in ['fixed', 'random', 'full_loop']:
                        quotas = [n // 32 + int(i < n % 32) for i in range(32)]
                        perclone = [0] * 32
                        chosen = []
                        for row in rows:
                            i = row['clone']
                            if perclone[i] < quotas[i]:
                                chosen.append(row)
                                perclone[i] += 1
                        rows = chosen
                    if len(rows) != result['episodes']:
                        raise RuntimeError(f'Terminal telemetry mismatch {test}: {len(rows)}/{result["episodes"]}')
                    save_json(path.with_suffix('.telemetry.json'), rows)
                    save_json(path.with_suffix('.inference.json'), client.requests)
                    result.update(elapsed_s=time.monotonic() - tick, protocol='phase15_v1',
                        checkpoints=checkpoints, checkpoint_sha256={k: sha256(v) for k, v in checkpoints.items()},
                        selection_split=job['split'], controller_unchanged=True, episode_budget_ticks=600,
                        full_loop_no_injections=test == 'full_loop', rl_updates=0,
                        inference=client.stats(), recovery_stages=summarize(rows))
                    save_json(path, result)
                    print('COMPLETE ' + json.dumps(dict(label=job['label'], seed=job['seed'], test=test,
                        success=result['success'], elapsed_s=result['elapsed_s'],
                        recovery=result['recovery_stages'])), flush=True)
                save_json(out / 'batch1_latency.json', client.benchmark())
    finally:
        client.stop()
        env.step = original_step
        env.close()
    save_json(a.jobs + f'.slot{a.slot}.complete.json', dict(complete=True, rl_updates=0))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        text = traceback.format_exc()
        print(text, flush=True)
        save_json(a.jobs + f'.slot{a.slot}.failed.json', dict(error=text))
        os._exit(1)
    print('EVALUATOR COMPLETE', flush=True)
    os._exit(0)
