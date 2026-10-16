"""Reuse the original evaluation/action/reset protocol in a new namespace."""
import argparse
import os
import traceback
import time
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--jobs', required=True)
p.add_argument('--slot', type=int, default=0)
p.add_argument('--slots', type=int, default=1)
p.add_argument('--socket', required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(headless=True).app

import json
import torch
import isaaclab_tasks
from randomized_env.door_randomization import create
from door_env.isaac_env import transition_after_step
from evaluation.recovery_policy_eval import pressure as original_pressure
from phase15_1.evaluation.pressure_eval import pressure
from experiments.phase14_3_pi05_bc.ordinary_eval import ordinary
from phase15.evaluation.recovery_stage_eval import summarize
from phase15_1.common import ROOT, RESULTS, save_json, sha256, isolated
from phase15_1.severe_reacquire.failure_analysis import DiagnosticPolicy, write_analysis


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    env = create(32, a.device or 'cuda:0', 143501, level=2)
    jobs = json.loads(open(a.jobs).read())
    client = DiagnosticPolicy(a.socket, env)
    original_step = env.step
    ticks=0;started=time.monotonic()
    def observed(action):
        nonlocal ticks
        result = original_step(action)
        robot, gt, done = transition_after_step(env, result[2], result[3])
        client.observe(robot, gt, done)
        ticks+=1
        if ticks%100==0:
            status=dict(ticks=ticks,terminal_episodes=len(client.telemetry),requests=len(client.requests),
                elapsed_s=time.monotonic()-started,current_test=test,output=str(out),rl_updates=0)
            save_json(out/'progress.json',status)
            print('PROGRESS '+json.dumps(status),flush=True)
        return result
    env.step = observed
    try:
        for job in jobs[a.slot::a.slots]:
            if not job.get('diagnostic'):
                from phase15_1.policy_client import SystemPolicy
                # Keep the persistent authenticated connection, change client configuration.
                if not isinstance(client, SystemPolicy):
                    old_conn=client.conn
                    client.__class__=SystemPolicy
                    client.conn=old_conn
                client.variant=job.get('controller','A')
                client.schedule=job.get('schedule',{'normal':5,'recovery':5})
                client.horizons=job.get('horizons',{'normal':10,'recovery':10})
                client.force_planner=job.get('force_planner',False)
                client.capture_trace=job.get('capture_trace',False)
                client.planner=None
                client.configure()
            out = isolated(ROOT / job['output']);out.mkdir(parents=True, exist_ok=True)
            meta = client.load(job['checkpoints'], job['arm'])['normal']
            meta = dict(meta, variant=job['label'], seed=job['seed'])
            mean, std = [torch.tensor(meta[k], device=env.device) for k in ['mean','std']]
            for test in job['tests']:
                path = out / (test+'.json')
                if path.exists():
                    continue
                client.seed = job['evaluation_seed'] + 300
                if test in ['fixed','random','full_loop']:
                    client.configure()
                    n = job['fixed_n'] if test=='fixed' else job['random_n']
                    seed = job['full_loop_seed'] if test=='full_loop' else job['evaluation_seed']
                    client.seed = seed + 300
                    result = ordinary(env, client, mean, std, meta, 0 if test=='fixed' else 2, n, path, seed)
                else:
                    cohort = ROOT / job['cohorts'] / (test+'.h5')
                    env.set_randomization(2);client.configure(cohort)
                    result = original_pressure(env,client,mean,std,meta,cohort,path) if job.get('diagnostic') else pressure(
                        env,client,mean,std,meta,cohort,path,restore_clock=job.get('pressure_clock_restore',True))
                    result['cohort_sha256'] = sha256(cohort)
                rows = client.telemetry
                if test in ['fixed','random','full_loop']:
                    quota = [n//32 + int(i<n%32) for i in range(32)]
                    used = [0]*32;chosen=[]
                    for row in rows:
                        i=row['clone']
                        if used[i]<quota[i]:chosen.append(row);used[i]+=1
                    rows=chosen
                if len(rows)!=result['episodes']:
                    raise RuntimeError('Terminal telemetry coverage mismatch')
                save_json(path.with_suffix('.telemetry.json'), rows)
                save_json(path.with_suffix('.trace.json'), client.trace)
                save_json(path.with_suffix('.inference.json'), client.requests)
                result.update(phase15_1=True, diagnostic_only=job.get('diagnostic', False),
                    checkpoints=job['checkpoints'], checkpoint_sha256={k:sha256(v) for k,v in job['checkpoints'].items()},
                    recovery_stages=summarize(rows), inference=client.stats(),
                    controller_unchanged=job.get('controller','A')=='A', episode_budget_ticks=600, rl_updates=0)
                save_json(path,result)
                print('COMPLETE '+json.dumps(dict(test=test,success=result['success'],episodes=result['episodes'])),flush=True)
            if job.get('diagnostic'):
                write_analysis(out)
            save_json(out/'batch1_latency.json',client.benchmark())
    finally:
        client.stop();env.step=original_step;env.close()
    save_json(a.jobs+f'.slot{a.slot}.complete.json',dict(complete=True,rl_updates=0))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        message=traceback.format_exc();print(message,flush=True)
        save_json(a.jobs+f'.slot{a.slot}.failed.json',dict(error=message));os._exit(1)
    os._exit(0)
