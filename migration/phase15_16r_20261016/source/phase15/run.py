"""Resumable staged supervisor; formal fits are blocked by the real pilot gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from phase15.common import ROOT, RESULTS, CHECKPOINTS, LOGS, save_json, historical, sha256

TESTS = ['fixed', 'random', 'ee_offset', 'contact_loss', 'door_regression', 'stagnation', 'natural_severe', 'full_loop']
ARTIFICIAL = TESTS[2:6]
ISAAC = str(ROOT / '.venv/bin/python')
PI = str(ROOT / '.venv_pi05/bin/python')


class Runner:
    def __init__(self):
        RESULTS.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / 'lifecycle.json'
        self.jobs = json.loads(path.read_text()) if path.exists() else {}
        self.procs = {}
        self.stage = 'pilot_gate'

    def record(self):
        save_json(RESULTS / 'lifecycle.json', self.jobs)
        pilot = []
        registry = RESULTS / 'pilot_processes.json'
        if registry.exists():
            for entry in json.loads(registry.read_text()).get('processes', []):
                cmd = Path(f'/proc/{entry["pid"]}/cmdline')
                actual = cmd.read_bytes().decode().split('\0') if cmd.exists() else []
                verified = all(token in actual for token in entry['identity_tokens'])
                pilot.append(dict(**entry, running=bool(actual) and verified,
                                  command_identity_verified=verified))
        save_json(RESULTS / 'progress.json', dict(stage=self.stage, updated=time.time(),
            running=[dict(name=n, **v) for n, v in self.jobs.items() if v['state'] == 'running'],
            pilot_processes=pilot, supervisor=dict(pid=os.getpid(), command=sys.argv),
            complete=sum(v['state'] == 'complete' for v in self.jobs.values()),
            failed=sum(v['state'] == 'failed' for v in self.jobs.values()), rl_updates=0))

    def launch(self, name, command, gpu):
        if name in self.jobs and self.jobs[name]['state'] == 'running':
            pid = self.jobs[name]['pid']
            if Path(f'/proc/{pid}/cmdline').exists():
                raise RuntimeError('Previous running job: do not duplicate ' + name)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        log = LOGS / (name + '.log')
        with log.open('a') as f:
            p = subprocess.Popen(['nice', '-n', '10', 'bash', '-e', '-c',
                'source configs/runtime_env.sh\nexec "$@"', 'phase15-worker', *command],
                cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        self.procs[name] = p
        self.jobs[name] = dict(state='running', pid=p.pid, gpu=gpu, started=time.time(), command=command, log=str(log))
        self.record()

    def poll(self, name):
        code = self.procs[name].poll()
        if code is None:
            return False
        if self.jobs[name]['state'] == 'running':
            self.jobs[name].update(state='complete' if code == 0 else 'failed', code=code, finished=time.time())
            self.record()
        if code:
            raise RuntimeError(f'{name} failed ({code}); {self.jobs[name]["log"]}')
        return True

    def evaluate(self, jobs, label):
        path = RESULTS / (label + '_jobs.json')
        save_json(path, jobs)
        slots = min(8, len(jobs))
        for gpu in range(slots):
            socket = RESULTS / f'{label}_rpc{gpu}_{int(time.time())}.sock'
            self.launch(f'{label}_server{gpu}', [PI, '-u', '-m', 'phase15.inference_server', '--socket', str(socket)], gpu)
            self.launch(f'{label}_eval{gpu}', [ISAAC, '-u', '-m', 'phase15.evaluation.full_loop_eval',
                '--jobs', str(path), '--slot', str(gpu), '--slots', str(slots), '--socket', str(socket), '--device', 'cuda:0'], gpu)
        names = [f'{label}_eval{i}' for i in range(slots)]
        while True:
            states = [self.poll(n) for n in names]
            if all(states):
                break
            time.sleep(3)
            self.record()
        for i in range(slots):
            name = f'{label}_server{i}'
            deadline = time.monotonic() + 60
            while not self.poll(name):
                if time.monotonic() > deadline:
                    raise RuntimeError('Own inference server did not exit: ' + name)
                time.sleep(1)
        for j in jobs:
            for test in j['tests']:
                if not (ROOT / j['output'] / (test + '.json')).exists():
                    raise RuntimeError('Missing evaluation: ' + str(j))

    def gate(self, pilot_jobs):
        print('WAITING FOR REAL ONE-SEED PILOT GATE', flush=True)
        complete = Path(str(pilot_jobs) + '.slot0.complete.json')
        failed = Path(str(pilot_jobs) + '.slot0.failed.json')
        while not complete.exists():
            if failed.exists():
                raise RuntimeError(failed.read_text())
            time.sleep(3)
            self.record()
        job = json.loads(pilot_jobs.read_text())[0]
        out = ROOT / job['output']
        metrics = {t: json.loads((out / (t + '.json')).read_text()) for t in TESTS}
        rows = [r for t in TESTS for r in json.loads((out / (t + '.telemetry.json')).read_text())]
        plans = [p for r in rows for p in r['planner']]
        model = json.loads((RESULTS / 'pilot/model_gate.json').read_text())
        audit = json.loads((RESULTS / 'environment_audit.json').read_text())
        checks = dict(model_restore=bool(model['checkpoint_restore']), relative_geometry_used=model['relative_geometry_action_difference'] > 0,
            inputs_byte_verified=bool(audit['original_scientific_sha256_all_match']), real_physics_all_eight_tests=len(metrics) == 8,
            pressure_handoff_cache_cleared=all(
                metrics[t]['inference']['handoff_replans'] == metrics[t]['episodes']
                and len(metrics[t]['inference'].get('handoff_records', [])) == metrics[t]['episodes']
                and len({r['cohort_id'] for r in metrics[t]['inference']['handoff_records']}) == metrics[t]['episodes']
                and all(r['cache_cleared'] and r['refreshed'] for r in metrics[t]['inference']['handoff_records'])
                for t in TESTS[2:7]),
            collision_checked_plan=any(p['planning_success'] for p in plans),
            real_pregrasp_and_regrasp=any(p['reached_pregrasp'] and p['regrasp'] and p['handback'] for p in plans),
            successful_task_after_planner_handback=any(p.get('handback') and p.get('final_task_success') for p in plans))
        gate = dict(checks=checks, passed=all(checks.values()), pilot_jobs=str(pilot_jobs),
            pilot_output=str(out), pilot_model_updates=50, rl_updates=0,
            code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'phase15').rglob('*.py')},
            performance_not_formal=True, note='Functional gate, not evidence that targets are met')
        save_json(RESULTS / 'pilot_gate.json', gate)
        if not gate['passed']:
            raise RuntimeError('Pilot gate did not pass: ' + json.dumps(checks))
        print('REAL PILOT GATE PASSED', flush=True)

    def fit(self):
        self.stage = 'recovery_training'
        pending = [(v, s) for v in ['R0', 'R1', 'R2'] for s in range(5)
            if not (CHECKPOINTS / 'recovery_policy/formal' / f'{v}_seed{s}_complete.json').exists()]
        active = {}
        while pending or active:
            for gpu in range(8):
                if gpu in active or not pending:
                    continue
                variant, seed = pending.pop(0)
                name = f'train_{variant}_seed{seed}'
                self.launch(name, [PI, '-u', '-m', 'phase15.recovery_policy.train', '--variant', variant, '--seed', str(seed)], gpu)
                active[gpu] = name
            for gpu, name in list(active.items()):
                if self.poll(name):
                    del active[gpu]
            self.record()
            time.sleep(3)

    def job(self, arm, seed, split, recovery=None, label=None, tests=None):
        validation = split == 'validation'
        source = historical('D' if arm == 'B' else 'C', seed)
        normal = source if arm == 'B' else CHECKPOINTS / 'normal_policy' / source.name
        checkpoints = {'normal': str(normal.relative_to(ROOT))}
        if recovery is not None:
            checkpoints['recovery'] = str(Path(recovery).relative_to(ROOT))
        label = label or arm
        return dict(arm=arm, label=label, seed=seed, split=split, checkpoints=checkpoints,
            output=f'phase15/results/{split}/{label}_seed{seed}', tests=tests or TESTS,
            evaluation_seed=143501 if validation else 143502, full_loop_seed=150501 if validation else 150502,
            fixed_n=32 if validation else 64, random_n=64 if validation else 128,
            cohorts=f'results/phase14_3_pi05_bc/cohorts/{split}')

    def select(self):
        candidates = {}
        for variant in ['R0', 'R1', 'R2']:
            perseed = {}
            scores = []
            for seed in range(5):
                options = []
                for step in [1250, 2500, 5000]:
                    label = f'C_{variant}_step{step}'
                    out = RESULTS / 'validation' / f'{label}_seed{seed}'
                    d = {t: json.loads((out / (t + '.json')).read_text()) for t in TESTS}
                    score = (sum(d[t]['success'] for t in ARTIFICIAL) / 4, d['full_loop']['success'],
                             d['natural_severe']['success'], d['fixed']['success'], -step)
                    options.append((score, step))
                score, step = max(options)
                perseed[str(seed)] = dict(step=step, validation_priority=score)
                scores.append(score[:-1])
            mean = [sum(s[k] for s in scores) / 5 for k in range(4)]
            candidates[variant] = dict(seeds=perseed, validation_priority=mean)
        best = max(candidates, key=lambda v: tuple(candidates[v]['validation_priority']))
        choice = dict(selected_variant=best, candidates=candidates, selected_by='validation_only', rl_updates=0)
        save_json(RESULTS / 'selection.json', choice)
        return choice

    def main(self, pilot):
        self.gate(pilot)
        self.fit()
        self.stage = 'validation'
        self.record()
        validation = [self.job('C', s, 'validation',
            CHECKPOINTS / 'recovery_policy/formal' / f'{v}_seed{s}_step{step}.pt', f'C_{v}_step{step}')
            for v in ['R0', 'R1', 'R2'] for s in range(5) for step in [1250, 2500, 5000]]
        self.evaluate(validation, 'validation')
        selected = self.select()
        self.stage = 'test'
        self.record()
        test = []
        best = selected['selected_variant']
        for seed in range(5):
            step = selected['candidates'][best]['seeds'][str(seed)]['step']
            weight = CHECKPOINTS / 'recovery_policy/formal' / f'{best}_seed{seed}_step{step}.pt'
            for arm in ['A', 'B', 'C', 'D']:
                test.append(self.job(arm, seed, 'test', weight if arm in ['C', 'D'] else None))
            for variant in ['R0', 'R1', 'R2']:
                if variant == best:
                    continue
                step = selected['candidates'][variant]['seeds'][str(seed)]['step']
                weight = CHECKPOINTS / 'recovery_policy/formal' / f'{variant}_seed{seed}_step{step}.pt'
                test.append(self.job('C', seed, 'test', weight, 'C_' + variant, tests=ARTIFICIAL + ['natural_severe']))
        self.evaluate(test, 'test')
        self.stage = 'analysis'
        self.record()
        subprocess.run([ISAAC, '-m', 'phase15.report'], cwd=ROOT, check=True)
        subprocess.run(['python3', '-m', 'phase15.audit'], cwd=ROOT, check=True)
        self.stage = 'complete'
        self.record()
        save_json(RESULTS / 'experiment_complete.json', dict(recovery_fits=15, validation_candidates=45,
            main_test_systems=20, recovery_ablations=10, rl_updates=0, report='docs/phase15_normal_recovery_report.md'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pilot-jobs', required=True)
    a = p.parse_args()
    runner = Runner()
    try:
        runner.main(Path(a.pilot_jobs))
    except BaseException:
        save_json(RESULTS / 'pipeline_failed.json', dict(stage=runner.stage, error=traceback.format_exc()))
        for name, process in runner.procs.items():
            if process.poll() is None:
                process.terminate()  # only subprocesses created by this supervisor
        runner.stage = 'failed'
        runner.record()
        raise
