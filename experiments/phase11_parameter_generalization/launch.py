"""Own-process queue; fixed per-run update ratio, no historical overwrites."""
import csv
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, LOG, CKPT, ARMS, prepared
def now(): return datetime.now(timezone.utc).isoformat()


def main():
    lock = (OUT/'queue.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not json.loads((OUT/'preflight.json').read_text())['passed'] or prepared()['curriculum_enabled'] is None:
        raise RuntimeError('Preflight and pilot decision required')
    pending = [(arm, seed) for seed in range(5) for arm in ARMS]
    running = {};completed = [];cap = 3
    (OUT/'queue_config.json').write_text(json.dumps(dict(total_jobs=20, max_jobs_per_gpu=cap,
        training_envs=32, updates_per_vector_step=4, batch_size=256, paired_evaluator_envs=32,
        total_training_steps=6000000, external_processes='untouched', started_at=now()), indent=2))
    with (OUT/'launch_status.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('arm', 'seed', 'gpu', 'pid', 'start', 'end', 'exit_code'))
        writer.writeheader()
        while pending or running:
            for pid, (proc, record, log) in list(running.items()):
                code = proc.poll()
                if code is None: continue
                log.close()
                marker = CKPT/f"P11{record['arm']}_seed{record['seed']}"/'completed.json'
                if code == 0 and (not marker.exists() or json.loads(marker.read_text())['steps'] != 300000): code = 1
                record.update(end=now(), exit_code=code);writer.writerow(record);stream.flush()
                completed.append(record);del running[pid];print('finished', record, flush=True)
                if code: raise RuntimeError('Own run failed; no other job stopped or overwritten')
            readings = [tuple(int(v.strip()) for v in line.split(',')) for line in subprocess.check_output([
                'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()]
            counts = {g: sum(r[1]['gpu'] == g for r in running.values()) for g, _, _ in readings}
            while pending:
                available = [r for r in readings if counts[r[0]] < cap and r[1] < 80000]
                if not available: break
                gpu, _, _ = min(available, key=lambda r: (counts[r[0]], r[2], r[1], r[0]))
                arm, seed = pending.pop(0);name = f'P11{arm}_seed{seed}'
                if (CKPT/name).exists(): raise RuntimeError('Existing run cannot be overwritten')
                log = (LOG/f'{name}.log').open('x')
                proc = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-u', 'experiments/phase11_parameter_generalization/train.py',
                    '--arm', arm, '--seed', str(seed), '--device', 'cuda:0'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                    stdout=log, stderr=subprocess.STDOUT)
                record = dict(arm=arm, seed=seed, gpu=gpu, pid=proc.pid, start=now())
                running[proc.pid] = (proc, record, log);counts[gpu] += 1;print('started', record, flush=True)
            state = dict(timestamp=now(), running=[v[1] for v in running.values()], pending=pending, completed=completed)
            temp = OUT/'queue_state.tmp';temp.write_text(json.dumps(state, indent=2));os.replace(temp, OUT/'queue_state.json')
            time.sleep(5)
    (OUT/'training_completed.json').write_text(json.dumps(dict(runs=20, steps=6000000, completed_at=now()), indent=2))
if __name__ == '__main__': main()
