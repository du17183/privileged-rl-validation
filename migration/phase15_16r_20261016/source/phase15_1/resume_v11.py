"""Resume evaluation only after the reproduced endpoint regression passes."""
import csv
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path('/home/xiaolong/privileged_rl_validation').resolve()
RESULTS = ROOT / 'phase15_1/results'
VERSION = RESULTS / 'code_versions/v11'
REG = RESULTS / 'regression_v11'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return ''


def audit(path, expected):
    data = read(path.with_suffix('.endpoint_audit.json'))
    rows = list(csv.DictReader(path.with_suffix('.csv').open()))
    assert len(rows) == expected
    assert sum(b['selected'] for b in data['batches']) == expected
    assert all(b['complete'] == b['selected'] and not b['missing'] for b in data['batches'])
    assert all(int(r['charged_prefix_ticks']) + int(r['recovery_ticks']) <= 600 for r in rows)
    assert all(float(r['prefix_eef_error_m']) <= .003 and float(r['prefix_handle_error_m']) <= .003
               and float(r['prefix_q_error_rad']) <= .001 for r in rows)
    return dict(episodes=len(rows), maximum_total_ticks=max(int(r['charged_prefix_ticks']) + int(r['recovery_ticks']) for r in rows),
                csv_sha256=digest(path.with_suffix('.csv')), audit_sha256=digest(path.with_suffix('.endpoint_audit.json')),
                restored_clock_deficits=sum(int(r['charged_prefix_ticks']) != int(r['replayed_clock_before_restore']) for r in rows))


def main():
    VERSION.mkdir(parents=True, exist_ok=True)
    lock = (VERSION / 'resume.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + 1800
    required = [REG / (name + '_jobs.json') for name in ['before', 'after', 'after_full', 'after_sequence']]
    required += [RESULTS / 'pilot_v11' / ('jobs_' + v + '.json') for v in ['B', 'C', 'D']]
    while True:
        failed = [str(j) for j in required if Path(str(j) + '.slot0.failed.json').exists()]
        if failed:
            raise RuntimeError('Repair gate failure: ' + str(failed))
        waiting = [str(j) for j in required if not Path(str(j) + '.slot0.complete.json').exists()]
        live = []
        for lifecycle in [REG / 'lifecycle.json', RESULTS / 'pilot_v11/lifecycle.json']:
            for name, row in read(lifecycle).items():
                cmd = identity(row['pid'])
                if cmd:
                    assert ' -m phase15_1.' in cmd, 'PID identity mismatch'
                    live.append(dict(name=name, pid=row['pid'], command=cmd))
        save(RESULTS / 'resume_v11.json', dict(status='waiting_regression', pid=os.getpid(), waiting=waiting,
                                             live=live, updated=time.time(), rl_updates=0))
        if not waiting and not live:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Regression gate wait exceeded 30 minutes; retain evidence, do not restart blindly')
        time.sleep(5)

    old = read(REG / 'before_sequence/natural_severe.endpoint_audit.json')
    missing = [m for b in old['batches'] for m in b['missing']]
    assert Path(str(REG / 'before_sequence_jobs.json') + '.slot0.failed.json').exists()
    assert missing and all(m['initial']['prefix_auto_resets'] > 0 and
                           m['initial']['replayed_clock_before_restore'] < m['initial']['charged_prefix_ticks'] for m in missing)
    checks = {name: audit(REG / name / 'natural_severe.json', count)
              for name, count in [('after', 32), ('after_full', 160), ('after_sequence', 160)]}
    assert checks['after_sequence']['restored_clock_deficits'] >= 1
    matched_before = read(REG / 'before/natural_severe.json')['success']
    matched_after = read(REG / 'after/natural_severe.json')['success']
    assert matched_before == matched_after
    for variant in ['B', 'C', 'D']:
        for case in ['natural_severe', 'ee_offset']:
            audit(RESULTS / 'pilot_v11' / variant / (case + '.json'), 32)
    save(REG / 'gate.json', dict(passed=True, checks=checks, reproduced_original_failure=True,
                               original_missing=missing, matched32_before_success=matched_before,
                               matched32_after_success=matched_after, original_episode_budget=600,
                               performance_threshold_not_a_technical_gate=True, rl_updates=0))

    pressure_cases = ['ee_offset', 'contact_loss', 'door_regression', 'stagnation', 'natural_severe',
                      'grade_05', 'grade_10', 'grade_15', 'grade_20']
    invalidated, reused = [], []
    for out in sorted((RESULTS / 'validation').iterdir()):
        if not out.is_dir():
            continue
        for f in list(out.iterdir()):
            if not f.is_file():
                continue
            if any(f.name.startswith(c + '.') for c in pressure_cases):
                target = VERSION / 'invalidated_pressure' / out.name / f.name
                target.parent.mkdir(parents=True, exist_ok=True)
                assert not target.exists()
                invalidated.append(dict(original=str(f), archived=str(target), sha256=digest(f)))
                shutil.move(f, target)
            elif any(f.name.startswith(c + '.') for c in ['fixed', 'random', 'full_loop']):
                reused.append(dict(path=str(f), sha256=digest(f)))
    jobs = RESULTS / 'jobs/controller_val_prefetch'
    if jobs.exists():
        target = VERSION / 'old_controller_val_prefetch'
        assert not target.exists()
        shutil.move(jobs, target)
    if (RESULTS / 'pipeline_failed.json').exists():
        shutil.move(RESULTS / 'pipeline_failed.json', VERSION / 'pipeline_failed_v10_retained.json')
    save(VERSION / 'result_reuse_manifest.json', dict(invalidated=invalidated, reused=reused,
         reason='Re-evaluate all old pressure cases with explicit original episode age; fresh-reset cases retain SHA', rl_updates=0))

    plan = read(RESULTS / 'collection_plan_v9.json')
    plan['revision'] = 'v11'
    row = next(r for r in plan['tasks'] if r['split'] == 'test' and r['case'] == 'grade_20')
    source = Path(row['output']) / 'grade_20.h5'
    import h5py
    with h5py.File(source, 'r') as h:
        assert len(h) == 78
        assert all(all(k in g for k in ['prefix_actions', 'reset_parameters', 'handoff_robot', 'handoff_state']) for g in h.values())
    row['resume_from'] = row['output']
    row['output'] = str(RESULTS / 'cohorts/source_v11/test/grade_20_seed0')
    row['reset_seed'] = 55170000
    plan['v11_provenance'] = dict(original_partial_grade20_sha256=digest(source), preserved_grade20=78,
                                additional_required=22, preserved_validation_manifest_sha256=digest(RESULTS / 'cohorts/validation/manifest.json'))
    save(RESULTS / 'collection_plan_v11.json', plan)
    for f in reused:
        assert digest(f['path']) == f['sha256']

    finish_resume()


def finish_resume():
    """Resume the deployment step without repeating completed data archival."""
    assert read(REG / 'gate.json')['passed']
    manifest = read(VERSION / 'result_reuse_manifest.json')
    assert all(digest(f['path']) == f['sha256'] for f in manifest['reused'])
    assert all(digest(f['archived']) == f['sha256'] for f in manifest['invalidated'])
    plan = read(RESULTS / 'collection_plan_v11.json')
    assert digest(RESULTS / 'cohorts/validation/manifest.json') == plan['v11_provenance']['preserved_validation_manifest_sha256']
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            assert ' -m phase15_1.run ' not in identity(proc.name), 'Supervisor is already active'
    candidate = VERSION / 'run_candidate.py'
    if not candidate.exists():
        shutil.copy2(Path('/home/xiaolong/phase15_1_run_v11.py'), candidate)
    assert digest(candidate) == '3e477417dcd72c95d0bbe84899c32713001264d5a28436a739cdd251fe1f7a39'
    shutil.copy2(candidate, ROOT / 'phase15_1/run.py')
    code = {str(p): digest(p) for p in [ROOT / 'phase15_1/run.py', ROOT / 'phase15_1/evaluation/pressure_eval.py',
                                       ROOT / 'phase15_1/evaluation/full_system_eval.py', Path(__file__)]}
    save(VERSION / 'deployed_code.json', dict(files=code, gate_sha256=digest(REG / 'gate.json'), rl_updates=0))
    note = ('\n\n## v11 endpoint/clock repair\n'
            'The v10 A_seed4 sequential evaluator failed with 31/32 pressure endpoints. Matched sequential replay reproduced it: '
            'clone3 reset once during its physical prefix; handoff age39 instead of the charged144 ticks. '
            'The Phase15.1-only wrapper restores prefix+one neutral tick age, retaining the original600 tick budget. '
            'A fresh32 matched comparison is unchanged;160 fresh and160 sequential corrected starts pass endpoint, geometry and budget audits. '
            'All old pressure validation artifacts are archived and re-evaluated; fresh fixed/random/full-loop outputs are SHA-preserved. '
            'The interrupted grade20 source retains78 valid starts and appends22 from an independent reset stream. '
            'Negative controller results remain results, not technical gate failures. No model was trained.\n')
    note_path = RESULTS / 'code_versions/v9/throughput_revision.md'
    if '## v11 endpoint/clock repair' not in note_path.read_text(encoding='utf-8'):
        with note_path.open('a', encoding='utf-8') as f:
            f.write(note)
    args = [str(ROOT / '.venv/bin/python'), '-u', '-m', 'phase15_1.run', '--initial-pilots', 'pilot_v11',
            '--replicas-per-gpu', '4', '--collection-plan', str(RESULTS / 'collection_plan_v11.json')]
    log = ROOT / 'logs/phase15_1/supervisor_v11.log'
    command = 'cd ' + shlex.quote(str(ROOT)) + ' && source configs/runtime_env.sh && ' + shlex.join(args)
    child = subprocess.Popen(['bash', '-lc', command], stdout=log.open('ab', buffering=0), stderr=subprocess.STDOUT,
                             start_new_session=True)
    launch = dict(pid=child.pid, args=args, command=command, log=str(log), gate=str(REG / 'gate.json'), started=time.time(), rl_updates=0)
    save(RESULTS / 'launch_v11.json', launch)
    save(RESULTS / 'resume_v11.json', dict(status='launched', manager_pid=os.getpid(), **launch))


if __name__ == '__main__':
    try:
        import sys
        finish_resume() if '--finish-only' in sys.argv else main()
    except BaseException:
        save(RESULTS / 'resume_v11.json', dict(status='blocked', manager_pid=os.getpid(), error=traceback.format_exc(),
                                            updated=time.time(), rl_updates=0))
        raise
