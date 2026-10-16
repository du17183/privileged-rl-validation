"""Verify final deliverables against episode CSVs and original job manifests."""
import csv
import itertools
import json
import math
from collections import Counter
import numpy as np
from scipy import stats
from phase15.common import ROOT, RESULTS, CHECKPOINTS, save_json, sha256


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def close(a, b):
    return bool(np.allclose(a, b, atol=1e-11, rtol=1e-9))


def main():
    weights, cohorts = {}, {}
    result_count = episodes = 0
    for split, systems in [('validation', 45), ('test', 30)]:
        jobs = json.loads((RESULTS / (split + '_jobs.json')).read_text())
        require(len(jobs) == systems, 'Job manifest count ' + split)
        for job in jobs:
            for test in job['tests']:
                out = ROOT / job['output']
                d = json.loads((out / (test + '.json')).read_text())
                telemetry = json.loads((out / (test + '.telemetry.json')).read_text())
                with (out / (test + '.csv')).open() as f:
                    rows = list(csv.DictReader(f))
                expected = job['fixed_n'] if test == 'fixed' else job['random_n'] if test in ['random', 'full_loop'] else 32 if split == 'validation' else 64
                require(d['episodes'] == len(rows) == len(telemetry) == expected, 'Episode count ' + str(out / test))
                success = [int(row['success']) for row in rows]
                require(close(np.mean(success), d['success']), 'CSV success mismatch ' + str(out / test))
                require(close(np.mean([row['success'] for row in telemetry]), d['success']), 'Telemetry success mismatch')
                require(all(0 < row['original_episode_ticks'] <= 600 for row in telemetry), 'Original episode budget')
                require(d['rl_updates'] == 0 and d['controller_unchanged'] and d['episode_budget_ticks'] == 600, 'Experiment boundary')
                require(d['selection_split'] == split, 'Split mismatch')
                if test == 'full_loop':
                    require(d['full_loop_no_injections'], 'Fresh full-loop protocol')
                if test not in ['fixed', 'random', 'full_loop']:
                    records = d['inference']['handoff_records']
                    require(d['inference']['handoff_replans'] == len(records) == expected, 'Handoff count')
                    require(len({row['cohort_id'] for row in rows}) == expected, 'Duplicate cohort CSV')
                    require({row['cohort_id'] for row in rows} == {row['cohort_id'] for row in records}, 'Handoff IDs')
                    require(all(row['cache_cleared'] and row['refreshed'] for row in records), 'Stale handoff action cache')
                    path = ROOT / job['cohorts'] / (test + '.h5')
                    if str(path) not in cohorts:
                        cohorts[str(path)] = sha256(path)
                    require(cohorts[str(path)] == d['cohort_sha256'], 'Historical cohort SHA')
                for module, path in d['checkpoints'].items():
                    if path not in weights:
                        weights[path] = sha256(path)
                    require(weights[path] == d['checkpoint_sha256'][module], 'Evaluated weight SHA')
                result_count += 1
                episodes += expected
    require(result_count == 570 and episodes == 30400, 'Formal coverage')
    for variant, ratio in [('R0', 1.), ('R1', .8), ('R2', .5)]:
        for seed in range(5):
            d = json.loads((CHECKPOINTS / 'recovery_policy/formal' / f'{variant}_seed{seed}_complete.json').read_text())
            require(d['updates'] == 5000 and d['batch'] == 128 and not d['pilot'], 'Formal fit budget')
            require(d['recovery_draws'] == round(d['total_draws'] * ratio), 'Training mixture')
    summary = json.loads((RESULTS / 'summary.json').read_text())
    with (RESULTS / 'metrics.csv').open() as f:
        metrics = list(csv.DictReader(f))
    require(len(metrics) == 36, 'Summary CSV coverage')
    for row in metrics:
        arm, metric = row['arm'], row['metric']
        d = summary['metrics'][arm][metric]
        values = np.asarray([float(row[f'seed{i}']) for i in range(5)])
        require(close(values, d['seeds']), 'Seed statistics CSV/JSON')
        mean, sd = values.mean(), values.std(ddof=1)
        margin = stats.t.ppf(.975, 4) * sd / math.sqrt(5)
        require(close([mean, sd, mean - margin, mean + margin],
                      [d['mean'], d['sd'], *d['ci95']]), 'Five-seed 95% CI')
        if metric != 'artificial_macro':
            require(close(values, [json.loads((RESULTS / 'test' / f'{arm}_seed{i}' / (metric + '.json')).read_text())['success'] for i in range(5)]), 'Statistics/episode source')
        else:
            require(close(values, np.mean([summary['metrics'][arm][t]['seeds'] for t in ['ee_offset', 'contact_loss', 'door_regression', 'stagnation']], axis=0)), 'Artificial macro')
    comparisons = summary['paired_comparisons']
    require(len(comparisons) == 12, 'Paired comparison coverage')
    for d in comparisons:
        left, right = d['comparison'].split('-')
        delta = np.asarray(summary['metrics'][left][d['metric']]['seeds']) - np.asarray(summary['metrics'][right][d['metric']]['seeds'])
        t = delta.mean() / (delta.std(ddof=1) / math.sqrt(5))
        p = 2 * stats.t.sf(abs(t), 4)
        exact = np.mean([abs(np.mean(delta * sign)) >= abs(delta.mean()) - 1e-12 for sign in itertools.product([-1, 1], repeat=5)])
        require(close([p, exact, *delta], [d['paired_t_p'], d['exact_two_sided_signflip_p'], *d['delta']['seeds']]), 'Paired test')
    order = np.argsort([d['paired_t_p'] for d in comparisons])
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (12 - rank) * comparisons[index]['paired_t_p']))
        require(close(previous, comparisons[index]['paired_t_holm_p']), 'Holm correction')
    stages = []
    failures = Counter()
    for arm in ['A', 'B', 'C', 'D']:
        for seed in range(5):
            for test in ['fixed', 'random', 'ee_offset', 'contact_loss', 'door_regression', 'stagnation', 'natural_severe', 'full_loop']:
                rows = json.loads((RESULTS / 'test' / f'{arm}_seed{seed}' / (test + '.telemetry.json')).read_text())
                plans = [p for row in rows for p in row['planner']]
                counts = {k: sum(bool(p.get(k)) for p in plans) for k in ['ik_success', 'planning_success', 'start_collision', 'reached_pregrasp', 'regrasp', 'handback']}
                after = sum(bool(p.get('handback') and p.get('final_task_success')) for p in plans)
                stages.append(dict(arm=arm, seed=seed, test=test, episodes=len(rows), planner_events=len(plans), **counts, task_success_after_handback_events=after))
                if arm == 'D' and test == 'natural_severe':
                    failures.update(p.get('failure_stage') or ('handback' if p.get('handback') else 'episode_end_without_handback') for p in plans)
    with (RESULTS / 'stage_metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(stages[0]));w.writeheader();w.writerows(stages)
    output = dict(passed=True, formal_result_files=result_count, formal_episodes=episodes,
        recovery_fits=15, validation_candidates=45, main_test_systems=20, recovery_ablations=10,
        unique_evaluated_weights_sha256_verified=len(weights), unique_historical_cohorts_verified=len(cohorts),
        metrics_rows=36, five_seed_ci_checked=True, paired_tests_checked=12, holm_checked=True,
        stage_rows=len(stages), natural_severe_D_failure_events=dict(failures), rl_updates=0)
    save_json(RESULTS / 'delivery_verification.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
