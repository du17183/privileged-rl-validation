"""Independent post-run audit. No simulator, model update, or model selection."""
import csv
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

ROOT = Path('/home/xiaolong/privileged_rl_validation').resolve()
R = ROOT / 'phase15_1/results'
METHODS = ['A', 'B', 'C', 'D', 'E_final']
MAIN = ['fixed', 'random', 'ee_offset', 'contact_loss', 'door_regression', 'stagnation', 'natural_severe', 'full_loop']
GRADES = ['grade_05', 'grade_10', 'grade_15', 'grade_20']
ART = ['ee_offset', 'contact_loss', 'door_regression', 'stagnation']
checks = Counter()
hashes = {}


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    p = Path(p)
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    value = digest.hexdigest()
    hashes[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] = value
    return value


def check(ok, label):
    if not ok:
        raise AssertionError(label)
    checks[label.split(':')[0]] += 1


def eq(a, b, label):
    check(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-10), label)


def readcsv(p):
    return list(csv.DictReader(Path(p).open(newline='')))


def ci(x):
    x = np.asarray(x, float)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    half = float(stats.t.ppf(.975, 4) * sd / np.sqrt(5))
    return dict(mean=mean, seed_sd=sd, ci95_low=mean-half, ci95_high=mean+half, seeds=5)


def key(row):
    return row.get('cohort_id') or (int(row['clone']), int(row['episode']))


def audit():
    check(load(R/'experiment_complete.json')['complete'], 'complete marker')
    check(not (R/'pipeline_failed.json').exists(), 'no failure marker')
    check(not list((R/'jobs').rglob('*.failed.json')), 'no job failure markers')
    selection = load(R/'selection.json')
    check(sha(R/'selection.json') == 'd4ea52b85c1009a7a0d15a9826f7fb049399427a4349dcc84dfe75f00c50758f', 'locked selection SHA')
    check(selection['selected_by'] == 'validation_only', 'validation-only selection')
    check(selection['numerical_candidate_physical_retention'], 'physical retention gate')
    for k, value in load(R/'frozen_inputs.json')['files'].items():
        p = Path(k) if Path(k).is_absolute() else ROOT/k
        check(sha(p) == value, 'frozen original SHA:'+k)
    check(load(R/'frozen_verification.json')['passed'], 'pipeline frozen check')

    natural_ids = {}
    cohort_ids = {}
    for split in ['validation', 'test']:
        manifest = load(R/f'cohorts/{split}/manifest.json')
        sha(R/f'cohorts/{split}/manifest.json')
        for test, meta in manifest.items():
            p = R/f'cohorts/{split}/{test}.h5'
            check(sha(p) == meta['sha256'], 'cohort SHA:'+split+'/'+test)
            check(sha(meta['collection_plan']) == meta['collection_plan_sha256'], 'collection plan SHA:'+split+'/'+test)
            for source, value in meta['sources'].items():
                check(sha(source) == value, 'source cohort SHA:'+source)
            with h5py.File(p, 'r') as f:
                check(len(f) == meta['episodes'], 'cohort coverage:'+split+'/'+test)
                check(bool(f.attrs['evaluation_only']) and not bool(f.attrs['training_labels']), 'evaluation-only cohort:'+split+'/'+test)
                cohort_ids[(split, test)] = set(f.keys())
                if test == 'natural_severe':
                    by_seed = Counter(int(g.attrs['reference_model_seed']) for g in f.values())
                    check(by_seed == {s:32 for s in range(5)}, 'natural five-source balance:'+split)
                    check(all(bool(g.attrs['natural_source_episode_failed']) for g in f.values()), 'natural failed source labels:'+split)
                    natural_ids[split] = {(int(g.attrs['reference_model_seed']), int(g.attrs['reset_seed']), int(g.attrs['clone']), int(g.attrs['source_episode'])) for g in f.values()}
                    check(len(natural_ids[split]) == 160, 'unique natural source episodes:'+split)
                if test in GRADES and split == 'test':
                    check(len(f) >= 100, 'graded minimum coverage:'+test)
    check(not (natural_ids['test'] & natural_ids['validation']), 'natural split disjoint identities')

    raw = {}
    stages = {}
    per = readcsv(R/'per_seed_metrics.csv')
    stage_csv = readcsv(R/'stage_metrics.csv')
    check(len(per) == 300 and len(stage_csv) == 300, 'formal 300 test rows')
    per_index = {(x['method'],int(x['seed']),x['test']):x for x in per}
    stage_index = {(x['method'],int(x['seed']),x['test']):x for x in stage_csv}
    check(len(per_index) == 300 and len(stage_index) == 300, 'unique formal test rows')
    failure_classes = Counter()
    plan_failure_counts = Counter()
    controller_timing = {}
    max_budget = 0
    matched_ids = {}
    for method in METHODS:
        for seed in range(5):
            for test in MAIN+GRADES:
                folder = R/f'test/{method}_seed{seed}'
                if test in GRADES:
                    folder = Path(str(folder)+'_grades')
                p = folder/(test+'.json')
                d = load(p)
                rows = readcsv(p.with_suffix('.csv'))
                tele = load(p.with_suffix('.telemetry.json'))
                expected = 64 if test == 'fixed' else 128 if test in ['random','full_loop'] else 160 if test == 'natural_severe' else 100
                check(len(rows) == len(tele) == d['episodes'] == expected, 'raw coverage:'+str(p))
                check(d['rl_updates'] == 0, 'no RL updates:'+str(p))
                check(len({key(x) for x in rows}) == expected, 'unique raw endpoints:'+str(p))
                tele_by_id = {key(x):x for x in tele}
                check(set(tele_by_id) == {key(x) for x in rows}, 'telemetry endpoint identities:'+str(p))
                for row in rows:
                    t = tele_by_id[key(row)]
                    eq(t['success'], row['success'], 'raw telemetry success')
                    eq(t['final_angle_rad'], row['final_angle_rad'], 'raw telemetry final angle')
                    if 'original_episode_ticks' in t:
                        check(int(t['original_episode_ticks']) <= 600, 'telemetry episode budget')
                        max_budget = max(max_budget,int(t['original_episode_ticks']))
                for metric in ['success','max_angle_rad','final_angle_rad']:
                    eq(d[metric], np.mean([float(x[metric]) for x in rows]), 'summary raw mean:'+metric)
                    eq(per_index[(method,seed,test)][metric],d[metric], 'per-seed metric:'+metric)
                eq(per_index[(method,seed,test)]['interactions'],d['interactions'], 'interaction counter')
                for module, ck in d['checkpoints'].items():
                    relative = str(Path(ck).relative_to(ROOT))
                    eq_sha = hashes.get(relative)
                    check(eq_sha is not None and eq_sha == d['checkpoint_sha256'][module], 'formal frozen checkpoint SHA')
                if test not in ['fixed','random','full_loop']:
                    check(d['cohort_sha256'] == hashes[str((R/f'cohorts/test/{test}.h5').relative_to(ROOT))], 'formal input cohort SHA')
                    check({key(x) for x in rows} == cohort_ids[('test',test)], 'formal uses full matched cohort')
                    endpoint = load(p.with_suffix('.endpoint_audit.json'))
                    check(endpoint['restore_clock'] and endpoint['episode_budget'] == 600, 'pressure restored clock')
                    check(all(b['selected'] == b['complete'] and not b['missing'] for b in endpoint['batches']), 'complete pressure batches')
                    check(sum(b['selected'] for b in endpoint['batches']) == expected, 'pressure batch totals')
                    for row in rows:
                        budget = int(row['charged_prefix_ticks']) + int(row['recovery_ticks'])
                        check(budget <= 600, 'charged prefix recovery budget')
                        check(float(row['prefix_eef_error_m']) <= .003 and float(row['prefix_handle_error_m']) <= .003 and float(row['prefix_q_error_rad']) <= .001, 'matched physical handoff geometry')
                        max_budget = max(max_budget,budget)
                    sha(p.with_suffix('.endpoint_audit.json'))
                else:
                    identity = {(int(x['clone']),int(x['episode'])):(float(x['angle0_rad']),float(x['offset_x']),float(x['offset_y']),float(x['offset_z'])) for x in rows}
                    base_key = (seed,test)
                    if base_key in matched_ids:
                        check(identity == matched_ids[base_key], 'matched fresh test initial parameters')
                    else:
                        matched_ids[base_key] = identity
                raw[(method,seed,test)] = d['success']
                plans = [x for t in tele for x in t['planner']]
                stage = dict(episodes=expected,planner_attempts=len(plans),episodes_with_planner=sum(bool(t['planner']) for t in tele),safe_retreat_success=sum(bool(x.get('safe_retreat')) for x in plans),pregrasp_success=sum(bool(x.get('reached_pregrasp')) for x in plans),regrasp_success=sum(bool(x.get('regrasp')) for x in plans),handbacks=sum(bool(x.get('handback')) for x in plans),final_success_after_handback=sum(bool(x.get('handback')) and bool(x.get('final_task_success')) for x in plans),final_success_episodes=sum(bool(t['success']) for t in tele),progress_resumed_episodes=sum(any(e['flags'][3] for e in t['events']) for t in tele))
                stages[(method,seed,test)] = stage
                for k,v in stage.items():
                    eq(stage_index[(method,seed,test)][k],v,'stage CSV recomputation:'+k)
                if method == 'E_final':
                    controller_timing.setdefault(test,[]).extend(tele)
                if method == 'E_final' and test == 'natural_severe':
                    plan_failure_counts.update(x['failure_stage'] for x in plans if x.get('failure_stage'))
                    for t in tele:
                        if t['success']:
                            continue
                        last = t['planner'][-1] if t['planner'] else None
                        if last is None:
                            category = 'no_reacquire_attempt'
                        elif last.get('handback'):
                            category = 'handback_then_task_failure'
                        else:
                            category = last.get('failure_stage') or last.get('failure') or 'unfinished_reacquire'
                        failure_classes[category] += 1
                for suffix in ['.json','.csv','.telemetry.json','.inference.json']:
                    sha(p.with_suffix(suffix))

    for method in METHODS:
        for seed in range(5):
            raw[(method,seed,'artificial_macro')] = float(np.mean([raw[(method,seed,test)] for test in ART]))
    metrics = readcsv(R/'metrics.csv')
    check(len(metrics) == 65, '65 aggregate metric rows')
    for row in metrics:
        values = [raw[(row['method'],s,row['test'])] for s in range(5)]
        for k,v in ci(values).items():
            eq(row[k],v,'aggregate statistics recomputation:'+k)
    paired = readcsv(R/'paired_tests.csv')
    check(len(paired) == 25, '25 paired test rows')
    tps = []
    for row in paired:
        a,b = row['comparison'].split('-')
        d = np.array([raw[(a,s,row['test'])]-raw[(b,s,row['test'])] for s in range(5)])
        for k,v in ci(d).items():
            eq(row[k],v,'paired effect CI recomputation:'+k)
        exact = sum(abs(float(np.mean(d*np.array(sign)))) >= abs(float(np.mean(d)))-1e-12 for sign in itertools.product([-1,1],repeat=5))/32
        eq(row['exact_signflip_p'],exact,'paired exact signflip recomputation')
        check(exact >= .0625,'five seed exact p lower bound')
        tp = float(stats.ttest_rel([raw[(a,s,row['test'])] for s in range(5)],[raw[(b,s,row['test'])] for s in range(5)]).pvalue) if np.std(d)>1e-12 else (1. if abs(np.mean(d))<1e-12 else 0.)
        eq(row['paired_t_p'],tp,'paired t recomputation')
        tps.append(tp)
    adjusted = np.maximum.accumulate(np.array(sorted(tps))*np.arange(25,0,-1)).clip(0,1)
    for rank,index in enumerate(np.argsort(tps)):
        eq(paired[index]['paired_t_holm_p'],adjusted[rank],'Holm 25-endpoint family recomputation')

    timing = readcsv(R/'timing.csv')
    check(len(timing) == 10, 'ten deployment timing rows')
    latencies = []
    lifecycle = load(R/'lifecycle.json')['processes']
    physics = [x for x in lifecycle.values() if 'phase15_1.evaluation.full_system_eval' in x.get('command',[])]
    check(all(x.get('exit_code') == 0 for x in physics), 'all physics workers clean exit')
    physics_end = max(Path(x['completion_evidence']).stat().st_mtime for x in physics)
    for row in timing:
        seed = int(row['seed'])
        d = load(R/f'deployment_latency/seed{seed}.json')[row['module']]
        check(d['batch'] == 1 and d['concurrent_robot_workers_on_gpu'] == 1 and not d['simulation_running_on_gpu'], 'exclusive GPU batch1 latency protocol')
        check(not d['real_robot_hardware_test'] and d['rl_updates'] == 0,'latency scientific scope')
        check(d['engineering']['backend'] == selection['backend'], 'latency selected graph backend')
        for k in ['mean_model_s','mean_rpc_s']:
            sample_key = 'model_s' if k == 'mean_model_s' else 'rpc_s'
            eq(d[k],np.mean([x[sample_key] for x in d['samples']]),'latency raw mean:'+k)
            eq(row[k],d[k],'timing CSV raw mean:'+k)
        eq(d['p95_rpc_s'],np.quantile([x['rpc_s'] for x in d['samples']],.95),'latency P95')
        eq(row['budget_s'],d['execute_ticks']/60,'nominal latency budget')
        latencies.append(dict(seed=seed,module=row['module'],rpc_mean_s=d['mean_rpc_s'],rpc_p95_s=d['p95_rpc_s'],sample_deadline_misses=sum(x['rpc_s']>d['replan_budget_s'] for x in d['samples'])))
        sha(R/f'deployment_latency/seed{seed}.json')
    for name,meta in lifecycle.items():
        if name.startswith('deployment_latency_'):
            check(float(meta['started']) >= physics_end, 'deployment latency after all physics')
            check(meta.get('exit_code') == 0,'deployment worker clean exit')
    own_live = []
    for name,meta in lifecycle.items():
        p = Path('/proc')/str(meta['pid'])/'cmdline'
        if p.exists():
            cmd = p.read_bytes().replace(b'\x00',b' ').decode(errors='replace').strip()
            if cmd and all(x in cmd for x in meta['command']):
                own_live.append(name)
    check(not own_live,'no remaining owned evaluation processes')

    for filename,value in load(R/'delivery_verification.json')['formal_input_and_output_sha256'].items():
        check(hashes[filename] == value,'pipeline output SHA:'+filename)
    for p in [R/'selection.json',R/'summary.json',R/'per_seed_metrics.csv',R/'metrics.csv',R/'stage_metrics.csv',R/'paired_tests.csv',R/'timing.csv',R/'frozen_inputs.json',R/'frozen_verification.json',ROOT/'docs/phase15_1_reacquire_latency_report.md',ROOT/'docs/phase15_1_severe_failure_analysis.md']:
        sha(p)
    check(sha(ROOT/'docs/phase15_1_reacquire_latency_report.md') == load(R/'delivery_verification.json')['report_sha256'],'generated report SHA')
    aggregate_stages = {}
    for method in METHODS:
        for test in MAIN+GRADES:
            selected_stages = [stages[(method,s,test)] for s in range(5)]
            aggregate_stages[method+'/'+test] = {k:sum(x[k] for x in selected_stages) for k in selected_stages[0]}
    control_costs = {}
    for test,tele in controller_timing.items():
        plan_values = [p.get('plan_s',0) for t in tele for p in t['planner']]
        control_costs[test] = dict(episodes=len(tele),planner_attempts=len(plan_values),planner_start_mean_s=float(np.mean(plan_values)) if plan_values else None,planner_start_p95_s=float(np.quantile(plan_values,.95)) if plan_values else None,episode_planner_cost_mean_s=float(np.mean([t['costs_s']['planner'] for t in tele])),episode_normal_cost_mean_s=float(np.mean([t['costs_s']['normal'] for t in tele])),episode_recovery_cost_mean_s=float(np.mean([t['costs_s']['recovery'] for t in tele])))
    output = dict(passed=True,scope='independent raw-result/statistics/provenance audit; no additional physics or training',time_utc_epoch=time.time(),checks=dict(checks),formal_tests=300,frozen_files=40,max_charged_episode_ticks=max_budget,aggregate_stages=aggregate_stages,natural_terminal_failure_classes=dict(failure_classes),natural_plan_attempt_failure_stages=dict(plan_failure_counts),deployment_latency=latencies,control_costs=control_costs,sha256=hashes,rl_updates=0)
    (R/'final_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ['sha256','aggregate_stages','control_costs']},ensure_ascii=True))


if __name__ == '__main__':
    audit()
