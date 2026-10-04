"""Read-only matched-condition audit and rejection/quality evidence."""
import csv
import json
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, ARMS


def read_heldout(arm,seed,frozen=False):
    name=f'{arm}_seed{seed}'+('_frozen' if frozen else '')
    return json.loads((OUT/'heldout'/f'{name}.json').read_text())


def plan(records):
    fields=('initial_angle_deg','offset_x_m','offset_y_m','offset_z_m','friction_scale')
    return {(r['env_index'],r['episode_index']):tuple(r[k] for k in fields) for r in records}


def main():
    comparisons=[];failures=[]
    for seed in range(5):
        frozen=read_heldout('A',seed,True)
        reference={(r['metrics']['condition'],r['metrics']['mode']):plan(r['records']) for r in frozen['results']}
        for arm in ARMS:
            model=read_heldout(arm,seed)
            for row in model['results']:
                key=(row['metrics']['condition'],row['metrics']['mode'])
                matched=plan(row['records'])==reference[key]
                comparisons.append(dict(seed=seed,arm=arm,endpoint=row['endpoint'],condition=key[0],mode=key[1],matched=matched))
                if not matched:failures.append(comparisons[-1])
                if row['metrics']['effective_std']>.010001:raise RuntimeError('Execution cap changed')
    summary=json.loads((OUT/'summary.json').read_text());rejections={};quality={}
    for arm in ARMS:
        rows=[]
        for seed in range(5):
            with (OUT/f'updates_P11{arm}_seed{seed}.csv').open() as stream:
                rows.extend(dict(r,seed=seed) for r in csv.DictReader(stream))
        rejected=[r for r in rows if int(r['accepted'])==0]
        nominal_only=[];curriculum_only=[];both=[]
        for r in rejected:
            reasons=r['reasons'].split(';')
            nominal=any(x.startswith('nominal/') for x in reasons)
            curriculum=any(x.startswith('curriculum') for x in reasons)
            (both if nominal and curriculum else nominal_only if nominal else curriculum_only).append(r)
        rejections[arm]=dict(blocks=len(rows),rejections=len(rejected),fraction=len(rejected)/len(rows),
            nominal_only=len(nominal_only),curriculum_only=len(curriculum_only),both=len(both),
            rejected_candidates_with_curriculum_success_ge80=sum(float(r['curriculum_candidate_success'])>=.8 for r in rejected),
            max_rejected_curriculum_success=max([float(r['curriculum_candidate_success']) for r in rejected] or [0]),
            note='High curriculum success can coincide with nominal decline; rejection alone does not establish which loss caused drift.')
        all_episodes=[]
        for seed in range(5):
            with (OUT/f'episodes_P11{arm}_seed{seed}.csv').open() as stream:all_episodes.extend(csv.DictReader(stream))
        n=len(all_episodes);scores=[float(r['quality_score']) for r in all_episodes]
        successes=sum(float(r['success']) for r in all_episodes)
        quality[arm]=dict(episodes=n,successes=int(successes),pooled_success_ratio=successes/n,
            seed_mean_success_ratio=summary['groups'][arm]['online_success_ratio']['mean'],
            pooled_low_score_lt_0p1=sum(q<.1 for q in scores)/n,
            pooled_middle_score_0p1_0p9=sum(.1<=q<.9 for q in scores)/n,
            pooled_high_score_ge_0p9=sum(q>=.9 for q in scores)/n,
            note='Pooled proportions weight seeds by completed episodes; reported inference uses equal seed weight.')
    result=dict(passed=not failures,matched_condition_comparisons=len(comparisons),condition_plan_failures=failures,
        rejection_analysis=rejections,quality_analysis=quality,
        postprocessing_repair='NumPy1.26 uses np.trapz for the identical trapezoid AUC; simulation, training and evaluation unchanged. First failure log preserved.')
    (OUT/'review_evidence.json').write_text(json.dumps(result,indent=2))
    if failures:raise RuntimeError('Heldout physical plans differ')
    print(json.dumps({k:v for k,v in result.items() if k!='condition_plan_failures'},indent=2))
if __name__=='__main__':main()
