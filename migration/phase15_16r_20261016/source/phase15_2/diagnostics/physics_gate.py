"""Technical gate; negative task success is retained and does not reject a method."""
from pathlib import Path
import h5py
from phase15_2.common import ROOT,RESULTS,read,save_json,sha256

def run():
    stale=[]
    for label in ['C_R2','D_R2','E_R2']:
        for test in ['natural_severe','ee_offset','random']:
            path=RESULTS/f'pilot/{label}_seed0/{test}.json'
            if read(path)['chain_config'].get('entry_distance_exit_m')!=.06:stale.append(str(path))
    if stale:
        save_json(RESULTS/'entry_gate_requires_rerun.json',dict(affected=['C_R2','D_R2','E_R2'],
            stale=stale,reason='explicit absolute 60mm Normal Entry gate requires fresh physics validation',
            planned_engineering_revision=True,negative_success_not_the_reason=True,rl_updates=0))
        raise RuntimeError('Explicit absolute entry distance requires v4 physical rerun; original data retained')
    checks=[]
    for label in ['A_R0','B_R1','B_R2','B_R3','C_R2','D_R2','E_R2']:
        for test in ['natural_severe','ee_offset','random']:
            path=RESULTS/f'pilot/{label}_seed0/{test}.json';d=read(path);rows=read(path.with_suffix('.telemetry.json'))
            if len(rows)!=32 or d['episodes']!=32:raise RuntimeError('Pilot endpoint coverage mismatch')
            if len({(r['clone'],r['episode']) for r in rows})!=32:raise RuntimeError('Repeated pilot endpoint')
            if any(r['original_episode_ticks']>600 for r in rows):raise RuntimeError('Pilot budget expanded')
            if abs(sum(r['success'] for r in rows)/32-d['success'])>1e-9:raise RuntimeError('Pilot success mismatch')
            with h5py.File(path.with_suffix('.stage_trace.h5'),'r') as h:
                data=h['steps'][:]
                if int(data['terminal'].sum())<32:raise RuntimeError('Stage trace lacks terminal coverage')
                if data['actual_arm_collision_known'].any():raise RuntimeError('Uninstrumented physical collision claimed')
                if (data['remaining_ticks']<0).any():raise RuntimeError('Negative remaining budget')
            if test!='random':
                if not d['cohort_sha256']==sha256(RESULTS/f'cohorts/pilot/seed0/{test}.h5'):raise RuntimeError('Wrong pilot inputs')
            checks.append(dict(label=label,test=test,episodes=32,success=d['success'],trace_records=len(data)))
    save_json(RESULTS/'pilot/gate.json',dict(passed=True,technical_only=True,checks=checks,
        selection_uses_new_validation_only=True,task_success_not_a_technical_gate=True,
        single_model_seed_diagnostic=True,rl_updates=0))

if __name__=='__main__':run()
