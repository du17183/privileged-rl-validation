"""Evaluation-only, restartable Phase15.2 pipeline. No training entry points."""
import argparse
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path
import phase15_1.run as scheduler
from phase15_2.common import ROOT,RESULTS,LOGS,read,save_json,sha256,frozen

# Process-local scheduler output redirection; historical source is unchanged.
scheduler.RESULTS=RESULTS;scheduler.LOGS=LOGS
ART=scheduler.ART;GRADES=scheduler.GRADES;MAIN=scheduler.MAIN


class Runner(scheduler.Runner):
    def __init__(self,replicas=4):
        super().__init__(replicas);self.backend='graph'

    def pair(self,name,args,gpu,backend=None):
        socket=RESULTS/'sockets'/(name+'.sock')
        if socket.exists():raise RuntimeError('Socket requires identity audit '+str(socket))
        server=name+'_inference';evaluator=name+'_evaluation'
        self.launch(server,[str(ROOT/'.venv_pi05/bin/python'),'-u','-m','phase15_2.inference_worker',
            '--backend',backend or self.backend,'--socket',str(socket)],gpu)
        self.launch(evaluator,[str(ROOT/'.venv/bin/python'),'-u','-m',args[0],*args[1:],
            '--socket',str(socket),'--device','cuda:0'],gpu)
        return [server,evaluator]

    def abort(self):
        for kind in ['evaluation','inference']:
            live=[(n,p) for n,p in self.procs.items() if n.endswith('_'+kind) and p.poll() is None]
            def owned(n,p):
                cmd=scheduler.identity(p.pid)
                return cmd and all(str(a) in cmd for a in self.records[n]['command'])
            for n,p in live:
                if owned(n,p):os.kill(p.pid,signal.SIGINT)
            deadline=time.monotonic()+15
            while any(p.poll() is None for _,p in live) and time.monotonic()<deadline:time.sleep(.2)
            for n,p in live:
                if p.poll() is None and owned(n,p):
                    self.records[n]['forced_cleanup_after_graceful_timeout']=True
                    os.kill(p.pid,signal.SIGKILL)
        self.record()

    def preflight(self):
        self.stage='preflight';self.record()
        audit=read(ROOT/'phase15_1/results/final_audit.json')
        if not audit['passed']:raise RuntimeError('Historical independent audit not passed')
        path=RESULTS/'frozen_inputs.json'
        if not path.exists():
            files=dict(read(ROOT/'phase15_1/results/frozen_inputs.json')['files'])
            for source in (ROOT/'phase15_1').rglob('*.py'):
                files[str(source.relative_to(ROOT))]=sha256(source)
            for relative in ['phase15_1/results/selection.json','phase15_1/results/summary.json',
                    'phase15_1/results/final_audit.json','docs/phase15_1_reacquire_latency_report.md',
                    'datasets/random_door_expert/collection_v1/trajectories.h5',
                    'datasets/random_door_expert/split_v1/split.json']:
                files[relative]=sha256(ROOT/relative)
            save_json(path,dict(files=files,rl_updates=0))
        self.verify_frozen()
        protocol=dict(revision='v1',methods={'A':'Phase15.1 final B/15/15/graph',
            'B':'A + validation-selected retreat','C':'B + probe/Valid Grasp/strict Normal Entry',
            'D':'C + 15tick transition','E':'D + train-only geometric pregrasp/progress guard'},
            normal_recovery_checkpoints={str(s):frozen(s) for s in range(5)},
            test_natural_per_model_seed=100,validation_natural_per_model_seed=32,
            source='frozen Phase15.1 final system; source must actually fail',
            source_reset_seed_base=35100000,source_test_seed_offset=1000000,
            source_snapshot_rule='one uniform reservoir sample per failed source episode, age120-420, distance10-24cm',
            cohort_pairing='same initial state/prefix/reset and inference noise within model seed for A-E',
            artificial_test_per_case=100,graded_test_per_case=100,
            fixed_test_per_seed=64,random_test_per_seed=128,full_loop_test_per_seed=128,
            episode_budget_ticks=600,training=False,rl_updates=0,
            backend='frozen previously verified graph',training_horizon=10,inference_horizon=15,
            inference_horizon_note='actual suffix extrapolation, not padded/repeated actions',
            selection_rule='validation lexicographic mean safe-pose/pregrasp/final natural success; R0 first on ties',
            natural_distance_bins_m=[.10,.125,.175,.25],
            unchanged=['task','reward','reset','action_interface','model_weights','collision_boundaries'],
            realtime_note='simulation waits for inference; GPU throughput is not robot latency')
        if (RESULTS/'protocol.json').exists() and read(RESULTS/'protocol.json')!=protocol:
            raise RuntimeError('Locked protocol changed')
        save_json(RESULTS/'protocol.json',protocol)
        save_json(RESULTS/'code_manifest.json',dict(files={str(p.relative_to(ROOT)):sha256(p)
            for p in (ROOT/'phase15_2').rglob('*.py')},created=time.time()))
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','phase15_2.diagnostics.unit_gate'],cwd=ROOT,check=True)

    def job(self,method,seed,split,retreat='R2',label=None):
        return dict(arm='D',method=method,label=label or method,seed=seed,retreat='R0' if method=='A' else retreat,
            evaluation_seed=38100100+(1000000 if split=='test' else 0)+seed*100,
            full_loop_seed=39100100+(1000000 if split=='test' else 0)+seed*100,
            fixed_n=64 if split=='test' else 32,random_n=128 if split=='test' else 32,
            tests=MAIN,cohorts=f'phase15_2/results/cohorts/{split}/seed{seed}',
            output=f'phase15_2/results/{split}/{label or method}_seed{seed}',checkpoints=frozen(seed),
            pressure_clock_restore=True,evaluation_revision='phase15_2_v1')

    def evaluate(self,jobs,prefix,backend=None):
        tasks=[]
        for index,job in enumerate(jobs):
            out=ROOT/job['output']
            if all((out/(test+'.json')).exists() and (out/(test+'.telemetry.json')).exists()
                    for test in job['tests']):continue
            path=RESULTS/'jobs'/prefix/f'{index:03d}.json';save_json(path,[job])
            tasks.append(['phase15_2.evaluation.full_loop_eval','--jobs',str(path)])
        if tasks:self.dispatch(tasks,prefix,backend or 'graph')

    def pilot(self):
        if (RESULTS/'pilot/gate.json').exists():
            if not read(RESULTS/'pilot/gate.json')['passed']:raise RuntimeError('Pilot gate rejected')
            return
        self.stage='technical_physics_pilot';self.record()
        import h5py
        target=RESULTS/'cohorts/pilot/seed0';target.mkdir(parents=True,exist_ok=True)
        for case in ['natural_severe','ee_offset']:
            dst=target/(case+'.h5')
            if not dst.exists():
                src=ROOT/f'phase15_1/results/cohorts/validation/{case}.h5'
                with h5py.File(src) as a,h5py.File(dst,'x') as b:
                    for key in sorted(a)[:32]:a.copy(key,b,name=key)
                    b.attrs.update(evaluation_only=True,diagnostic_only=True,source_sha256=sha256(src))
        jobs=[]
        for method,retreat in [('A','R0'),('B','R1'),('B','R2'),('B','R3'),('C','R2'),('D','R2'),('E','R2')]:
            label=method+'_'+retreat
            job=self.job(method,0,'pilot',retreat,label);job['tests']=['natural_severe','ee_offset','random']
            job['diagnostic_only']=True;jobs.append(job)
        self.evaluate(jobs,'pilot_v4')
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','phase15_2.diagnostics.physics_gate'],cwd=ROOT,check=True)
        self.verify_frozen()

    def cohorts(self):
        self.stage='independent_cohort_collection';self.record();plan=[];tasks=[]
        for split in ['validation','test']:
            for seed in range(5):
                n=32 if split=='validation' else 100
                reset=35100000+(1000000 if split=='test' else 0)+seed*10000
                out=RESULTS/f'cohorts/source/{split}/seed{seed}/natural_severe'
                plan.append(dict(split=split,model_seed=seed,case='natural_severe',episodes=n,reset_seed=reset,output=str(out)))
            for j,case in enumerate(ART+GRADES):
                n=32 if split=='validation' else 100
                reset=37100000+(1000000 if split=='test' else 0)+j*10000
                out=RESULTS/f'cohorts/source/{split}/seed0/{case}'
                plan.append(dict(split=split,model_seed=0,case=case,episodes=n,reset_seed=reset,output=str(out)))
        path=RESULTS/'collection_plan.json'
        if path.exists() and read(path)!=plan:raise RuntimeError('Collection plan changed')
        save_json(path,plan)
        for row in plan:
            out=Path(row['output'])
            if (out/'complete.json').exists():continue
            tasks.append(['phase15_2.evaluation.natural_severe_eval','--output',str(out),
                '--model-seed',str(row['model_seed']),'--reset-seed',str(row['reset_seed']),
                '--cases',row['case'],'--episodes',str(row['episodes']),'--natural-episodes',str(row['episodes']),
                '--max-screening-episodes','8192'])
        self.dispatch(tasks,'cohort_v1','graph')
        import h5py
        identities={}
        for split in ['validation','test']:
            ids=set()
            for seed in range(5):
                dest=RESULTS/f'cohorts/{split}/seed{seed}';dest.mkdir(parents=True,exist_ok=True);manifest={}
                for case in ['natural_severe']+ART+GRADES:
                    row=next(r for r in plan if r['split']==split and r['case']==case and
                        r['model_seed']==(seed if case=='natural_severe' else 0))
                    source=Path(row['output'])/(case+'.h5');file=dest/(case+'.h5')
                    if not file.exists():
                        with h5py.File(source,'r') as src,h5py.File(file,'x') as h:
                            for key in sorted(src):src.copy(key,h,name=key)
                            h.attrs.update(evaluation_only=True,training_labels=False,source_sha256=sha256(source),
                                collection_plan_sha256=sha256(path),split=split,model_seed=seed,rl_updates=0)
                    with h5py.File(file,'r') as h:
                        if len(h)!=row['episodes']:raise RuntimeError('Incomplete independent cohort')
                        if case=='natural_severe':
                            for g in h.values():
                                if not bool(g.attrs['natural_source_episode_failed']):raise RuntimeError('Natural sample lacks failed source')
                                identity=tuple(int(g.attrs[k]) for k in ['reference_model_seed','reset_seed','clone','source_episode'])
                                if identity in ids:raise RuntimeError('Duplicate natural source episode')
                                ids.add(identity)
                        manifest[case]=dict(episodes=len(h),sha256=sha256(file),source=str(source),source_sha256=sha256(source))
                    if sha256(source)!=sha256(file):
                        # Attributes intentionally differ; group payloads must match instead.
                        with h5py.File(source,'r') as src,h5py.File(file,'r') as h:
                            for key in src:
                                for field in src[key]:
                                    import numpy as np
                                    if not np.array_equal(src[key][field][:],h[key][field][:]):raise RuntimeError('Cohort copy changed physical input')
                    save_json(dest/'manifest.json',manifest)
            identities[split]=ids
        if identities['validation']&identities['test']:raise RuntimeError('Validation/test overlap')
        save_json(RESULTS/'cohort_integrity_gate.json',dict(passed=True,natural_validation=len(identities['validation']),
            natural_test=len(identities['test']),independent_source_episodes=True,training=False,rl_updates=0))

    def select(self):
        path=RESULTS/'selection.json'
        if path.exists():return read(path)
        self.stage='retreat_validation';self.record()
        jobs=[]
        for retreat in ['R0','R1','R2','R3']:
            for seed in range(5):
                job=self.job('B',seed,'validation',retreat,label='B_'+retreat)
                job['tests']=['natural_severe'];jobs.append(job)
        self.evaluate(jobs,'retreat_validation')
        scores={}
        for retreat in ['R0','R1','R2','R3']:
            seeds=[]
            for seed in range(5):
                rows=read(RESULTS/f'validation/B_{retreat}_seed{seed}/natural_severe.telemetry.json')
                seeds.append([sum(any(p.get('safe_retreat') for p in r['planner']) for r in rows)/len(rows),
                    sum(any(p.get('reached_pregrasp') for p in r['planner']) for r in rows)/len(rows),
                    sum(r['success'] for r in rows)/len(rows)])
            scores[retreat]=[sum(row[k] for row in seeds)/5 for k in range(3)]
        retreat=max(scores,key=lambda r:tuple(scores[r]))
        selected=dict(retreat=retreat,validation_scores=scores,selected_by='validation_only',
            selection_rule=read(RESULTS/'protocol.json')['selection_rule'],backend='graph',
            schedule=dict(normal=15,recovery=15),horizons=dict(normal=15,recovery=15),rl_updates=0)
        save_json(path,selected);return selected

    def formal(self,selected):
        self.stage='formal_five_model_seed_evaluation';self.record()
        jobs=[]
        for method in ['A','B','C','D','E']:
            for seed in range(5):
                job=self.job(method,seed,'test',selected['retreat']);jobs.append(job)
                jobs.append(dict(job,tests=GRADES,force_planner=True,output=job['output']+'_grades'))
        self.evaluate(jobs,'formal_v1')
        self.stage='exclusive_batch1_latency';self.record();old=self.replicas;self.replicas=1
        tasks=[['phase15_2.evaluation.deployment_benchmark','--seed',str(seed),'--normal','15','--recovery','15']
            for seed in range(5) if not (RESULTS/f'deployment_latency/seed{seed}.json').exists()]
        try:self.dispatch(tasks,'deployment_latency','graph')
        finally:self.replicas=old

    def main(self,pilot_only=False):
        self.preflight();self.pilot()
        if pilot_only:self.stage='pilot_complete';self.record();return
        self.cohorts();selected=self.select();self.formal(selected)
        self.stage='statistics_and_audit';self.record();self.verify_frozen()
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','phase15_2.report'],cwd=ROOT,check=True)
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','phase15_2.diagnostics.final_audit'],cwd=ROOT,check=True)
        self.stage='complete';self.record()
        save_json(RESULTS/'experiment_complete.json',dict(complete=True,report='docs/phase15_2_recovery_chain_report.md',
            success_criteria_met=read(RESULTS/'summary.json')['success_criteria_met'],training=False,rl_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replicas-per-gpu',type=int,choices=[1,2,3,4],default=4)
    p.add_argument('--pilot-only',action='store_true');a=p.parse_args();runner=Runner(a.replicas_per_gpu)
    try:runner.main(a.pilot_only)
    except BaseException:
        save_json(RESULTS/'pipeline_failed.json',dict(stage=runner.stage,error=traceback.format_exc(),rl_updates=0))
        runner.abort();runner.stage='failed';runner.record();raise
