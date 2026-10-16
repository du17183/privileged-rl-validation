"""Restartable evaluation-only pipeline with independent validation/test cohorts."""
import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import time
import traceback
from pathlib import Path
from phase15_1.common import ROOT,RESULTS,LOGS,save_json,sha256,frozen

ART=['ee_offset','contact_loss','door_regression','stagnation']
GRADES=['grade_05','grade_10','grade_15','grade_20']
MAIN=['fixed','random']+ART+['natural_severe','full_loop']


def read(path):return json.loads(Path(path).read_text())


def identity(pid):
    try:return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return ''


class AdoptedProcess:
    """Track a verified orphan; successful completion requires output evidence."""
    def __init__(self,pid,command):self.pid=pid;self.command=command
    def poll(self):
        current=identity(self.pid)
        if current and current!=self.command:raise RuntimeError('Adopted PID reused')
        return None if current else 0  # Disappearance, not an observed exit status.
    def terminate(self):
        if identity(self.pid)==self.command:os.kill(self.pid,15)


class Runner:
    def __init__(self,replicas=4,resume=None,collection_plan=None):
        RESULTS.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
        self.lock=(RESULTS/'supervisor.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.stage='preflight';self.procs={};self.records={};self.counter=0
        self.backend='unroll'
        self.replicas=replicas
        self.collection_plan=collection_plan
        self.adopted={}
        if resume:
            snapshot=read(resume)
            if snapshot['stage']!='independent_cohort_collection':raise RuntimeError('Unsupported live adoption stage')
            for name,row in snapshot['processes'].items():
                cmd=identity(row['pid'])
                if not cmd:continue
                if not name.startswith('cohort_') or '-m phase15_1.' not in cmd:raise RuntimeError('Unexpected process adoption')
                if any(str(arg) not in cmd for arg in row['command']):raise RuntimeError('Process identity mismatch')
                self.procs[name]=AdoptedProcess(row['pid'],cmd)
                self.records[name]=dict(row,adopted=True,exit_status_observable=False)
                self.adopted.setdefault(name.rsplit('_',1)[0],[]).append(name)

    def record(self,detail=None):
        save_json(RESULTS/'lifecycle.json',dict(supervisor_pid=os.getpid(),stage=self.stage,processes=self.records,updated=time.time(),rl_updates=0))
        save_json(RESULTS/'progress.json',dict(supervisor_pid=os.getpid(),status=self.stage,detail=detail,updated=time.time(),
            max_pairs_per_gpu=self.replicas,environments_per_pair=32,max_parallel_environments=8*self.replicas*32,rl_updates=0))

    def launch(self,name,args,gpu):
        if name in self.procs and self.procs[name].poll() is None:raise RuntimeError('Duplicate child')
        command='cd '+shlex.quote(str(ROOT))+' && source configs/runtime_env.sh && '+shlex.join(args)
        log=LOGS/(name+'.log');f=log.open('ab',buffering=0)
        proc=subprocess.Popen(['bash','-lc',command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
            start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PHASE15_1_GRAPH_MEMORY_GIB='48'))
        self.procs[name]=proc;self.records[name]=dict(pid=proc.pid,command=args,gpu=gpu,log=str(log),started=time.time())
        self.record();return proc

    def wait(self,names):
        while True:
            left=[]
            for name in names:
                proc=self.procs[name];code=proc.poll()
                if code is None:left.append(name)
                else:
                    self.records[name]['exit_code']=code
                    if code!=0:raise RuntimeError(f'{name} exited {code}; original log: {self.records[name]["log"]}')
            self.record(dict(waiting=left))
            if not left:return
            time.sleep(5)

    def bootstrap(self,pilot_revision):
        self.stage='waiting_initial_pilots';self.record()
        path=RESULTS/pilot_revision/'lifecycle.json'
        if path.exists():
            for name,row in read(path).items():
                if not name.startswith('pilot_'):continue
                while identity(row['pid']):
                    cmd=identity(row['pid'])
                    if '-m phase15_1.' not in cmd:raise RuntimeError('PID reused: '+str(row['pid']))
                    self.record(dict(waiting_for_external=name,pid=row['pid']));time.sleep(5)
        for variant in ['B','C','D']:
            jobs=RESULTS/pilot_revision/f'jobs_{variant}.json'
            if not Path(str(jobs)+'.slot0.complete.json').exists():
                raise RuntimeError('Initial controller pilot missing/failing: '+variant)
        gate=read(RESULTS/'diagnosis/gate.json')
        if gate['recorded_episodes']!=64 or not gate['required_state_and_command_fields_present']:raise RuntimeError('Original diagnosis incomplete')
        self.verify_frozen()

    def verify_frozen(self):
        files=read(RESULTS/'frozen_inputs.json')['files']
        bad=[]
        for name,digest in files.items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            if sha256(path)!=digest:bad.append(name)
        save_json(RESULTS/'frozen_verification.json',dict(checked=len(files),changed=bad,passed=not bad,rl_updates=0))
        if bad:raise RuntimeError('Historical inputs changed')

    def abort(self):
        # Close evaluator HDF5 contexts before terminating inference servers.
        for kind in ['evaluation','inference']:
            live=[(name,proc) for name,proc in self.procs.items() if name.endswith('_'+kind) and proc.poll() is None]
            for name,proc in live:
                if '-m phase15_1.' in identity(proc.pid):os.kill(proc.pid,signal.SIGINT)
            deadline=time.monotonic()+15
            while any(identity(proc.pid) for _,proc in live) and time.monotonic()<deadline:time.sleep(.2)
            for name,proc in live:
                if '-m phase15_1.' in identity(proc.pid):
                    self.records[name]['forced_cleanup_after_graceful_timeout']=True
                    os.kill(proc.pid,signal.SIGKILL)
        self.record()

    def latency(self):
        self.stage='latency_engineering';self.record()
        names=[]
        for gpu,backend in zip([4,5,6,7],['cached','graph','compile_approx','unroll']):
            sentinel=RESULTS/f'latency/verified_v3/{backend}.complete.json'
            if sentinel.exists():continue
            name='latency_v3_'+backend;names.append(name)
            self.launch(name,[str(ROOT/'.venv_pi05/bin/python'),'-u','-m','phase15_1.inference_opt.latency_benchmark',
                '--backend',backend,'--revision','verified_v3'],gpu)
        self.wait(names)
        candidates=[]
        for backend in ['graph','compile_approx','cached','unroll']:
            rows=[RESULTS/f'latency/verified_v3/{backend}_{module}_chunk{chunk}.json'
                  for module in ['normal','recovery'] for chunk in [5,10,15,20,30]]
            if all(p.exists() and 'failed' not in read(p) for p in rows):
                candidates.append((sum(read(p)['model_mean_s'] for p in rows)/len(rows),backend))
        if candidates:self.backend=min(candidates)[1]
        save_json(RESULTS/'engineering_selection.json',dict(backend=self.backend,candidates=candidates,
            provisional=True,selected_by='single_gpu_speed_after_numerical_gate; task retention still required',rl_updates=0))

    def pair(self,name,args,gpu,backend=None):
        socket=RESULTS/'sockets'/(name+'.sock')
        if socket.exists():
            # Reusing an abandoned socket is never automatic.
            raise RuntimeError('Retained socket requires identity audit: '+str(socket))
        server=name+'_inference';evaluator=name+'_evaluation'
        self.launch(server,[str(ROOT/'.venv_pi05/bin/python'),'-u','-m','phase15_1.inference_worker','--backend',backend or self.backend,'--socket',str(socket)],gpu)
        self.launch(evaluator,[str(ROOT/'.venv/bin/python'),'-u','-m',args[0],*args[1:],'--socket',str(socket),'--device','cuda:0'],gpu)
        return [server,evaluator]

    def dispatch(self,tasks,prefix,backend=None):
        pending=list(enumerate(tasks));active={}
        arguments={}
        if prefix.startswith('cohort') and self.adopted:
            for name,names in self.adopted.items():
                if len(names)!=2:raise RuntimeError('Incomplete adopted pair; retained for audit')
                row=self.records[name+'_evaluation'];command=row['command']
                output=command[command.index('--output')+1]
                match=next(((i,args) for i,args in pending if '--output' in args and args[args.index('--output')+1]==output),None)
                if match is None:raise RuntimeError('Adopted collection has no matching task')
                pending.remove(match);slot=row['gpu']
                while slot in active:slot+=8
                active[slot]=names;arguments[slot]=match[1]
            self.adopted={}
        while pending or active:
            for slot in range(8*self.replicas):
                if slot in active or not pending:continue
                gpu=slot%8
                inventory=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
                uuid={int(line.split(',')[0]):line.split(',')[1].strip() for line in inventory.splitlines()}[gpu]
                occupied=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
                own={p.pid for p in self.procs.values() if p.poll() is None}
                if any(line.split(',')[0].strip()==uuid and int(line.split(',')[1]) not in own for line in occupied.splitlines() if ',' in line):continue
                memory=subprocess.run(['nvidia-smi','--id',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
                if float(memory.strip())<65000:continue
                index,args= pending.pop(0);name=prefix+f'_{index:03d}'
                active[slot]=self.pair(name,args,gpu,backend)
                arguments[slot]=args
            for slot,names in list(active.items()):
                codes=[self.procs[n].poll() for n in names]
                if any(c is not None and c!=0 for c in codes):raise RuntimeError('Evaluation worker failure '+str(names))
                if all(c is not None for c in codes):
                    args=arguments[slot]
                    if '--output' in args:
                        evidence=Path(args[args.index('--output')+1])/'complete.json'
                    elif '--jobs' in args:
                        evidence=Path(args[args.index('--jobs')+1]+'.slot0.complete.json')
                    else:evidence=RESULTS/f'deployment_latency/seed{args[args.index("--seed")+1]}.json'
                    if not evidence.exists():raise RuntimeError('Finished pair lacks completion evidence: '+str(evidence))
                    for n,c in zip(names,codes):
                        self.records[n]['exit_code']=None if self.records[n].get('adopted') else c
                        self.records[n]['completion_evidence']=str(evidence)
                    del active[slot]
                    del arguments[slot]
            self.record(dict(pending=len(pending),active_pairs=len(active),active_environments=0 if prefix=='deployment_latency' else len(active)*32));time.sleep(5)

    def cohorts(self):
        self.stage='independent_cohort_collection';self.record()
        if self.collection_plan:return self.planned_cohorts()
        tasks=[]
        for split in ['validation','test']:
            for seed in range(5):
                out=RESULTS/f'cohorts/source/{split}/seed{seed}'
                if (out/'complete.json').exists():continue
                cases=['natural_severe']
                if seed==0:cases+=ART+GRADES
                n=32 if split=='validation' else 100 if seed==0 else 32
                # Natural cohorts balance all five source models, 32 each.
                reset=21100000+(1000000 if split=='test' else 0)+seed*10000
                tasks.append(['phase15_1.evaluation.severe_deviation_eval','--output',str(out),'--model-seed',str(seed),
                    '--reset-seed',str(reset),'--cases',','.join(cases),'--episodes',str(n),'--max-screening-episodes','4096'])
        self.dispatch(tasks,'cohort',backend='graph' if self.backend=='graph' else 'unroll')
        import h5py
        for split in ['validation','test']:
            dest=RESULTS/f'cohorts/{split}';dest.mkdir(parents=True,exist_ok=True);manifest={}
            for case in ['natural_severe']+ART+GRADES:
                file=dest/(case+'.h5')
                sources=[RESULTS/f'cohorts/source/{split}/seed{s}/{case}.h5' for s in (range(5) if case=='natural_severe' else [0])]
                if not file.exists():
                    with h5py.File(file,'x') as h:
                        count=0
                        for source in sources:
                            with h5py.File(source,'r') as src:
                                chosen=sorted(src)[:32] if case=='natural_severe' else sorted(src)
                                for key in chosen:
                                    src.copy(key,h,name=f'state_{count:04d}');count+=1
                        h.attrs.update(evaluation_only=True,split=split,training_labels=False,rl_updates=0)
                with h5py.File(file,'r') as h:
                    count=len(h)
                    if case=='natural_severe' and not all(bool(g.attrs['natural_source_episode_failed']) for g in h.values()):raise RuntimeError('Natural proxy found')
                minimum=100 if split=='test' else 32
                if count<minimum:raise RuntimeError('Cohort quota failed')
                manifest[case]=dict(episodes=count,sha256=sha256(file),sources={str(p):sha256(p) for p in sources})
            save_json(dest/'manifest.json',manifest)

    def planned_cohorts(self):
        plan=read(self.collection_plan);tasks=[]
        for row in plan['tasks']:
            out=Path(row['output'])
            if (out/'complete.json').exists():continue
            args=['phase15_1.evaluation.severe_deviation_eval','--output',str(out),'--model-seed',str(row['model_seed']),
                '--reset-seed',str(row['reset_seed']),'--cases',row['case'],'--episodes',str(row['episodes']),
                '--natural-episodes','32','--max-screening-episodes','4096']
            if row.get('resume_from'):args+=['--resume-from',row['resume_from']]
            tasks.append(args)
        # Independent validation inputs can be finalized while a test collector
        # is still running. Execute the same planned validation jobs once; the
        # later selection stage reuses their completed evidence.
        validation_members=[row for row in plan['tasks'] if row['split']=='validation']
        if all((Path(row['output'])/'complete.json').exists() for row in validation_members):
            self.assemble_planned(plan,['validation'])
            prefetched=0
            for variant in ['A','B','C','D']:
                for seed in range(5):
                    job=self.job(variant,seed,'validation');out=ROOT/job['output']
                    if all((out/(test+'.json')).exists() for test in job['tests']):continue
                    path=RESULTS/f'jobs/controller_val_prefetch/{variant}_seed{seed}.json';save_json(path,[job])
                    tasks.append(['phase15_1.evaluation.full_system_eval','--jobs',str(path)]);prefetched+=1
            save_json(RESULTS/'validation_prefetch.json',dict(jobs=prefetched,
                protocol='same controller validation jobs, overlap with independent test collection',rl_updates=0))
        self.dispatch(tasks,'cohort_'+plan['revision'],backend='graph' if self.backend=='graph' else 'unroll')
        self.assemble_planned(plan,['validation','test'])

    def assemble_planned(self,plan,splits):
        import h5py
        for split in splits:
            dest=RESULTS/f'cohorts/{split}';dest.mkdir(parents=True,exist_ok=True);manifest={}
            manifest_path=dest/'manifest.json'
            if manifest_path.exists():
                existing=read(manifest_path)
                for case,row in existing.items():
                    if sha256(dest/(case+'.h5'))!=row['sha256']:raise RuntimeError('Finalized cohort changed')
                    if any(sha256(source)!=digest for source,digest in row['sources'].items()):raise RuntimeError('Finalized cohort source changed')
                continue
            for case in ['natural_severe']+ART+GRADES:
                members=[row for row in plan['tasks'] if row['split']==split and row['case']==case]
                if len(members)!=(5 if case=='natural_severe' else 1):raise RuntimeError('Unbalanced planned cohort')
                file=dest/(case+'.h5');sources=[Path(row['output'])/(case+'.h5') for row in members]
                if not file.exists():
                    with h5py.File(file,'x') as h:
                        count=0
                        for source in sources:
                            with h5py.File(source,'r') as src:
                                keys=sorted(src)[:32] if case=='natural_severe' else sorted(src)
                                if len(keys)<(32 if split=='validation' or case=='natural_severe' else 100):raise RuntimeError('Source quota incomplete')
                                for key in keys:src.copy(key,h,name=f'state_{count:04d}');count+=1
                        h.attrs.update(evaluation_only=True,split=split,training_labels=False,rl_updates=0,
                            collection_plan_sha256=sha256(self.collection_plan))
                with h5py.File(file,'r') as h:
                    count=len(h)
                    if case=='natural_severe' and not all(bool(g.attrs['natural_source_episode_failed']) for g in h.values()):raise RuntimeError('Natural proxy found')
                if count<(100 if split=='test' else 32):raise RuntimeError('Combined quota incomplete')
                manifest[case]=dict(episodes=count,sha256=sha256(file),sources={str(p):sha256(p) for p in sources},
                    collection_plan=str(self.collection_plan),collection_plan_sha256=sha256(self.collection_plan))
            save_json(dest/'manifest.json',manifest)

    def job(self,controller,seed,split,label=None,schedule=None):
        schedule=schedule or dict(normal=5,recovery=5)
        return dict(arm='D',controller=controller,label=label or controller,seed=seed,
            evaluation_seed=23100100+(1000000 if split=='test' else 0)+seed*100,
            full_loop_seed=24100100+(1000000 if split=='test' else 0)+seed*100,
            fixed_n=64 if split=='test' else 32,random_n=128 if split=='test' else 64,
            tests=MAIN,cohorts=f'phase15_1/results/cohorts/{split}',
            output=f'phase15_1/results/{split}/{label or controller}_seed{seed}',checkpoints=frozen(seed),
            schedule=schedule,horizons={k:max(10,v) for k,v in schedule.items()},capture_trace=False,pressure_clock_restore=True,evaluation_revision="v11_clock")

    def evaluate(self,jobs,prefix,backend=None):
        tasks=[]
        for index,job in enumerate(jobs):
            out=ROOT/job['output']
            if all((out/(test+'.json')).exists() for test in job['tests']):continue
            path=RESULTS/'jobs'/prefix/f'{index:03d}.json';save_json(path,[job])
            tasks.append(['phase15_1.evaluation.full_system_eval','--jobs',str(path)])
        self.dispatch(tasks,prefix,backend)

    def select(self):
        self.stage='validation';self.record()
        jobs=[self.job(v,s,'validation') for v in ['A','B','C','D'] for s in range(5)]
        self.evaluate(jobs,'controller_val',backend='graph' if self.backend=='graph' else 'unroll')
        def score(v):
            values=[]
            for seed in range(5):
                out=RESULTS/f'validation/{v}_seed{seed}';values.append([read(out/(k+'.json'))['success'] for k in ['natural_severe','full_loop','random']])
            return tuple(sum(row[k] for row in values)/5 for k in range(3))
        controller=max(['A','B','C','D'],key=score)
        schedules={f'uniform_{n}':dict(normal=n,recovery=n) for n in [5,10,15,20,30]}
        schedules.update(normal15_recovery5=dict(normal=15,recovery=5),normal20_recovery5=dict(normal=20,recovery=5),normal20_recovery10=dict(normal=20,recovery=10))
        chunks=[]
        for name,schedule in schedules.items():
            job=self.job(controller,0,'validation',label=name,schedule=schedule)
            job['tests']=['random']+ART+['full_loop'];chunks.append(job)
        self.evaluate(chunks,'chunk_val')
        def rates(name):
            out=RESULTS/f'validation/{name}_seed0'
            return [read(out/'random.json')['success'],sum(read(out/(k+'.json'))['success'] for k in ART)/4,read(out/'full_loop.json')['success']]
        reference=rates('uniform_5');eligible=[]
        for name,schedule in schedules.items():
            r=rates(name)
            if all(v>=b-.03 for v,b in zip(r,reference)):
                eligible.append((sum(60/v for v in schedule.values()),name))
        name=min(eligible)[1] if eligible else 'uniform_5'
        # Numerical fusion candidates also require matched physical reference retention.
        refjob=self.job(controller,0,'validation',label='engineering_reference',schedule=schedules[name]);refjob['tests']=['random']+ART+['full_loop']
        self.evaluate([refjob],'engineering_retention',backend='unroll')
        engineering_reference=rates('engineering_reference');optimized=rates(name)
        retained=all(v>=b-.03 for v,b in zip(optimized,engineering_reference))
        if not retained:self.backend='unroll'
        selected=dict(controller=controller,schedule=schedules[name],backend=self.backend,selected_by='validation_only',
            numerical_candidate_physical_retention=retained,controller_scores={v:score(v) for v in ['A','B','C','D']},
            schedules={k:dict(config=v,validation_rates=rates(k)) for k,v in schedules.items()},rl_updates=0)
        save_json(RESULTS/'selection.json',selected);return selected

    def formal(self,selected):
        self.stage='formal_test';self.record()
        jobs=[]
        for variant in ['A','B','C','D']:
            for seed in range(5):
                job=self.job(variant,seed,'test');jobs.append(job)
                grade=dict(job,tests=GRADES,force_planner=True,output=job['output']+'_grades');jobs.append(grade)
        self.evaluate(jobs,'formal_controllers',backend='graph' if self.backend=='graph' else 'unroll')
        final=[]
        for seed in range(5):
            job=self.job(selected['controller'],seed,'test',label='E_final',schedule=selected['schedule']);final.append(job)
            final.append(dict(job,tests=GRADES,force_planner=True,output=job['output']+'_grades'))
        self.evaluate(final,'formal_final',backend=selected['backend'])
        self.stage='exclusive_single_robot_latency';self.record()
        tasks=[]
        for seed in range(5):
            if (RESULTS/f'deployment_latency/seed{seed}.json').exists():continue
            tasks.append(['phase15_1.inference_opt.deployment_benchmark','--seed',str(seed),
                '--normal',str(selected['schedule']['normal']),'--recovery',str(selected['schedule']['recovery'])])
        replicas=self.replicas;self.replicas=1
        try:self.dispatch(tasks,'deployment_latency',selected['backend'])
        finally:self.replicas=replicas

    def main(self,revision):
        self.stage='preflight';self.record();self.verify_frozen()
        clock_gate=read(RESULTS/'regression_v11/gate.json')
        if not clock_gate['passed']:raise RuntimeError('Pressure clock regression gate incomplete')
        gate=read(RESULTS/'diagnosis/gate.json')
        if gate['recorded_episodes']!=64 or not gate['required_state_and_command_fields_present']:raise RuntimeError('Original diagnosis incomplete')
        # Frozen reference data collection is independent of the new-controller pilot.
        # Scheduler avoids GPUs still occupied by those verified external pilot PIDs.
        self.latency();self.cohorts();self.bootstrap(revision);selected=self.select();self.formal(selected)
        self.stage='analysis';self.record();self.verify_frozen()
        subprocess.run([str(ROOT/'.venv/bin/python'),'-m','phase15_1.report'],cwd=ROOT,check=True)
        self.stage='complete';self.record()
        save_json(RESULTS/'experiment_complete.json',dict(complete=True,report='docs/phase15_1_reacquire_latency_report.md',rl_updates=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--initial-pilots',default='pilot_v1')
    parser.add_argument('--replicas-per-gpu',type=int,choices=[1,2,3,4],default=4)
    parser.add_argument('--adopt-lifecycle');parser.add_argument('--collection-plan');args=parser.parse_args()
    runner=Runner(args.replicas_per_gpu,args.adopt_lifecycle,args.collection_plan)
    try:runner.main(args.initial_pilots)
    except BaseException:
        error=traceback.format_exc();save_json(RESULTS/'pipeline_failed.json',dict(stage=runner.stage,error=error,rl_updates=0))
        runner.abort()
        runner.stage='failed';runner.record();raise
