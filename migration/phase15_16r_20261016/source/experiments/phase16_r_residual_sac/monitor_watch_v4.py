"""Bounded read-only monitoring; never starts or signals an experiment worker."""
import sys,json,hashlib,subprocess,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,sha,save,utc
from experiments.phase16_r_residual_sac.orchestrate_v4 import proc

def main():
    save(RES/'monitor/watch_v4_launch.json',dict(utc=utc(),process=proc(__import__('os').getpid()),source_sha256=sha(Path(__file__)),read_only_execution=True,workers_started=0,workers_signalled=0))
    last=None
    for i in range(720):
        assert sha(EXP/'monitor_v3.py')=='15b01d220d7bdcf3d53757b19b351fe5c83f5531c5781597b4320e2efe70829a'
        for version in [3,4]:
            for rel,wanted in json.loads((RES/f'supplementary_source_v{version}.json').read_text())['files'].items():assert sha(ROOT/rel)==wanted
        result=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','experiments.phase16_r_residual_sac.monitor_v3'],capture_output=True,text=True)
        if result.returncode:print('MONITOR_TOOL_ERROR',result.stderr[-2000:],flush=True);break
        sample=json.loads(sorted((RES/'monitor').glob('*_v3.json'))[-1].read_text());nodes=[]
        for path in sorted((RES/'train').glob('*/node*.validation_done.json')):
            node=json.loads(path.read_text());nodes.append([path.parent.name,path.name,node.get('sampled_success'),node.get('passed')])
        progress=[[v['arm'],v['seed'],v['steps'],v['updates'],v['online_successes']] for v in sample['training_progress']]
        active=json.loads((RES/'lifecycle.json').read_text())['active'];finished=sample['first_round_complete'] or (RES/'first_round_execution_finished.json').exists()
        state=dict(passed=sample['passed'],issues=sample['issues'],progress=progress,nodes=nodes,active=[[v['key'],v['gpu']] for v in active],pipeline_failed=sample['pipeline_failed'],execution_finished=finished)
        if state!=last:print(json.dumps(dict(utc=sample['utc'],**state)),flush=True);last=state
        if not sample['passed'] or sample['pipeline_failed'] or finished or (RES/'training_handoff_v4/failed.json').exists():break
        time.sleep(30)
    save(RES/'monitor/watch_v4_exit.json',dict(utc=utc(),last_observation=last,workers_started=0,workers_signalled=0,source_sha256=sha(Path(__file__))))
if __name__=='__main__':main()
