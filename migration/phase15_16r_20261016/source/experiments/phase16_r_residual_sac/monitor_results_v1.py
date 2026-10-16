"""Read completed rollout metadata and print changes; never starts a worker."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,sha,save,utc,protocol

def main():
    protocol();record=RES/'monitor/results_reader_launch.json'
    assert not record.exists(),'Reader already registered; inspect its process before repeating'
    save(record,dict(utc=utc(),pid=__import__('os').getpid(),source_sha256=sha(Path(__file__)),scope='read completed evaluation JSON only; zero physical interactions and zero updates'))
    seen={};names={f'{scope}_{dist}.json' for scope in ['validation','random','fixed'] for dist in ['sampled','mean']}
    for _ in range(720):
        for fail in [RES/'pipeline_failed.json',RES/'training_handoff_v5/failed.json']:
            if fail.exists():
                print(json.dumps(dict(utc=utc(),stop='current pipeline failure',path=str(fail.relative_to(ROOT)))),flush=True);return
        for path in sorted((RES/'evaluation').glob('*/*/*.json')):
            if path.name not in names:continue
            timestamp=path.stat().st_mtime_ns
            if seen.get(path)==timestamp:continue
            data=json.loads(path.read_text());seen[path]=timestamp
            print(json.dumps(dict(utc=utc(),completed_rollout=str(path.relative_to(RES)),success=data['success'],episodes=data['episodes'],physical_interactions=data['physical_interactions'],wall_s=data['elapsed_s'],preliminary_single_seed=True)),flush=True)
        if (RES/'first_round_complete.json').exists() or (RES/'first_round_execution_finished.json').exists():
            print(json.dumps(dict(utc=utc(),stop='main execution ended, postprocessing may still be pending')),flush=True);return
        time.sleep(30)
    print(json.dumps(dict(utc=utc(),stop='bounded read-only watch elapsed')),flush=True)
if __name__=='__main__':main()
