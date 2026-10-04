"""Read-only compact progress from completed artifacts, not exit codes alone."""
import csv,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_3_pi05_bc';C=ROOT/'checkpoints/phase14_3_pi05_bc'
def main():
    lifecycle=json.loads((R/'lifecycle.json').read_text()) if (R/'lifecycle.json').exists() else {}
    training={}
    for f in C.glob('*_training.csv'):
        try:
            rows=list(csv.DictReader(f.open()));last=rows[-1] if rows else {};training[f.stem]=last
        except Exception:pass
    counts={}
    for split in ['validation','test']:
        files=[f for f in (R/split).rglob('*.json') if f.name not in ['manifest.json']]
        counts[split]=len(files)
    d={'utc':time.time(),'running':{k:v for k,v in lifecycle.items() if v['status']=='running'},
       'mse_seed_complete':len(list(C.glob('mse_seed*_complete.json'))),
       'pi05_models_complete':sum((C/f'{arm}_seed{seed}_complete.json').exists() for arm in ['C','D'] for seed in range(5)),
       'pi05_target':10,'pilot_gate':(R/'pilot_gate.json').exists(),'training':training,'completed_tests':counts,
       'validation_target':700,'test_target':140,'complete':(R/'experiment_complete.json').exists(),'failed':(R/'pipeline_failed.txt').exists(),'rl_updates':0}
    print(json.dumps(d,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
