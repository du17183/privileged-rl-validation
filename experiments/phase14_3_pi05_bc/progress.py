"""Compact read-only progress, with real process liveness and failures."""
import csv,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_3_pi05_bc';C=ROOT/'checkpoints/phase14_3_pi05_bc'
def main():
    fit={p.stem:list(csv.DictReader(p.open()))[-1]['update'] for p in C.glob('[CD]*training.csv') if len(p.read_text().splitlines())>1}
    live=[]
    if (R/'lifecycle.json').exists():
        for name,d in json.loads((R/'lifecycle.json').read_text()).items():
            path=Path('/proc')/str(d['pid'])/'cmdline'
            if d['status']=='running' and path.exists() and path.read_bytes():live.append(name)
    failures={p.name:p.read_text()[-1500:] for p in [R/'pipeline_failed.txt',R/'finalize_failed.txt',R/'overlap_validation_failed.txt'] if p.exists()}
    d=dict(utc=time.time(),training=fit,pi05_complete=len(list(C.glob('[CD]*complete.json'))),
        validation_tests=len(list((R/'validation').glob('*/*/*.json'))),validation_target=700,
        formal_test=len(list((R/'test').glob('*/*.json'))),test_target=140,
        gt_ablation=len(list((R/'gt_ablation').glob('*/*.json'))),diagnostic_seeds=len(list((R/'action_diagnosis').glob('seed*.json'))),
        live=live,failures=failures,formal_complete=(R/'experiment_complete.json').exists(),delivery_complete=(R/'delivery_complete.json').exists())
    print(json.dumps(d),flush=True)
if __name__=='__main__':main()
