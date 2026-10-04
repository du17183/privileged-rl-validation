"""Artifact-based progress; an Isaac process returning0 is not completion."""
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_2_recovery_bc'
ARMS=['A','B','C','D','E','F','D_no_handle'];TESTS=['fixed','random','ee_offset','contact_loss','door_regression','stagnation','natural_severe']

def main():
    trained=len(list((ROOT/'checkpoints/phase14_2_recovery_bc').glob('*_best.pt')))
    found=0;models=0;bytest={t:0 for t in TESTS};random={};missing=[]
    for seed in range(5):
        for arm in ARMS:
            count=0
            for test in TESTS:
                file=R/'evaluation'/f'{arm}_seed{seed}'/(test+'.json')
                ok=file.exists()
                if ok:
                    d=json.loads(file.read_text());ok=test in ['fixed','random'] or d.get('restore_snapshot_after_warmup') is True
                if ok:
                    count+=1;found+=1;bytest[test]+=1
                    if test=='random':random[f'{arm}_seed{seed}']=d['success']
                else:missing.append(str(file.relative_to(ROOT)))
            models+=count==7
    final=len(list((R/'evaluation_final').glob('*/random.json')))
    summary=dict(utc=datetime.now(timezone.utc).isoformat(),trained=trained,trained_target=35,
        main_tests_complete=found,main_tests_target=245,models_fully_evaluated=models,models_target=35,
        final_sensitivity_complete=final,final_sensitivity_target=30,bytest=bytest,random_success=random,
        all_evaluation_complete=found==245 and final==30,missing=missing,rl_updates=0)
    (R/'progress.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='missing'},indent=2),flush=True)

if __name__=='__main__':main()
