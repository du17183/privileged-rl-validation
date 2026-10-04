"""Preserve active7 pilot/fits, stop only owned processes, require new pilot.

Correct the32-dimensional flow's known-zero padding target before any formal
pi05 performance selection. MSE models/cohorts/previous phases stay intact.
"""
import json,os,signal,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_3_pi05_bc';CP=ROOT/'checkpoints/phase14_3_pi05_bc';L=ROOT/'logs/phase14_3_pi05_bc'
def stop(pid,module):
    p=Path('/proc')/str(pid);argv=(p/'cmdline').read_bytes().split(bytes([0]));cwd=(p/'cwd').resolve()
    assert cwd==ROOT.resolve() and module.encode() in argv and b'-m' in argv,(pid,argv,cwd)
    os.kill(pid,signal.SIGTERM)
def main():
    assert not list((R/'validation').glob('[CD]*/*/*.json')),'Repair before formal pi05 evaluations only'
    assert not (R/'padding_repair.json').exists()
    lifecycle=json.loads((R/'lifecycle.json').read_text());supervisors=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=(p/'cmdline').read_bytes().split(bytes([0]))
            if b'experiments.phase14_3_pi05_bc.run' in argv and (p/'cwd').resolve()==ROOT.resolve():supervisors.append(int(p.name))
        except (PermissionError,FileNotFoundError,ProcessLookupError):pass
    assert len(supervisors)==1,supervisors
    stop(supervisors[0],'experiments.phase14_3_pi05_bc.run');stopped=[]
    for name,task in lifecycle.items():
        if name.startswith('pi05_') and task['status']=='running':
            stop(task['pid'],'pi05.train');stopped.append(task['pid']);task.update(status='archived_padding_v1',finished=time.time())
    time.sleep(3)
    for pid in stopped:
        p=Path('/proc')/str(pid)/'cmdline';assert not p.exists() or not p.read_bytes(),pid
    checkpoint_archive=CP/'active7_padding_v1';checkpoint_archive.mkdir()
    for p in CP.iterdir():
        if p.is_file() and p.name.startswith(('C_seed','D_seed')):p.rename(checkpoint_archive/p.name)
    if (CP/'pilot').exists():(CP/'pilot').rename(CP/'pilot_active7_v1')
    for name in ['pilot_evaluation','pilot_gate.json','preflight_b128.json']:
        p=R/name
        if p.exists():p.rename(R/(name.replace('.json','')+'_active7_v1'+('.json' if name.endswith('.json') else '')))
    source=R/'active7_padding_v1_source';source.mkdir()
    if (R/'pi05_train_active7_v1.py').exists():shutil.move(str(R/'pi05_train_active7_v1.py'),source/'train.py')
    for p in L.glob('pi05_[CD]_seed*.log'):p.rename(L/(p.stem+'_active7_v1.log'))
    (R/'lifecycle.json').write_text(json.dumps(lifecycle,indent=2))
    (R/'padding_repair.json').write_text(json.dumps(dict(reason='32dim sampler integrates all channels, old loss supervised7only; fix zero-padding flow targets',
       time=time.time(),stopped_own_training_pids=stopped,supervisor=supervisors[0],old_fits_excluded_from_formal_results=True,
       unchanged='same data,control,GT,environment,5seeds,5000formalupdates,batch128',mse_preserved=True),indent=2))
    print('PADDING REPAIR PRESERVED ALL OLD TRIALS',flush=True)
if __name__=='__main__':main()
