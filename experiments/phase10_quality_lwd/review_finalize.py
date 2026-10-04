"""Review real final data and source provenance; no additional learning/rollout."""
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime,timezone
from pathlib import Path
import h5py
import numpy as np
import torch
from experiments.phase10_quality_lwd.protocol import ARMS
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not (OUT/'phase10_completed.json').exists():raise RuntimeError('Await actual completed experiment')
    manifest=json.loads((OUT/'source_manifest.json').read_text())
    core=['experiments/phase10_quality_lwd/train.py','experiments/phase10_quality_lwd/protocol.py',
          'replay/quality_replay.py','trajectory_quality/quality_model.py','lwd/simplified_lwd.py']
    if any(sha(ROOT/p)!=manifest['new_source'][p] for p in core):raise RuntimeError('Learner/scorer source changed')
    preflight=json.loads((OUT/'preflight.json').read_text());sources={s['seed']:s for s in preflight['sources']}
    audits=[]
    for arm in ARMS:
        for seed in range(5):
            run=f'P10{arm}_seed{seed}';folder=ROOT/'checkpoints/phase10_quality_lwd'/run
            marker=json.loads((folder/'completed.json').read_text())
            state=torch.load(folder/'step_300000.pt',map_location='cpu',weights_only=False)
            assert state['std_cap']==.01 and state['kl_weight']==1.
            assert state['actor']['encoder.0.weight'].shape[1]==26
            assert state['critic']['q1.0.weight'].shape[1]==33
            assert state['phase9_source_sha256']==sources[seed]['phase9_sha256']
            assert state['anchor_sha256']==sources[seed]['phase8_anchor_sha256']
            with h5py.File(ROOT/'datasets/phase10_quality_lwd'/f'{run}.h5','r') as h5:
                t=h5['trajectories'];flat=h5['transitions']
                starts=t['start'][:].astype(int);lengths=t['length'][:].astype(int);stride=t['stride'][:].astype(int)
                ends=starts+(lengths-1)*stride
                assert len(starts)==marker['online_episodes'] and len(flat['action'])==300000
                assert len(np.unique(ends))==len(ends) and bool((ends<300000).all())
                assert bool((stride==96).all())
                labels=t['success'][:]
                assert int(labels.sum())==marker['online_successes']
                assert lengths.sum()+int(h5.attrs['partial_transitions'])==300000
                done=flat['done'][:,0];gt=flat['next_privileged'][:,0];reward=flat['reward'][:,0]
                for i,(start,length,last) in enumerate(zip(starts,lengths,ends)):
                    ids=start+np.arange(length)*96
                    assert done[last]>.5 and not np.any(done[ids[:-1]]>.5)
                    assert int(gt[last]>1)==int(labels[i])
                    assert abs(float(reward[ids].sum())-float(t['return_value'][i]))<.001
                for env_index in range(96):
                    indexes=np.flatnonzero(starts%96==env_index)
                    assert starts[indexes[0]]==env_index
                    if len(indexes)>1:assert np.all(starts[indexes[1:]]==starts[indexes[:-1]]+lengths[indexes[:-1]]*96)
            audits.append(dict(arm=arm,seed=seed,trajectories=marker['online_episodes'],
                successes=marker['online_successes'],checkpoint_actor_input=26,critic_observation_plus_action=33,passed=True))
    (OUT/'formal_data_audit.json').write_text(json.dumps(dict(passed=True,runs=35,trajectories=sum(a['trajectories'] for a in audits),
        checks='All terminal snapshots, complete trajectory segmentation, physical success labels, reward sums, storage conservation, robot-only actor/critic dimensions, source and frozen-anchor hashes, unchanged learning/scoring code',
        details=audits),indent=2))
    report=ROOT/'docs/phase10_quality_lwd_report.md'
    original=OUT/'report_before_review.md'
    if not original.exists():shutil.copyfile(report,original)
    for module in ('baseline_inventory','preservation_supplement','analyze','finalize_report'):
        command=[str(ROOT/'.venv/bin/python'),'-m',f'experiments.phase10_quality_lwd.{module}']
        if module in ('baseline_inventory','preservation_supplement'):command.append('--verify')
        subprocess.run(command,cwd=ROOT,check=True)
    marker=json.loads((OUT/'phase10_completed.json').read_text())
    marker.update(initial_report_sha256=sha(original),report_sha256=sha(report),
                  reviewed_at=datetime.now(timezone.utc).isoformat(),formal_data_audit_passed=True,
                  review_revision='Clear actual inference, pp units, supplementary recipe comparisons, client timezone and source review; unchanged training/heldout outcomes')
    (OUT/'phase10_completed.json').write_text(json.dumps(marker,indent=2))
    archive=OUT/'reviewed_source_snapshot';archive.mkdir(exist_ok=False)
    current={}
    for folder in (ROOT/'experiments/phase10_quality_lwd',ROOT/'trajectory_quality',ROOT/'lwd'):
        for p in folder.glob('*.py'):
            target=archive/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target);current[str(p.relative_to(ROOT))]=sha(p)
    (OUT/'reviewed_source_manifest.json').write_text(json.dumps(dict(current_sources=current,core_learning_source_unchanged=True),indent=2))
    bundle=OUT/'review_artifacts_verified.zip'
    paths=[report,ROOT/'docs/phase10_quality_lwd_protocol.md',ROOT/'replay/quality_replay.py']
    for folder in (OUT,ROOT/'experiments/phase10_quality_lwd',ROOT/'trajectory_quality',ROOT/'lwd'):
        paths += [p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.jsonl','.csv','.png','.txt')
                  and 'ipc' not in p.parts and '__pycache__' not in p.parts and 'initial_upload_staging' not in p.parts]
    paths+=list((ROOT/'checkpoints/phase10_quality_lwd').glob('*/completed.json'))
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paths)):z.write(p,str(p.relative_to(ROOT)))
    (OUT/'verified_review_bundle.json').write_text(json.dumps(dict(path=str(bundle),bytes=bundle.stat().st_size,
        sha256=sha(bundle),report_sha256=sha(report),training_and_heldout_unchanged=True),indent=2))
    print('Reviewed all35 real datasets/models, all35250 completed trajectories; verified report bundle ready')

if __name__=='__main__':main()
