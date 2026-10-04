"""Review bundle: selected models, all measurements, exact used data/source.

All candidate checkpoints remain on server and receive a SHA256 index. The
14GB pretrained dependency is referenced by hash, not redundantly copied.
"""
import hashlib,json,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_3_pi05_bc'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    assert (R/'delivery_complete.json').exists(),'All experiments and diagnostics must finish first'
    files=set()
    for directory in ['pi05','experiments/phase14_3_pi05_bc','datasets/phase14_3/pi05_chunks','results/phase14_3_pi05_bc','logs/phase14_3_pi05_bc']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix not in ['.sock','.lock','.tmp'] and 'artifact_' not in p.name:files.add(p)
    for rel in ['evaluation/pi05_closed_loop_eval.py','evaluation/multimodal_action_analysis.py','evaluation/pi05_action_diagnosis.py',
        'evaluation/mse_checkpoint_diagnosis.py','evaluation/recovery_policy_eval.py','docs/phase14_3_pi05_bc_report.md']:
        files.add(ROOT/rel)
    for directory in ['door_env','randomized_env','environment_state','recovery_expert','experiments/phase13_random_expert_bc']:
        files.update(p for p in (ROOT/directory).rglob('*.py') if '__pycache__' not in p.parts)
    for rel in ['experiments/phase14_2_recovery_bc/model.py','experiments/phase14_2_recovery_bc/analyze.py','configs/runtime_env.sh']:
        files.add(ROOT/rel)
    data=ROOT/'datasets/phase14_2/prepared_v1';files.add(data/'manifest.json')
    files.update(data/f'{source}_{split}.npz' for source in ['base','recovery'] for split in ['train','validation','test'])
    cp=ROOT/'checkpoints/phase14_3_pi05_bc';files.update(cp.glob('*.json'));files.update(cp.glob('*.csv'))
    selection=json.loads((R/'selection.json').read_text())
    for name,d in selection.items():files.add(cp/f'{name}_step{d["step"]}.pt')
    index={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in cp.glob('*_step*.pt')}
    (R/'candidate_checkpoint_manifest.json').write_text(json.dumps(index,indent=2));files.add(R/'candidate_checkpoint_manifest.json')
    manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)}
    info={'files':manifest,'file_count':len(manifest),'created':time.time(),
        'scope':'All metrics/logs/derived chunks/used prepared S/R splits and20validation-selected weights; all100candidate weights indexed and retained on server.',
        'external_pretrained_dependency':json.loads((R/'pi05_final_provenance.json').read_text())['pretrained_base']}
    path=R/'artifact_manifest.json';path.write_text(json.dumps(info,indent=2));files.add(path)
    archive=ROOT/'results/phase14_3_review_bundle.tar.gz'
    assert not archive.exists(),'Do not overwrite an existing delivery bundle'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(files):tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    digest=sha(archive);(R/'artifact_bundle.json').write_text(json.dumps(dict(path=str(archive),sha256=digest,bytes=archive.stat().st_size,files=len(files)),indent=2))
    print(json.dumps(dict(archive=str(archive),sha256=digest,bytes=archive.stat().st_size,files=len(files))),flush=True)
if __name__=='__main__':main()
