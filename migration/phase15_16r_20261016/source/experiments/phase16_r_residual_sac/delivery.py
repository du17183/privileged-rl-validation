"""Archive a completed reviewed payload and every file SHA; no historic writes."""
import sys,json,tarfile,os
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,sha,save,utc,protocol

def main():
    protocol();audit=json.loads((RES/'independent_audit.json').read_text());assert audit['passed'],audit['issues']
    report=ROOT/'docs/phase16_r_report.md';assert sha(report)==audit['report_sha256']
    visual=json.loads((RES/'visual_review.json').read_text());assert visual['passed']
    for relative,wanted in visual['files'].items():assert sha(ROOT/relative)==wanted
    out=RES/'delivery';out.mkdir(parents=True,exist_ok=True)
    if (out/'phase16_r_delivery.tar.gz').exists():raise RuntimeError('Archive already exists; preserve original delivery before revision')
    status=json.loads((RES/'report_status.json').read_text());status.update(audit_pending=False,independent_audit_sha256=sha(RES/'independent_audit.json'),visual_review_sha256=sha(RES/'visual_review.json'));save(RES/'report_status.json',status)
    files={}
    for root in [EXP,ROOT/'algorithms/residual_sac',ROOT/'evaluation/phase16_r',RES,ROOT/'checkpoints/phase16_r',ROOT/'logs/phase16_r']:
        for p in root.rglob('*'):
            if p.is_file() and not p.is_socket() and 'delivery' not in p.parts and '__pycache__' not in p.parts and p.suffix not in ['.tmp','.sock','.pyc'] and p.name not in ['delivery_manifest.json','local_sha_verified.json','final_delivery_complete.json']:
                files[str(p.relative_to(ROOT))]=p
    files[str(report.relative_to(ROOT))]=report
    hist=json.loads((RES/'historical_inputs.json').read_text())
    for seed,row in hist['normal'].items():
        p=ROOT/row['checkpoint'];assert sha(p)==row['sha256'];files[f'references/Normal_seed{seed}_{p.name}']=p
    manifest=dict(utc=utc(),files={rel:dict(sha256=sha(p),bytes=p.stat().st_size) for rel,p in sorted(files.items())},payload_files=len(files),report_sha256=sha(report),independent_audit_sha256=sha(RES/'independent_audit.json'),visual_review_sha256=sha(RES/'visual_review.json'),excluded='archive output, transient Python/socket/tmp files and manifest/receipt self-reference; all saved scientific source/checkpoint/replay/CSV/H5/log/provenance included',local_verification_pending=True,base_model_reference=hist['checks'].get('base_sha256'))
    save(RES/'delivery_manifest.json',manifest)
    dest=out/'phase16_r_delivery.tar.gz'
    with tarfile.open(dest,'w:gz',compresslevel=1) as archive:
        for rel,p in sorted(files.items()):archive.add(p,arcname=rel,recursive=False)
        archive.add(RES/'delivery_manifest.json',arcname='results/phase16_r/delivery_manifest.json')
    save(out/'archive.json',dict(utc=utc(),sha256=sha(dest),bytes=dest.stat().st_size,payload_files=len(files),delivery_manifest_sha256=sha(RES/'delivery_manifest.json'),report_sha256=sha(report),audit_passed=True,local_delivery_complete=False))
    print(json.dumps(json.loads((out/'archive.json').read_text())))
if __name__=='__main__':main()
