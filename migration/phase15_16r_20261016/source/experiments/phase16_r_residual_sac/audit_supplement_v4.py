"""Audit added calendar evidence and byte-exact frozen gripper commands."""
import sys,json
from pathlib import Path
from datetime import datetime
sys.path.insert(0,str(Path.cwd()))
import h5py,numpy as np
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,save,sha,utc
from experiments.phase16_r_residual_sac.audit_supplement_v3 import main as prior

def main():
    assert sha(EXP/'audit_supplement_v3.py')=='2fd521cced1728318db52ae90609d1b1de404c9b6b65585b47f74add65a13785'
    prior()
    audit=RES/'independent_audit.json';old=json.loads(audit.read_text())
    folder=RES/'audit_revisions'/('calendar_'+utc().replace(':',''));folder.mkdir(parents=True,exist_ok=False)
    (folder/'prior_independent_audit.json').write_bytes(audit.read_bytes())
    issues=[];data=json.loads((RES/'calendar_timing.json').read_text())
    for rel,wanted in data['input_sha256'].items():
        if sha(ROOT/rel)!=wanted:issues.append('Calendar input SHA changed '+rel)
    for row in data['rows']:
        expected=(datetime.fromisoformat(row['end_utc'].replace('Z','+00:00'))-datetime.fromisoformat(row['start_utc'].replace('Z','+00:00'))).total_seconds()
        if expected!=row['calendar_wall_s'] or not row['not_active_compute_time']:issues.append('Calendar timing scope/calculation mismatch')
    count=0
    for namespace in ['baseline','train','evaluation']:
        for path in (RES/namespace).rglob('*.h5'):
            with h5py.File(path) as h:
                a=np.ascontiguousarray(h['base'][:,:,6]);b=np.ascontiguousarray(h['executed'][:,:,6])
                if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():issues.append('Gripper command bytes changed '+str(path.relative_to(ROOT)))
                count+=1
    post=RES/'postprocessing_source_v1.json';source=json.loads(post.read_text())
    for name in ['timing_completion_v1.py','report_supplement_v4.py','audit_supplement_v4.py']:source['files'][str((EXP/name).relative_to(ROOT))]=sha(EXP/name)
    save(post,source)
    old['supplementary_checks'].update(postprocessing_source_manifest_sha256=sha(post),gripper_byte_exact_closed_traces=count,calendar_timing_sha256=sha(RES/'calendar_timing.json'),calendar_intervals_nonadditive=True)
    old.update(utc=utc(),passed=not issues,issues=issues,auditor_source_sha256=sha(Path(__file__)))
    save(audit,old);print(json.dumps(dict(passed=not issues,issues=issues,gripper_byte_exact_closed_traces=count)))
    if issues:sys.exit(1)
if __name__=='__main__':main()
