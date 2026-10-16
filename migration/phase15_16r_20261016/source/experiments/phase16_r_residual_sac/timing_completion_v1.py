"""Calendar intervals from saved dispatch/completion timestamps, not summed timers."""
import sys,json
from pathlib import Path
from datetime import datetime
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,RES,save,sha,utc

def main():
    inputs={}
    def stamp(path):
        x=json.loads(path.read_text());inputs[str(path.relative_to(ROOT))]=sha(path)
        return x['utc']
    def interval(label,start,end):
        seconds=(datetime.fromisoformat(end.replace('Z','+00:00'))-datetime.fromisoformat(start.replace('Z','+00:00'))).total_seconds()
        assert seconds>=0
        return dict(scope=label,start_utc=start,end_utc=end,calendar_wall_s=seconds,includes_queue_and_other_project_contention=True,not_active_compute_time=True)
    marker=RES/'first_round_complete.json'
    if not marker.exists():marker=RES/'first_round_execution_finished.json'
    finished=RES/'training_handoff_v5/complete.json'
    assert marker.exists() and finished.exists(),'Only after first round and original report/audit complete'
    end=stamp(finished);rows=[]
    rows.append(interval('saved historical verification to original report/audit completion; preparation and technical attempts included after this timestamp',stamp(RES/'historical_inputs.json'),end))
    rows.append(interval('passed physical engineering dispatch to completion',stamp(RES/'engineering/lifecycle.json'),stamp(RES/'engineering/complete.json')))
    starts=[stamp(RES/f'baseline/seed{i}/lifecycle.json') for i in range(5)]
    rows.append(interval('five-seed new frozen baseline calendar interval, concurrent jobs',min(starts),stamp(RES/'Frozen_pi05_baseline.json')))
    starts=[stamp(RES/f'train/{arm}_seed{i}/lifecycle.json') for arm in ['C','D'] for i in range(3)]
    rows.append(interval('first native online training dispatch through final and validation-selected best tests',min(starts),stamp(marker)))
    save(RES/'calendar_timing.json',dict(utc=utc(),rows=rows,input_sha256=inputs,physical_interactions_added=0,source_sha256=sha(Path(__file__)),limitation='Saved UTC timestamps have one-second resolution. Calendar intervals overlap and must not be added. No claim about human work before the first saved historical verification. Active training, update, physics and evaluation pause timers are reported separately.'))
    print(json.dumps(dict(rows=rows,physical_interactions_added=0)))
if __name__=='__main__':main()
