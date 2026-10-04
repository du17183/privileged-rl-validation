"""User-specified expert qualification; this module never starts training."""
import argparse,json
from pathlib import Path


def decision(summary):
    n=int(summary['recovery_attempts']);k=int(summary['recovery_successes'])
    if not 0<=k<=n or not n:raise ValueError('Invalid recovery counts')
    rate=k/n
    if abs(rate-float(summary['recovery_success_rate']))>1e-9:raise ValueError('Summary rate/count mismatch')
    ready=n>=128 and rate>.90 and summary.get('takeover_resets',0)==0 and summary.get('robot_or_door_teleports',0)==0
    return dict(ready=ready,attempts=n,successes=k,success_rate=rate,required_rate='strictly >0.90',
        reason='Qualified for BC data preparation' if ready else 'Recovery expert not qualified; BC and RL remain blocked by the requested experimental gate')


def require(summary):
    value=decision(summary)
    if not value['ready']:raise RuntimeError(value['reason'])
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output')
    a=p.parse_args();value=decision(json.loads(Path(a.summary).read_text()))
    if a.output:Path(a.output).write_text(json.dumps(value,indent=2))
    print(json.dumps(value,indent=2))
