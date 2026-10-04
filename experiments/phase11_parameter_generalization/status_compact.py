import contextlib
import io
import json
from experiments.phase11_parameter_generalization.progress import main
from experiments.phase11_parameter_generalization.protocol import OUT, ARMS
with contextlib.redirect_stdout(io.StringIO()):main()
data=json.loads((OUT/'current_progress.json').read_text())
groups={}
for arm in ARMS:
    rows=[r for r in data['runs'] if r['arm']==arm]
    groups[arm]=dict(steps_min=min(r['steps'] for r in rows),steps_max=max(r['steps'] for r in rows),
        completed=sum(r['status']=='trained' for r in rows),
        random_success_latest_mean=sum(r.get('random_success',0) for r in rows)/5,
        rejections=sum(r.get('rejections',0) for r in rows),
        levels=[r.get('level',0) for r in rows],
        online_successes=sum(r.get('online_successes',0) for r in rows),online_episodes=sum(r.get('online_episodes',0) for r in rows))
print(json.dumps(dict(timestamp=data['timestamp'],completed_runs=data['completed_runs'],training_steps=data['recorded_training_steps'],
    progress_pct=100*data['recorded_training_steps']/data['budget'],heldout_completed=data['heldout_jobs_completed'],errors=data['errors'],groups=groups),indent=2))
