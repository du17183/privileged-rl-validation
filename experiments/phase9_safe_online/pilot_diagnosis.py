import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'results/phase9_safe_online/heldout_pilot/C_seed0.json'
for row in json.loads(path.read_text()):
    print(row['condition'],row['mode'],'success',row['success'],'requested_angle',row['requested_angle_deg'],
          'observed_angle',round(row['observed_angle_rad'],5),'requested_dy',row['requested_handle_dy'],
          'observed_handle_xyz',row['observed_handle_xyz'])
