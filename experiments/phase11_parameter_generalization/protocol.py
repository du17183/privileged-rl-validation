from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[2]
TAG = 'phase11_parameter_generalization'
OUT, LOG, CKPT = (ROOT/k/TAG for k in ('results', 'logs', 'checkpoints'))
ARMS = ('A', 'B', 'C', 'D')
CONFIG = dict(num_envs=32, batch_size=256, updates_per_vector_step=4, lr=3e-4,
              steps=300000, seeds=list(range(5)), expert_fraction=.5, bc_weight=10.,
              anchor_kl_weight=1., std_cap=.01, guard_interval=10000,
              source='Phase9 C final', anchor='Phase8 B best', task_success_angle=1.,
              normalization='radians; contacts binary; handle (workspace_position-expert_initial_mean)/0.1m',
              primary_evaluation='stochastic capped policy on fixed heldout Level2 distribution')
def prepared(): return json.loads((OUT/'protocol.json').read_text())
