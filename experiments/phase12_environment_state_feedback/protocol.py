from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
TAG='phase12_environment_state_feedback'
OUT,LOG,CKPT=(ROOT/k/TAG for k in ('results','logs','checkpoints'))
ARMS=('A','B','C','D')
STRENGTHS={'strong':1.,'medium':.1,'weak':.01,'none':0.}
VARIANTS=[(a,'strong') for a in ARMS]+[(a,s) for a in ('B','D') for s in ('medium','weak','none')]
CONFIG=dict(num_envs=32,batch_size=256,updates_per_vector_step=4,lr=3e-4,
    steps=300000,seeds=list(range(5)),expert_fraction=.5,bc_weight=10.,std_cap=.01,
    guard_interval=10000,source='Phase9 C final',anchor='Phase8 B best',task_success_angle=1.,
    full_randomization_from_first_episode=True,curriculum_enabled=False,
    train_distribution='door initial U(0,5 deg); cabinet/handle rigid XYZ each U(-.01,.01 m); nominal friction',
    anchor_strengths=STRENGTHS,variants=VARIANTS,observation_dims={'A':26,'B':31,'C':33,'D':36},
    primary_evaluation='stochastic capped policy; fixed full Level2 heldout distribution',
    normalization='angle/target/remaining rad; velocity rad/s; progress [0,1]; contact2 binary; handle centered/0.1m',
    guard='Original SafeUpdate threshold .10, common nominal+full-Level2 deterministic and stochastic tests',
    primary_contrasts=['B_strong-A_strong','C_strong-B_strong','D_strong-C_strong','D_strong-A_strong'],
    masking='Unconditional best/final input removal. Individual masks plus coherent angle-family and all-feedback masks; evaluation intervention, not retraining.',
    inference='seed-level paired t95 CI; exact sign-flip; Holm within predefined families; n=5 minimum exact two-sided p=.0625',
    extension_gate='B/D versus A: paired final t95 lower>0, Holm paired t p<.05 across 8 B/D candidates, >=4/5 favorable, mean gain>=.05, abs mean best-final gap<=.05, rejection fraction<=.20. Gate is model-based exploratory, not exact-test confirmation.',
    extension_budget='Only qualifying B/D and matched A: continue from300k to500k, no new architecture/data/reward.')
def prepared():return json.loads((OUT/'protocol.json').read_text())
def run_name(arm,strength,seed):return f'P12{arm}_{strength}_seed{seed}'
