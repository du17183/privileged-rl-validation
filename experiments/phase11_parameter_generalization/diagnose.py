"""Endpoint feature use and gradient balance; no online interaction."""
import json
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from safe_online.std_schedule import ControlledActor
from safe_online.kl_constraint import anchor_kl
from experiments.phase11_parameter_generalization.agent import MeasuredSAC
from experiments.phase11_parameter_generalization.data import MeasuredExpert
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, CKPT, ARMS, prepared
from environment_state.normalization import DIMS
from evaluation.parameter_sensitivity import sensitivity


def gradnorm(loss, params):
    gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(sum(g.square().sum() for g in gradients if g is not None).sqrt())


def main():
    torch.set_num_threads(4)
    protocol = prepared();results = []
    for arm in ARMS:
        for seed in range(5):
            run = f'P11{arm}_seed{seed}'
            initial = torch.load(ROOT/'checkpoints/phase9_safe_online'/f'P9C_seed{seed}'/'step_300000.pt', map_location='cpu', weights_only=False)
            anchor = torch.load(ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{seed}'/'best.pt', map_location='cpu', weights_only=False)
            with h5py.File(ROOT/'datasets'/OUT.name/f'{run}.h5', 'r') as h5:
                episodes = json.loads(h5['episodes_json'][()])
                ids = np.sort(np.random.default_rng(111000+seed).choice(len(h5['transitions/robot']), 256, replace=False))
                samples = {k: torch.as_tensor(h5['transitions'][k][ids], dtype=torch.float32) for k in h5['transitions']}
                reset_indices = sorted({int(r['start']) for r in episodes})[:128]
                reset_observations = torch.as_tensor(h5['transitions/robot'][reset_indices], dtype=torch.float32)
            expert = MeasuredExpert(ROOT/'door_dataset/door_expert_1000.h5', 'cpu', 1/60, arm, protocol['handle_center'])
            for endpoint, path in (('initial', None), ('best', CKPT/run/'best.pt'), ('final', CKPT/run/'step_300000.pt')):
                agent = MeasuredSAC(initial, anchor, arm, 'cpu')
                if path is not None: agent.load_state(torch.load(path, map_location='cpu', weights_only=False))
                torch.manual_seed(112000+seed)
                ex = expert.sample(256)
                obs = torch.cat((samples['robot'][:128], ex['robot'][:128]), 0)
                params = [p for p in agent.actor.parameters() if p.requires_grad]
                action, logp = agent.actor(obs)
                q1, q2 = agent.critic(obs, action)
                rl = (agent.log_alpha.exp().detach()*logp-torch.minimum(q1, q2)).mean()
                bc = F.mse_loss(agent.actor(ex['robot'], deterministic=True)[0], ex['action'])*10.
                kl = anchor_kl(agent.actor, agent.anchor, obs)
                norms = dict(rl=gradnorm(rl, params), weighted_bc=gradnorm(bc, params), weighted_kl=gradnorm(kl, params))
                results.append(dict(arm=arm, seed=seed, endpoint=endpoint,
                    online_state_probes=sensitivity(agent.actor, samples['robot'], arm),
                    reset_state_probes=sensitivity(agent.actor, reset_observations, arm),
                    gradient_norms=norms, loss_values=dict(rl=float(rl.detach()), weighted_bc=float(bc.detach()), weighted_kl=float(kl.detach())),
                    q_mean=float(torch.minimum(q1, q2).mean().detach()), q_variance=float(torch.minimum(q1, q2).var(unbiased=False).detach()),
                    added_actor_weight_norm=float(agent.actor.encoder[0].weight[:, 26:].norm().detach()),
                    observation_dim=DIMS[arm]))
    (OUT/'parameter_diagnosis.json').write_text(json.dumps(dict(offline_only=True, gradient_batch='128 recorded online +128 expert; separate identical BC expert batch',
        limitation='Gradient magnitudes are endpoint local diagnostics, not a causal anchor-weight ablation; no training hyperparameter was changed', results=results), indent=2))
    print('Saved 60 endpoint diagnoses', flush=True)
if __name__ == '__main__': main()
