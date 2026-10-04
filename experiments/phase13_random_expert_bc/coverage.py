"""Offline errors by planner phase and distance of rollout traces to data.

Nearest-neighbor distances are diagnostics, not density estimates or quality
weights. Nothing from this analysis changes training or replay.
"""
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial import cKDTree
import torch
from experiments.phase13_random_expert_bc.model import load_checkpoint,inputs

ROOT=Path(__file__).resolve().parents[2]


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    folder=ROOT/'datasets/random_door_expert/split_v1'
    manifest=json.loads((folder/'split.json').read_text())
    train=dict(np.load(folder/'train.npz'));test=dict(np.load(folder/'test.npz'))
    phases=[]
    with h5py.File(ROOT/'datasets/random_door_expert/collection_v1/trajectories.h5','r') as h:
        for key in manifest['splits']['test']:phases.append(h[key]['planner_phase'][:,0])
    phases=np.concatenate(phases)
    results=[]
    for seed in range(5):
        for arm in ['A','B']:
            checkpoint=ROOT/f'checkpoints/phase13_random_expert_bc/{arm}_seed{seed}_best.pt'
            policy,mean,std,_=load_checkpoint(checkpoint,'cpu')
            x=inputs(torch.tensor(test['robot']),torch.tensor(test['environment']),mean,std,arm)
            predictions=torch.cat([policy(x[lo:lo+4096]) for lo in range(0,len(x),4096)]).numpy()
            errors=(predictions-test['action'])**2
            by_phase={}
            for phase in range(7):
                mask=phases==phase
                if mask.any():
                    by_phase[str(phase)]=dict(transitions=int(mask.sum()),action_mse=float(errors[mask].mean()),
                        arm_mse=float(errors[mask,:6].mean()),gripper_mse=float(errors[mask,6].mean()),
                        gripper_sign_error=float(np.mean(np.sign(predictions[mask,6])!=np.sign(test['action'][mask,6]))))
            train_x=inputs(torch.tensor(train['robot']),torch.tensor(train['environment']),mean,std,arm).numpy()
            dims=26 if arm=='A' else 39
            tree=cKDTree(train_x[:,:dims])
            test_dist,_=tree.query(x.numpy()[::20,:dims],workers=4)
            trace=ROOT/f'results/phase13_random_expert_bc/heldout/{arm}_seed{seed}_best.trace.npz'
            diagnostics=None
            if trace.exists():
                t=dict(np.load(trace));tx=inputs(torch.tensor(t['robot']),torch.tensor(t['environment']),mean,std,arm).numpy()
                distance,idx=tree.query(tx[:,:dims],workers=4)
                disagreement=((t['action']-train['action'][idx])**2).mean(-1)
                support=np.quantile(test_dist,.95)
                # Trigger-style descriptive flags; no causal claim from one trace.
                contacts=(t['environment'][:,4:6]>.5).all(-1)
                pulling=(np.linalg.norm(t['action'][:,:3],axis=-1)>.2)&(t['action'][:,6]<0)&~contacts
                diagnostics=dict(trace_steps=len(distance),test_distance_median=float(np.median(test_dist)),
                    test_distance_p95=float(support),rollout_distance_median=float(np.median(distance)),
                    rollout_distance_p95=float(np.quantile(distance,.95)),
                    rollout_fraction_beyond_test_p95=float(np.mean(distance>support)),
                    action_disagreement_with_nearest_expert_mse=float(disagreement.mean()),
                    closed_gripper_translation_without_bilateral_contact_fraction=float(pulling.mean()),
                    trace_success=bool(t['next_state'][-1,0]>1),final_angle=float(t['next_state'][-1,0]))
            results.append(dict(seed=seed,arm=arm,phase_errors=by_phase,first_episode_trace=diagnostics))
    value=dict(rows=results,phase_names=['REST','CLEAR','APPROACH','ALIGN','GRASP','OPEN','HOLD'],
        interpretation='Nearest-neighbor distances are in each arm normalized input, not comparable across dimensions. Trace sample is first episode of clone0, selected before outcomes, not a population success estimate. Closed-gripper translation without contact is descriptive, not proof of the intended motion.')
    path=ROOT/'results/phase13_random_expert_bc/coverage_diagnosis.json';path.write_text(json.dumps(value,indent=2))
    print(json.dumps([dict(seed=r['seed'],arm=r['arm'],trace=r['first_episode_trace']) for r in results]),flush=True)


if __name__=='__main__':main()
