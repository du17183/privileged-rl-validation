"""Conservative action-averaging diagnosis; near states are not identical states."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/phase14_3_pi05_bc'

def prepare():
    data=ROOT/'datasets/phase14_2/prepared_v1';m=json.loads((data/'manifest.json').read_text())
    with np.load(data/'recovery_train.npz') as h:
        raw=np.concatenate((h['robot'],h['environment']),-1);labels=h['action'].copy()
    with np.load(data/'recovery_validation.npz') as h:
        query=np.concatenate((h['robot'],h['environment']),-1);query_actions=h['action'].copy()
    rng=np.random.default_rng(143910);trainidx=rng.choice(len(raw),min(12000,len(raw)),replace=False);validx=rng.choice(len(query),min(2000,len(query)),replace=False)
    mean=np.asarray(m['mean']);std=np.asarray(m['std']);tree=cKDTree((raw[trainidx]-mean)/std)
    distances,neighbors=tree.query((query[validx]-mean)/std,k=16,workers=4)
    candidates=[];qualified=[]
    for j,vi in enumerate(validx):
        indices=trainidx[neighbors[j]];disagreement=np.linalg.norm(labels[indices]-query_actions[vi],axis=-1)/np.sqrt(7)
        for k in np.argsort(-disagreement)[:2]:
            ti=indices[k];x,y=query[vi],raw[ti];a,b=query_actions[vi],labels[ti]
            movement=float(np.dot(a[:3],b[:3])/(np.linalg.norm(a[:3])*np.linalg.norm(b[:3])+1e-12))
            metrics=dict(state_rms=float(distances[j,k]/np.sqrt(39)),action_rms=float(disagreement[k]),
                eef_distance_m=float(np.linalg.norm(x[18:21]-y[18:21])),handle_distance_m=float(np.linalg.norm(x[32:35]-y[32:35])),
                joint_rms_rad=float(np.sqrt(np.mean((x[:9]-y[:9])**2))),velocity_rms=float(np.sqrt(np.mean((x[9:18]-y[9:18])**2))),
                door_difference_rad=float(abs(x[26]-y[26])),contact_equal=bool(np.array_equal(x[30:32]>.5,y[30:32]>.5)),position_action_cosine=movement)
            ok=metrics['state_rms']<=.3 and metrics['action_rms']>=.25 and metrics['eef_distance_m']<=.015 and metrics['handle_distance_m']<=.015 and metrics['joint_rms_rad']<=.08 and metrics['velocity_rms']<=.3 and metrics['door_difference_rad']<=.05 and metrics['contact_equal'] and movement<-.5 and min(np.linalg.norm(a[:3]),np.linalg.norm(b[:3]))>.05
            item={'validation_index':int(vi),'training_neighbor_index':int(ti),'qualified':bool(ok),**metrics}
            candidates.append(item)
            if ok:qualified.append(item)
    # Sample across candidate states, not repeatedly select the same trajectory tick.
    pool=sorted(qualified or candidates,key=lambda v:-(v['action_rms']/(.01+v['state_rms'])))
    chosen=[];seen=set()
    for row in pool:
        if row['validation_index'] in seen:continue
        seen.add(row['validation_index']);chosen.append(row)
        if len(chosen)==64:break
    vi=np.array([v['validation_index'] for v in chosen]);ti=np.array([v['training_neighbor_index'] for v in chosen])
    R.mkdir(exist_ok=True);np.savez(R/'action_pair_bank.npz',raw=query[vi],neighbor_raw=raw[ti],expert=query_actions[vi],neighbor_expert=labels[ti])
    d={'queried_validation_states':len(validx),'sampled_training_states':len(trainidx),'qualified_opposed_direction_pairs':len(qualified),'bank':chosen,
       'interpretation':'Opposite labels at approximate neighbors are evidence for diagnosis, not proof both commands solve exactly the same physical state. No labeled left/right routes are invented.'}
    (R/'action_pair_bank.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k!='bank'}),flush=True)

if __name__=='__main__':prepare()
