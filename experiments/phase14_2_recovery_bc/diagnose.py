"""Offline diagnostics; no architecture/loss changes and no new policy training."""
import json
from pathlib import Path
import h5py,numpy as np,torch
from scipy.spatial import cKDTree
from .model import load,inputs
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_2_recovery_bc';P=ROOT/'datasets/phase14_2/prepared_v1'
PHASES={0:'release',1:'clear',2:'approach',3:'align',4:'grasp',5:'open',6:'hold',9:'orient'}

def main():
    torch.set_num_threads(4);m=json.loads((P/'manifest.json').read_text());result={}
    train={s:dict(np.load(P/f'{s}_train.npz')) for s in ['base','recovery','ordinary']}
    result['sampling']={a:dict(total_examples=10240000,S_examples=10240000 if a in ['A','B'] else 5120000,
          added_examples=0 if a in ['A','B'] else 5120000,
          S_average_reuse=(10240000 if a in ['A','B'] else 5120000)/len(train['base']['action']),
          added_average_reuse=5120000/len(train['recovery' if a in ['C','D','D_no_handle'] else 'ordinary']['action']) if a not in ['A','B'] else 0)
          for a in ['A','B','C','D','E','F','D_no_handle']}
    phases=[];cases=[]
    for key in m['splits']['recovery']['test']:
        case,name=key.split('/')
        with h5py.File(ROOT/'datasets/recovery_expert/phase14_1/formal_v1'/case/'trajectories.h5','r') as h:
            v=h[name]['planner_phase'][:].flatten();phases.extend(v);cases.extend([case]*len(v))
    phases=np.asarray(phases);cases=np.asarray(cases)
    d=dict(np.load(P/'recovery_test.npz'));device='cuda:0' if torch.cuda.is_available() else 'cpu'
    r=torch.tensor(d['robot'],device=device);e=torch.tensor(d['environment'],device=device)
    metrics=[]
    for seed in range(5):
        net,mean,std,meta=load(ROOT/f'checkpoints/phase14_2_recovery_bc/D_seed{seed}_best.pt',device)
        with torch.no_grad():pred=torch.cat([net(inputs(r[k:k+4096],e[k:k+4096],mean,std,meta['mode'])) for k in range(0,len(r),4096)]).cpu().numpy()
        errors=(pred-d['action'])**2
        for phase in np.unique(phases):
            mask=phases==phase
            metrics.append(dict(seed=seed,phase=PHASES.get(int(phase),str(phase)),transitions=int(mask.sum()),
                mse=float(errors[mask].mean()),xyz_mse=float(errors[mask,:3].mean()),rotation_mse=float(errors[mask,3:6].mean()),
                gripper_mse=float(errors[mask,6].mean()),gripper_sign_accuracy=float(np.mean(np.sign(pred[mask,6])==np.sign(d['action'][mask,6]))),
                near_zero_gripper_prediction=float(np.mean(np.abs(pred[mask,6])<.2))))
    result['D_recovery_test_phase_errors']=metrics
    counts={}
    for case in ['ee_offset','contact_loss','door_regression','stagnation']:
        with h5py.File(ROOT/'datasets/recovery_expert/phase14_1/formal_v1'/case/'trajectories.h5','r') as h:
            for key in m['splits']['recovery']['train']:
                tag,name=key.split('/')
                if tag!=case:continue
                phase=h[name]['planner_phase'][:].flatten()
                for value,n in zip(*np.unique(phase,return_counts=True)):
                    label=PHASES.get(int(value),str(value));counts[label]=counts.get(label,0)+int(n)
    total=sum(counts.values())
    result['R_train_phase_exposure']={k:dict(transitions=v,R_fraction=v/total,whole_batch_fraction=.5*v/total,expected_samples_per512=256*v/total) for k,v in counts.items()}
    # Coarse matched-state label diagnostic. Approximate neighbours are not
    # evidence of an exact contradiction or a causal attribution.
    mean=np.array(m['mean']);std=np.array(m['std']);rng=np.random.default_rng(142901)
    sampled={}
    for s,t in train.items():
        ids=rng.choice(len(t['action']),min(12000,len(t['action'])),replace=False)
        sampled[s]=dict(x=(np.concatenate((t['robot'][ids],t['environment'][ids]),-1)-mean)/std,
                        y=t['action'][ids],traj=t['trajectory_index'][ids])
    disagreement=[]
    for representation,dim in [('robot',26),('fullGT',39)]:
        query=sampled['recovery'];ids=rng.choice(len(query['x']),512,replace=False)
        for target in ['base','recovery','ordinary']:
            t=sampled[target];tree=cKDTree(t['x'][:,:dim]);dist,near=tree.query(query['x'][ids,:dim],k=8)
            chosen=[];chosen_dist=[];source_ids=[]
            for j,row in enumerate(near):
                eligible=[k for k in range(8) if target!='recovery' or t['traj'][row[k]]!=query['traj'][ids[j]]]
                if not eligible:continue
                k=eligible[0];chosen.append(row[k]);chosen_dist.append(dist[j,k]);source_ids.append(ids[j])
            errors=(query['y'][source_ids]-t['y'][chosen])**2
            disagreement.append(dict(representation=representation,R_nearest_source=target,n=len(chosen),
                normalized_state_distance_median=float(np.median(chosen_dist)),action_label_mse=float(errors.mean()),
                gripper_sign_disagreement=float(np.mean(np.sign(query['y'][source_ids,6])!=np.sign(t['y'][chosen,6])))))
    result['approximate_neighbour_label_disagreement']=disagreement
    result['limits']='Nearest states are approximate, different expert/controller labels and hidden planner phases can coexist. This is hypothesis generation, not proof of multimodal policy necessity.'
    result['next_if_no_gain']=['Do not blindly add more recovery trajectories','Inspect approach/grasp command errors and expert stage/elapsed-time ambiguity',
        'Consider short history, relative EEF-handle geometry, stage supervision in a separately preregistered future study; keep this main experiment unchanged','No RL until a validated anchor exists']
    (R/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(sampling=result['sampling'],nearest=disagreement)),flush=True)

if __name__=='__main__':main()
