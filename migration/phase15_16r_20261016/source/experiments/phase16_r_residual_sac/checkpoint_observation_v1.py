"""CPU-only checkpoint observation on lexicographic stored engineering states."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch,h5py,numpy as np
from algorithms.residual_sac.sac import SAC
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,save,sha,utc,protocol,checkpoint

def main():
    P=protocol();torch.set_num_threads(4);source=RES/'engineering/A.h5'
    meta=torch.load(checkpoint(0),map_location='cpu',weights_only=False)
    mean=np.asarray(meta['mean'],np.float32);std=np.asarray(meta['std'],np.float32)
    with h5py.File(source) as h:
        raw=h['raw'][:16].reshape(-1,39);base=h['base'][:16].reshape(-1,7);chunk=h['chunk'][:16].reshape(-1,70);cur=h['cursor'][:16].reshape(-1)
    context=np.concatenate((2/np.pi*np.arctan((raw-mean)/std),base,chunk,np.eye(5,dtype=np.float32)[cur]),-1).astype(np.float32)
    observations=[]
    initial=torch.load(ROOT/'checkpoints/phase16_r/C_seed0/initial.pt',map_location='cpu',weights_only=False)
    for arm,step in [('C',0),('C',10000),('D',10000)]:
        path=ROOT/f'checkpoints/phase16_r/{arm}_seed0/step{step}.pt';state=torch.load(path,map_location='cpu',weights_only=False)
        s=SAC(0,P['residual']['small_bound'],'cpu',P['sac']);s.restore(state)
        c=torch.as_tensor(context);b=torch.as_tensor(base);row=dict(arm=arm,seed=0,step=step,checkpoint_sha256=sha(path),updates=s.updates,alpha=float(s.log_alpha.exp()),actor_tensor_change_L2=float(sum((v-initial['actor'][k]).square().sum() for k,v in state['actor'].items()).sqrt()),critic_tensor_change_L2=float(sum((v-initial['q'][k]).square().sum() for k,v in state['q'].items()).sqrt()))
        with torch.no_grad():
            for deterministic in [False,True]:
                torch.manual_seed(161888);policy=s.policy(c,b,deterministic=deterministic);q1,q2=s.q(c,policy['delta'],policy['executed'])
                row['mean' if deterministic else 'sampled']=dict(n=len(raw),residual_vector_L2_RMS=float(policy['delta'].square().sum(-1).mean().sqrt()),residual_abs_max_per_axis=policy['delta'].abs().max(0).values.tolist(),clip_element_fraction=float((policy['proposed'].abs()>1).float().mean()),Q_abs_max=float(torch.maximum(q1.abs(),q2.abs()).max()),finite=bool(torch.isfinite(policy['executed']).all() and torch.isfinite(q1).all() and torch.isfinite(q2).all()))
        observations.append(row)
    result=dict(utc=utc(),source_sha256=sha(Path(__file__)),stored_state_source_sha256=sha(source),source_selection='first16ticks x16clones lexicographically, independent of outcomes; same256 actual recorded engineering states for all checkpoint observations',observations=observations,physical_interactions_added=0,training_updates_added=0,used_for_selection=False,scope='offline CPU policy and predicted Q observation; not physical action execution, Q calibration or learned-policy success evidence')
    save(RES/'engineering/checkpoint10000_observation.json',result);print(json.dumps(result))
if __name__=='__main__':main()
