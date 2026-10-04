"""Audit terminal snapshots, feature parity, full reset exposure and old files."""
import hashlib,json
import h5py,numpy as np
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,CKPT,VARIANTS,run_name,prepared
def main():
 protocol=prepared();results=[];plans={}
 for a,s in VARIANTS:
  for seed in range(5):
   name=run_name(a,s,seed);marker=json.loads((CKPT/name/'completed.json').read_text())
   assert marker['steps']==300000 and marker['optimizer_steps']==37500
   with h5py.File(ROOT/'datasets'/OUT.name/f'{name}.h5','r') as h:
    d=h['transitions'];records=json.loads(h['episodes_json'][()]);done=d['done'][:,0];reward=d['reward'][:,0]
    gt=d['privileged'][:];ngt=d['next_privileged'][:];covered=np.zeros(300000,bool);per_env={i:[] for i in range(32)}
    for r in records:
     indices=r['start']+np.arange(r['length'])*32
     assert not covered[indices].any();covered[indices]=True
     assert done[indices[-1]]==1 and done[indices[:-1]].sum()==0
     assert bool(ngt[indices[-1],0]>1)==bool(r['success'])
     assert abs(float(reward[indices].sum())-r['return_value'])<.02
     assert abs(float(ngt[indices[-1],0])-r['final_angle'])<1e-6
     assert indices[0]%32==r['env_index'] and r['curriculum_level']==2
     assert 0<=r['initial_angle_deg']<=5.00001 and r['offset_linf_cm']<=1.00001 and r['friction_scale']==1.
     per_env[r['env_index']].append(r)
    for i,rows in per_env.items():
     if rows:assert rows[0]['start']==i
     for previous,current in zip(rows,rows[1:]):assert current['start']==previous['start']+previous['length']*32
    assert len(records)==marker['online_episodes'] and int(sum(r['success'] for r in records))==marker['online_successes']
    assert d['robot'].shape[1]==marker['observation_dim']==protocol['observation_dims'][a]
    if a!='A':
     for field,physical in [('robot',gt),('next_robot',ngt)]:
      obs=d[field][:]
      assert np.allclose(obs[:,26:28],physical[:,:2],atol=1e-7)
      assert np.allclose(obs[:,28],1.) and np.allclose(obs[:,30],1.-physical[:,0],atol=1e-7)
      assert np.min(obs[:,29])>=0 and np.max(obs[:,29])<=1.
      if a in ('C','D'):assert np.allclose(obs[:,31:33],physical[:,9:11],atol=1e-7)
      if a=='D':assert np.allclose(obs[:,33:36],(physical[:,2:5]-np.array(protocol['handle_center']))/.1,atol=1e-6)
    plans[(a,s,seed)]={i:[[r[k] for k in ('initial_angle_deg','offset_x_m','offset_y_m','offset_z_m','friction_scale')] for r in rows] for i,rows in per_env.items()}
    results.append(dict(arm=a,strength=s,seed=seed,episodes=len(records),successes=marker['online_successes'],
      covered_completed_transitions=int(covered.sum()),pending_transitions=int((~covered).sum()),full_level2_only=True,terminal_state_verified=True))
 matched=0
 for seed in range(5):
  reference=plans[('A','strong',seed)]
  for a,s in VARIANTS:
   current=plans[(a,s,seed)]
   for i in range(32):
    n=min(len(reference[i]),len(current[i]))
    assert reference[i][:n]==current[i][:n];matched+=n
 changes=[];inventory=json.loads((OUT/'baseline_inventory.json').read_text())
 for name,old in inventory.items():
  p=ROOT/name
  if not p.exists():changes.append(name);continue
  st=p.stat();current=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
  if 'sha256' in old:current['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  if current!=old:changes.append(name)
 manifest=json.loads((OUT/'training_source_manifest.json').read_text())
 source_changes=[name for name,digest in manifest.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
 result=dict(passed=not changes and not source_changes,datasets=50,results=results,prior_artifacts=len(inventory),prior_changed=changes,
  training_source_changes=source_changes,matched_training_episode_parameters=matched,
  total_completed_episodes=sum(r['episodes'] for r in results),total_successes=sum(r['successes'] for r in results))
 (OUT/'formal_data_audit.json').write_text(json.dumps(result,indent=2))
 if not result['passed']:raise RuntimeError('Historical/source audit failed')
 print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
if __name__=='__main__':main()
