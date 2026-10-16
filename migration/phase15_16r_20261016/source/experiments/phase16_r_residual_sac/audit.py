"""Independent saved-data audit; never executes or changes historical experiments."""
import sys,json,csv,traceback
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np,h5py,torch
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,save,sha,utc,protocol

def main():
    P=protocol();issues=[];checks={};hist=json.loads((RES/'historical_inputs.json').read_text())
    for rel,wanted in hist['protected'].items():
        if sha(ROOT/rel)!=wanted:issues.append('Historical SHA changed '+rel)
    checks['historical_protected_files']=len(hist['protected'])
    manifest=json.loads((RES/'source_manifest.json').read_text())
    for rel,wanted in manifest['files'].items():
        if sha(ROOT/rel)!=wanted:issues.append('Execution source changed '+rel)
    checks['execution_source_files']=len(manifest['files'])
    inputs=json.loads((ROOT/'datasets/phase14_2/prepared_v1/manifest.json').read_text());groups=[set().union(*(set(role[key]) for role in inputs['splits'].values())) for key in ['train','validation','test']]
    overlap=any(groups[i]&groups[j] for i in range(3) for j in range(i))
    if overlap:issues.append('Whole-trajectory data splits overlap')
    checks['old_data_split_disjoint']=not overlap
    five=[];traces=0;interactions=0;maximum_errors=dict(gripper=0.,composition=0.,chunk=0.,reward_C=0.,reward_D=0.)
    paths=[];statuses={};pairing=[]
    for seed in range(5):
        folder=RES/f'baseline/seed{seed}'
        if not (folder/'complete.json').exists():issues.append('New baseline incomplete '+str(seed));continue
        for arm in ['A','B']:
            for scope,n in [('random',128),('fixed',32),('validation',64)]:
                x=json.loads((folder/f'{arm}_{scope}.json').read_text());assert x['episodes']==n
                with (folder/f'{arm}_{scope}.csv').open() as f:rows=list(csv.DictReader(f))
                if len(rows)!=n or np.mean([int(r['success']) for r in rows])!=x['success']:issues.append('Baseline aggregation '+arm+scope+str(seed))
                if any(int(r['ticks'])>600 for r in rows):issues.append('Episode over budget')
                if arm=='A' and scope=='random':five.append(x['success'])
                paths.append(folder/f'{arm}_{scope}.h5')
    for arm in ['C','D']:
        for seed in range(3):
            folder=RES/f'train/{arm}_seed{seed}'
            status=next((folder/name for name in ['complete.json','stopped.json'] if (folder/name).exists()),None)
            if status is None:issues.append('Main trial missing preserved completion/controlled stop '+arm+str(seed));continue
            c=json.loads(status.read_text());statuses[arm+str(seed)]=c
            completed=status.name=='complete.json';steps=c['training_interactions']
            updates=max(0,steps//8-2048//8+2) if steps>=2048 else 0
            if steps>50000 or steps%16 or c['optimizer_updates']!=updates or (completed and steps!=50000):issues.append('Budget/update mismatch '+arm+str(seed))
            if not completed and not c.get('controlled_trial_stop'):issues.append('Unexplained incomplete training '+arm+str(seed))
            paths.append(folder/'training.h5')
            replay=np.load(folder/('replay_final.npz' if completed else 'replay_stopped.npz'));assert len(replay['context'])==steps
            from experiments.phase16_r_residual_sac.common import checkpoint
            normal=torch.load(checkpoint(seed),map_location='cpu',weights_only=False);mean=np.asarray(normal['mean'],np.float32);std=np.asarray(normal['std'],np.float32)
            with h5py.File(folder/'training.h5') as h:
                base=h['base'][:];chunk=h['chunk'][:];cur=h['cursor'][:];raw=h['raw'][:];done=h['done'][:]
                ctx=np.concatenate((2/np.pi*np.arctan((raw-mean)/std),base,chunk.reshape(*raw.shape[:2],70),np.eye(5,dtype=np.float32)[cur]),-1).astype(np.float32)
                if np.max(abs(ctx.reshape(-1,121)-replay['context']))>1e-6:issues.append('Replay context differs '+arm+str(seed))
                actual=replay['next_base'].reshape(*base.shape)
                mask=~done[:-1]
                if not np.array_equal(actual[:-1][mask],base[1:][mask]):issues.append('Next actual base substituted '+arm+str(seed))
                target=replay['next_context'].reshape(*ctx.shape)
                if np.max(abs(target[:-1][mask]-ctx[1:][mask]))>1e-6:issues.append('Next chunk context mismatch '+arm+str(seed))
                if np.any(actual[done]!=0) or np.any(target[done]!=0):issues.append('Terminal bootstrap context not zero')
                reward=replay['original_reward' if arm=='C' else 'signed_reward']
                if not np.array_equal(replay['reward'],reward):issues.append('Wrong C/D reward selected')
            for step in P['evaluation']['nodes']:
                if step>steps:continue
                ev=RES/f'evaluation/{arm}_seed{seed}/step{step}'
                if not (ev/'complete.json').exists():issues.append('Independent node missing '+str(ev));continue
                for path in ev.glob('*.h5'):paths.append(path)
            ev=RES/f'evaluation/{arm}_seed{seed}/best_test'
            if (ev/'complete.json').exists():paths.extend(ev.glob('*.h5'))
    for seed in range(3):
        files=[ROOT/f'checkpoints/phase16_r/{arm}_seed{seed}/initial.pt' for arm in ['C','D']]
        if all(p.exists() for p in files):
            c,d=[torch.load(p,map_location='cpu',weights_only=False) for p in files]
            for module in ['actor','q','target']:
                if any(not torch.equal(c[module][key],d[module][key]) for key in c[module]):issues.append('C/D initialization not paired '+str(seed)+module)
            if not torch.equal(c['log_alpha'],d['log_alpha']) or not torch.equal(c['bound'],d['bound']) or c['cfg']!=d['cfg']:issues.append('C/D initial temperature/config differs')
            if any(torch.count_nonzero(v) for k,v in c['actor'].items() if k.endswith('4.weight')) or torch.count_nonzero(c['actor']['net.4.bias'][:6]):issues.append('Actor initialization not zero')
    # Pair environmental draws by clone and episode, never by completion order.
    for seed in range(5):
        for scope in ['random','validation','fixed']:
            ref=RES/f'baseline/seed{seed}/A_{scope}.initial.json'
            if not ref.exists():continue
            a={(r['clone'],r['episode']):r for r in json.loads(ref.read_text())}
            alternatives=[RES/f'baseline/seed{seed}/B_{scope}.initial.json']
            if seed<3:
                for arm in ['C','D']:
                    for ev in (RES/f'evaluation/{arm}_seed{seed}').glob('*'):
                        for dist in ['sampled','mean']:alternatives.append(ev/f'{scope}_{dist}.initial.json')
            for path in alternatives:
                if not path.exists():continue
                b={(r['clone'],r['episode']):r for r in json.loads(path.read_text())};keys=a.keys()&b.keys()
                if not keys:issues.append('No paired reset identities '+str(path));continue
                perr=max(float(np.max(abs(np.asarray(a[k]['parameters'])-np.asarray(b[k]['parameters'])))) for k in keys)
                rerr=max(float(np.max(abs(np.asarray(a[k]['raw'])-np.asarray(b[k]['raw'])))) for k in keys)
                first=max(float(np.max(abs(np.asarray(a[k]['raw'])-np.asarray(b[k]['raw'])))) for k in keys if k[1]==0)
                if perr:issues.append('Outcome-dependent environmental draws '+str(path))
                if first>1e-6:issues.append('Fresh paired initial physical state differs '+str(path))
                pairing.append(dict(path=str(path.relative_to(ROOT)),pairs=len(keys),requested_parameters_max_error=perr,first_episode_observed_state_max_error=first,later_auto_reset_state_max_error=rerr,hidden_physics_reset_history='not fully observable; recorded state mismatch, if any, is disclosed rather than assumed zero'))
    for path in paths:
        with h5py.File(path) as h:
            base=h['base'][:];delta=h['delta'][:];actual=h['executed'][:];raw=h['raw'][:];nr=h['next_robot'][:];gt=h['next_gt'][:];cur=h['cursor'][:];chunk=h['chunk'][:]
            expected=np.concatenate((np.clip(base[:,:,:6]+delta,-1,1),base[:,:,6:7]),-1)
            maximum_errors['composition']=max(maximum_errors['composition'],float(abs(expected-actual).max()))
            maximum_errors['gripper']=max(maximum_errors['gripper'],float(abs(actual[:,:,6]-base[:,:,6]).max()))
            used=chunk[np.arange(len(chunk))[:,None],np.arange(base.shape[1])[None,:],cur]
            maximum_errors['chunk']=max(maximum_errors['chunk'],float(abs(used-base).max()))
            # Independently reconstruct the original integrated reward from saved
            # post-action physical EE/handle/contact/angle, including terminals.
            reach=1-np.tanh(5*np.linalg.norm(nr[:,:,18:21]-gt[:,:,2:5],axis=-1));grasp=(gt[:,:,9]>.5)&(gt[:,:,10]>.5)
            original=(reach+.5*grasp+20*np.maximum(gt[:,:,1],0)+600*(gt[:,:,0]>1))/60
            signed=h['reward_original'][:]-20*np.maximum(gt[:,:,1],0)/60+20*(gt[:,:,0]-raw[:,:,26])
            maximum_errors['reward_C']=max(maximum_errors['reward_C'],float(abs(original-h['reward_original'][:]).max()))
            maximum_errors['reward_D']=max(maximum_errors['reward_D'],float(abs(signed-h['reward_signed'][:]).max()))
            if not np.isfinite(actual).all() or np.any(abs(actual)>1):issues.append('Illegal physical command '+str(path))
            traces+=1;interactions+=actual.shape[0]*actual.shape[1]
    for k,value in maximum_errors.items():
        if value>(2e-6 if k.startswith('reward') else 1e-6):issues.append('Independent numerical mismatch '+k+' '+str(value))
    baseline=json.loads((RES/'Frozen_pi05_baseline.json').read_text()) if (RES/'Frozen_pi05_baseline.json').exists() else None
    if baseline and (len(five)!=5 or np.mean(five)!=baseline['mean_success']):issues.append('Five-seed baseline mismatch')
    checks.update(five_seed_baseline=five,audited_physical_traces=traces,audited_physical_interactions=interactions,maximum_errors=maximum_errors,main_training_seeds=3,significance_claim=False)
    checks.update(trial_status=statuses,full_300000_training_budget=all(v['training_interactions']==50000 for v in statuses.values()) and len(statuses)==6,paired_reset_rows=pairing)
    for name in ['physical_gate','sac_real_gate']:
        path=RES/'engineering'/f'{name}.json'
        if not json.loads(path.read_text()).get('passed'):issues.append('Real engineering gate failed '+name)
    for name in ['physical_cost_ledger.json','latency/complete.json']:
        path=RES/name
        if not path.exists() or not json.loads(path.read_text()).get('passed'):issues.append('Final delivery evidence missing '+name)
    if not (RES/'first_round_complete.json').exists() and not (RES/'first_round_execution_finished.json').exists():issues.append('Supervisor execution not finished')
    life=json.loads((RES/'lifecycle.json').read_text())
    from experiments.phase16_r_residual_sac.monitor import proc
    expected=[life['supervisor'],*life.get('launched',[])]
    for path in RES.rglob('physics_children.json'):
        expected.extend(r['process'] for r in json.loads(path.read_text())['children'])
    alive=[]
    for old in expected:
        current=proc(old['pid'])
        if current.get('alive') and all(current.get(k)==old.get(k) for k in ['argv','ppid','start_ticks']):alive.append(current)
    if alive:issues.append('Owned execution processes still active')
    checks['owned_active_processes']=alive
    report=ROOT/'docs/phase16_r_report.md'
    save(RES/'independent_audit.json',dict(utc=utc(),passed=not issues,issues=issues,checks=checks,auditor_source_sha256=sha(Path(__file__)),protocol_sha256=sha(EXP/'protocol.json'),report_sha256=sha(report) if report.exists() else None,scope='independent formulas over preserved physical traces, replay, paired checkpoint tensors and full historical SHA; not a runtime re-evaluation'))
    print(json.dumps(dict(passed=not issues,issues=issues,checks=checks)))
    if issues:sys.exit(1)
if __name__=='__main__':main()
