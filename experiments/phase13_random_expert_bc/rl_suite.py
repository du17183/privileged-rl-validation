"""Small controlled RL only after independent operational anchor validation."""
import json
import torch
from concurrent.futures import ThreadPoolExecutor
from experiments.phase13_random_expert_bc.pipeline import ROOT,RESULT,CHECK,run,state


def main():
    anchor=json.loads((RESULT/'anchor_validation.json').read_text())
    comparison=json.loads((RESULT/'bc_primary_summary.json').read_text())
    expert=json.loads((RESULT/'expert_validation/summary.json').read_text())
    entry=(expert['success_rate']>.9 and comparison['paired']['success']['ci95'][0]>0
           and comparison['paired']['success']['positive_seeds']>=4
           and anchor['deterministic_summary']['ci95'][0]>.25
           and anchor['stochastic_summary']['ci95'][0]>.25)
    if not entry:
        state('rl_not_started',status='complete',reason='User relative-benefit prerequisites failed');return
    with (RESULT/'rl_entry_amendment.json').open('x') as f:
        json.dump(dict(original_engineering_gates_unchanged=dict(primary=comparison['anchor_gate_passed'],
                       operational=anchor['operational_anchor_gate_passed']),
            rationale='The additional absolute-success engineering gates were stricter than the user Phase13 instruction. The user instructed small RL if conditioned BC clearly beats robot-only. That relative criterion passed in five paired seeds and candidate retest lower CIs exceed 25%. Run bounded feasibility experiments, not deployment qualification.',
            source=anchor['source'],sha256=anchor['source_sha256'],source_label='candidate anchor; not qualified stable anchor',
            no_reselection=True,budget='3 matched seeds x 50016 steps in each A/B/C arm; no extension'),f,indent=2)
    destination=CHECK/'randomized_gt_candidate_anchor.pt'
    with destination.open('xb') as f:f.write((ROOT/anchor['source']).read_bytes())
    protocol=dict(seeds=[0,1,2],actual_steps_per_run=50016,parallel_envs=32,
        arms=dict(A='frozen selected BC policy, online execution only',
                  B='BC initialized symmetric SAC + continuous expert BC regularization',
                  C='same B + fixed anchor Normal KL weight .1'),
        common=dict(actor_lr=3e-5,critic_lr=3e-4,critic_warmup_updates=1000,bc_weight=10.,
                    replay='uniform 50% train expert transitions + 50% online transitions',
                    action_std=.01,alpha=1e-4,update_ratio='4 minibatches /32 interactions',
                    robot_and_environment_inputs='same 39 channels for Actor and Critic',
                    randomization='existing Level2: angle0-5deg, XYZ +/-1cm; actuator/reward/reset unchanged',
                    gradient_noise_seed='separate from common collecting action-noise seed'),
        validation=dict(every_steps=10000,episodes=64,seed='14401+training_seed',deterministic=True,
                        selection='best checkpoint on validation only; final retained separately',
                        frozen_baseline='initial and final evaluation, intermediate checkpoints saved without invented measurements'),
        heldout=dict(episodes=128,seed='14701+training_seed',mode='deterministic',
                     checkpoints=['best','final'],identical_policy_reuse='If all actor tensors and normalizers match, reuse already measured result with explicit audit',
                     stochastic_check='C final only, 64 episodes; ongoing training stochastic successes recorded in all arms'),
        no_rollback=True,max_concurrent_training_jobs=2,training_steps_total=450144,
        scope='No LWD/DIVL/QAM/quality selection; selected-model retest does not reclassify original five-BC-seed gate')
    with (RESULT/'rl_protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    # Complete one constrained run first as an integration check within the
    # formal 50k budget. No additional pilot or larger training allocation.
    run('rl_C_seed0','experiments.phase13_random_expert_bc.rl_train',['--arm','C','--seed',0,'--device','cuda:0'],7200)
    jobs=[(arm,seed) for seed in range(3) for arm in ['A','B','C'] if (arm,seed)!=('C',0)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,f'rl_{arm}_seed{seed}','experiments.phase13_random_expert_bc.rl_train',
                    ['--arm',arm,'--seed',seed,'--device','cuda:0'],7200) for arm,seed in jobs]
        for future in futures:future.result()
    evaljobs=[];reuse=[]
    def identical(left,right):
        x=torch.load(left,map_location='cpu',weights_only=False);y=torch.load(right,map_location='cpu',weights_only=False)
        return x['mean']==y['mean'] and x['std']==y['std'] and all(torch.equal(v,y['model'][k]) for k,v in x['model'].items())
    for seed in range(3):
        for arm in ['A','B','C']:
            # A is frozen, so best/final weights are identical. Evaluate once
            # per execution mode and transparently reuse that identity.
            for checkpoint in ['final']:
                arguments=['--checkpoint',CHECK/f'rl/{arm}_seed{seed}/{checkpoint}.pt',
                           '--output',RESULT/f'rl_heldout/{arm}_seed{seed}_{checkpoint}_deterministic.json',
                           '--episodes',128,'--num-envs',32,'--seed',14701+seed,'--device','cuda:0']
                evaljobs.append((f'rl_heldout_{arm}_seed{seed}_{checkpoint}_deterministic',arguments))
            if arm!='A':
                chosen=CHECK/f'rl/{arm}_seed{seed}/best.pt'
                if identical(chosen,CHECK/f'rl/{arm}_seed{seed}/final.pt'):
                    reuse.append(dict(destination=f'{arm}_seed{seed}_best_deterministic',source=f'{arm}_seed{seed}_final_deterministic'))
                elif identical(chosen,CHECK/f'rl/A_seed{seed}/final.pt'):
                    reuse.append(dict(destination=f'{arm}_seed{seed}_best_deterministic',source=f'A_seed{seed}_final_deterministic'))
                else:
                    evaljobs.append((f'rl_heldout_{arm}_seed{seed}_best_deterministic',
                        ['--checkpoint',chosen,'--output',RESULT/f'rl_heldout/{arm}_seed{seed}_best_deterministic.json',
                         '--episodes',128,'--num-envs',32,'--seed',14701+seed,'--device','cuda:0']))
            if arm=='C':
                evaljobs.append((f'rl_heldout_C_seed{seed}_final_stochastic',
                    ['--checkpoint',CHECK/f'rl/C_seed{seed}/final.pt','--output',RESULT/f'rl_heldout/C_seed{seed}_final_stochastic.json',
                     '--episodes',64,'--num-envs',32,'--seed',14701+seed,'--stochastic','--device','cuda:0']))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,name,'experiments.phase13_random_expert_bc.evaluate',arguments,1800) for name,arguments in evaljobs]
        for future in futures:future.result()
    for item in reuse:
        source=RESULT/f"rl_heldout/{item['source']}.json";destination=RESULT/f"rl_heldout/{item['destination']}.json"
        result=json.loads(source.read_text());result['identical_policy_reuse']=item
        with destination.open('x') as f:json.dump(result,f,indent=2)
    (RESULT/'rl_identical_policy_reuse.json').write_text(json.dumps(reuse,indent=2))
    state('rl_suite_complete',status='complete',next='paired analysis, final reporting and preservation audit')


if __name__=='__main__':
    try:main()
    except Exception as error:state('rl_suite_failed',status='failed',error=repr(error));raise
