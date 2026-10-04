"""Matched seed inference; masking attribution accounts for redundant signals."""
import csv,json
import numpy as np
from experiments.phase11_parameter_generalization.analyze import summarize,paired,holm,csv_rows
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,CKPT,VARIANTS,run_name,prepared
def main():
 if not (OUT/'heldout_completed.json').exists():raise RuntimeError('Independent tests incomplete')
 rows=[];curves={};held={};gen={};masks={};probes={}
 for a,s in VARIANTS:
  variant=f'{a}_{s}';held[variant]={};curves[variant]={};probes[variant]={}
  for seed in range(5):
   name=run_name(a,s,seed);obj=json.loads((OUT/'heldout'/f'{name}.json').read_text())
   lookup={(r['endpoint'],r['metrics']['condition'],r['metrics']['mode'],r['ablation']):r for r in obj['results']}
   held[variant][seed]=lookup;probes[variant][str(seed)]=obj['sensitivity']
   m=json.loads((CKPT/name/'completed.json').read_text());curve=csv_rows(OUT/f'eval_{name}_level2.csv')
   x=np.array([float(r['env_steps']) for r in curve]);y=np.array([float(r['success']) for r in curve])
   assert len(x)==31 and x[0]==0 and x[-1]==300000
   auc=float(np.trapz(y,x)/300000);final=lookup[('final','level2','policy',None)]['metrics'];best=lookup[('best','level2','policy',None)]['metrics']
   episodes=csv_rows(OUT/f'episodes_{name}.csv');late=[r for r in episodes if float(r['env_steps'])>200000]
   updates=csv_rows(OUT/f'updates_{name}.csv')
   assert all(float(u['std'])<=.010001 for u in updates) and final['effective_std']<=.010001
   row=dict(variant=variant,arm=a,strength=s,seed=seed,auc=auc,final_success=final['success'],best_success=best['success'],gap=best['success']-final['success'],
     nominal_final_success=lookup[('final','nominal','policy',None)]['metrics']['success'],
     online_successes=m['online_successes'],online_episodes=m['online_episodes'],online_success_ratio=m['online_successes']/m['online_episodes'],
     late_online_successes=int(sum(float(r['success']) for r in late)),rollback_events=m['rejections'],rollback_fraction=m['rejections']/30,
     accepted_blocks=m['acceptances'],best_step=m['best_step'],training_interactions=m['steps'],eval_interactions=m['cumulative_eval_steps'],
     effective_std=final['effective_std'],anchor_kl=final['anchor_kl'],
     door_angle_coverage_min=min(float(r['initial_angle_deg']) for r in episodes),door_angle_coverage_max=max(float(r['initial_angle_deg']) for r in episodes))
   rows.append(row);curves[variant][str(seed)]=curve
 groups={}
 for a,s in VARIANTS:
  key=f'{a}_{s}';selected=[r for r in rows if r['variant']==key]
  groups[key]={k:summarize([r[k] for r in selected]) for k in rows[0] if k not in ('variant','arm','strength','seed')}
  keys=list(held[key][0]);gen[key]={};masks[key]={}
  for endpoint in ('best','final'):
   gen[key][endpoint]={condition:{mode:summarize([held[key][seed][(endpoint,condition,mode,None)]['metrics']['success'] for seed in range(5)])
      for mode in ('policy','deterministic') if (endpoint,condition,mode,None) in keys}
     for condition in sorted({c for e,c,m,mask in keys if mask is None})}
   masks[key][endpoint]={mask:dict(masked_success=summarize([held[key][seed][(endpoint,'level2','policy',mask)]['metrics']['success'] for seed in range(5)]),
      drop=paired([held[key][seed][(endpoint,'level2','policy',None)]['metrics']['success'] for seed in range(5)],
                  [held[key][seed][(endpoint,'level2','policy',mask)]['metrics']['success'] for seed in range(5)]))
     for mask in sorted({mask for e,c,m,mask in keys if mask is not None})}
 def contrast(left,right):
  return dict(contrast=f'{left}-{right}',metrics={m:paired([r[m] for r in rows if r['variant']==left],[r[m] for r in rows if r['variant']==right]) for m in ('auc','final_success','nominal_final_success','gap','rollback_fraction')})
 core=[contrast(*pair) for pair in [('B_strong','A_strong'),('C_strong','B_strong'),('D_strong','C_strong'),('D_strong','A_strong')]]
 anchor=[contrast(f'{a}_{s}',f'{a}_strong') for a in ('B','D') for s in ('medium','weak','none')]
 efficacy=[contrast(f'{a}_{s}','A_strong') for a in ('B','D') for s in ('strong','medium','weak','none')]
 for family in (core,anchor,efficacy):
  for metric in ('auc','final_success'):
   adjusted=holm([r['metrics'][metric]['paired_t_p'] for r in family]);exact=holm([r['metrics'][metric]['exact_signflip_p'] for r in family])
   for i,r in enumerate(family):r['metrics'][metric].update(holm_paired_t_p=adjusted[i],holm_exact_p=exact[i])
 gates=[]
 for c in efficacy:
  key=c['contrast'].split('-')[0];e=c['metrics']['final_success'];g=groups[key]
  passed=(e['ci95'][0]>0 and e['holm_paired_t_p']<.05 and sum(v>0 for v in e['values'])>=4 and e['mean']>=.05
      and abs(g['gap']['mean'])<=.05 and g['rollback_fraction']['mean']<=.2)
  gates.append(dict(variant=key,passed=bool(passed),final_effect=e,gap=g['gap'],rollback_fraction=g['rollback_fraction']))
 gate=dict(passed=any(g['passed'] for g in gates),candidates=gates,rule=prepared()['extension_gate'],
     action='Extend qualifying groups plus matched A to500k' if any(g['passed'] for g in gates) else 'Do not expand to500k; prespecified efficacy/stability gate not met')
 # Per-clone/episode heldout plans must be identical despite different lengths.
 matched=0
 ref=held['A_strong']
 for variant,seed_data in held.items():
  for seed,lookup in seed_data.items():
   for endpoint,condition,mode,mask in lookup:
    if mask is not None:continue
    left=ref[seed][(endpoint,condition,mode,None)]['records'];right=lookup[(endpoint,condition,mode,None)]['records']
    fields=('initial_angle_deg','offset_x_m','offset_y_m','offset_z_m','friction_scale')
    def plan(records):return {(r['env_index'],r['episode_index']):[r[k] for k in fields] for r in records}
    assert plan(left)==plan(right);matched+=1
 report=dict(protocol=prepared(),groups=groups,core_contrasts=core,anchor_contrasts=anchor,efficacy_contrasts=efficacy,
    generalization=gen,input_masking=masks,sensitivity=probes,extension_gate=gate,
    training=json.loads((OUT/'training_completed.json').read_text()),heldout=json.loads((OUT/'heldout_completed.json').read_text()),
    heldout_parameter_plan_comparisons=matched,
    notes=['Same-seed policies share exact per-clone initial plans; policy-dependent episode count is not an identical state visitation distribution.',
      'Mask tests evaluate input dependence and possible OOD effects, not retrained feature causality. Door angle, progress, remaining angle are redundant.',
      'AUC uses training validation; Best selected by that validation, independent endpoint tests do not select checkpoints.',
      'All C/D inherit all five B progress channels. Actor and critic observe the same state. GT is logged only outside those vectors.',
      'n=5 seed t intervals depend on assumptions; exact two-sided signflip cannot be below .0625. Multiple comparisons use Holm; no episode pseudoreplication.'])
 (OUT/'summary.json').write_text(json.dumps(report,indent=2));(OUT/'extension_gate.json').write_text(json.dumps(gate,indent=2))
 (OUT/'per_seed.json').write_text(json.dumps(rows,indent=2));(OUT/'curves.json').write_text(json.dumps(curves))
 with (OUT/'per_seed.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({k:{m:g[m]['mean'] for m in ('auc','final_success','gap','rollback_fraction')} for k,g in groups.items()},indent=2),flush=True)
if __name__=='__main__':main()
