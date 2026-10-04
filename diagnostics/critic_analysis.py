"""Join per-vector-step Phase 3 telemetry and fixed Phase 2 Q probes."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'door_privileged_ablation'
WINDOWS = ((0, 100000), (100000, 200000), (200000, 300000),
           (300000, 400000), (400000, 500001))
FIELDS = ('critic_loss', 'q_mean', 'q_variance', 'q_disagreement',
          'actor_loss', 'policy_entropy', 'online_success_rolling256')


def records(path):
    with path.open(newline='', encoding='utf-8') as stream:
        yield from csv.DictReader(stream)


def summarize_training():
    rows = []
    for variant in ('diag_A', 'diag_B'):
        for seed in range(5):
            path = OUT / f'diagnostics_{variant}_seed{seed}.csv'
            groups = defaultdict(lambda: defaultdict(list))
            for row in records(path):
                step = int(row['env_steps'])
                slot = next(i for i, (low, high) in enumerate(WINDOWS) if low <= step < high)
                for field in FIELDS:
                    value = float(row[field])
                    if np.isfinite(value): groups[slot][field].append(value)
            for slot, measures in sorted(groups.items()):
                rows.append({'variant':variant, 'seed':seed,
                             'window_start':WINDOWS[slot][0],
                             'window_end':min(WINDOWS[slot][1], 500000),
                             **{field:float(np.mean(measures[field])) if measures[field] else float('nan')
                                for field in FIELDS}})
    with (OUT / 'diagnostic_window_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def summarize_probe():
    grouped = defaultdict(list)
    for row in records(OUT / 'fixed_probe_phase2.csv'):
        grouped[(row['variant'], int(row['env_steps']))].append(row)
    result = {}
    for variant in ('A', 'B'):
        for step in (0, 100000, 200000, 300000, 400000, 500000):
            # Phase 2 checkpoint intervals round to the next 32-vector step.
            actual = min((s for v, s in grouped if v == variant), key=lambda s:abs(s-step))
            rows = grouped[(variant, actual)]
            result[(variant, step)] = {
                field:float(np.mean([float(r[field]) for r in rows]))
                for field in ('q_expert_mean', 'q_expert_variance', 'q_policy_mean',
                              'q_policy_minus_expert', 'q_twin_disagreement',
                              'expert_td_abs', 'policy_action_q_grad_l2',
                              'q_gradient_toward_expert_cos',
                              'q_gradient_toward_expert_fraction')}
    return result


def mean_window(rows, variant, low):
    subset = [r for r in rows if r['variant'] == variant and r['window_start'] == low]
    return {key: float(np.nanmean([r[key] for r in subset])) for key in FIELDS}


def collapse_timing():
    probe = defaultdict(dict)
    for row in records(OUT / 'fixed_probe_phase2.csv'):
        probe[(row['variant'], int(row['seed']))][int(row['env_steps'])] = row
    timing = []
    for seed in range(5):
        curve = list(records(ROOT / 'results' / 'door' / f'eval_B_seed{seed}.csv'))
        peak = max(range(len(curve)), key=lambda i: float(curve[i]['success_rate']))
        collapse = next((int(r['env_steps']) for r in curve[peak+1:]
                         if float(r['success_rate']) <= .1), None)
        a, b = probe[('A',seed)], probe[('B',seed)]
        divergence = next((step for step in sorted(b) if
                           float(b[step]['q_expert_variance']) >
                           10 * max(float(a[step]['q_expert_variance']), 1e-6)), None)
        timing.append({'seed':seed,'peak_steps':int(curve[peak]['env_steps']),
                       'peak_success':float(curve[peak]['success_rate']),
                       'first_success_le_0_1_after_peak':collapse,
                       'first_B_Qvar_gt_10x_A_Qvar':divergence,
                       'B_Qvar_at_peak':float(b[int(curve[peak]['env_steps'])]['q_expert_variance']),
                       'B_Qvar_final':float(b[500000]['q_expert_variance']),
                       'B_Qmean_final':float(b[500000]['q_expert_mean']),
                       'B_TD_abs_final':float(b[500000]['expert_td_abs']),
                       'B_grad_toward_expert_final':float(b[500000]['q_gradient_toward_expert_cos']),
                       'A_grad_toward_expert_final':float(a[500000]['q_gradient_toward_expert_cos'])})
    with (OUT/'collapse_timing.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(timing[0]))
        writer.writeheader();writer.writerows(timing)
    return timing


def plot_fixed_probe():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=defaultdict(list)
    for row in records(OUT/'fixed_probe_phase2.csv'):
        data[(row['variant'],int(row['seed']))].append(row)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    fields=('q_expert_mean','q_expert_variance','q_gradient_toward_expert_cos')
    titles=('Expert-state Q mean','Expert-state Q variance','Q gradient toward expert')
    for variant,color in (('A','#2271b2'),('B','#d04c4c')):
        for seed in range(5):
            rows=sorted(data[(variant,seed)],key=lambda r:int(r['env_steps']))
            x=[int(r['env_steps']) for r in rows]
            for ax,field in zip(axes,fields):
                ax.plot(x,[float(r[field]) for r in rows],color=color,alpha=.35,lw=1,
                        label=variant if seed==0 else None)
    for ax,title in zip(axes,titles):
        ax.set(title=title,xlabel='Online environment steps')
        ax.grid(alpha=.2)
    axes[0].set_yscale('symlog',linthresh=10)
    axes[1].set_yscale('log')
    axes[2].axhline(0,color='black',lw=.6)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT/'fixed_probe_trajectories.png',dpi=180)
    plt.close(fig)


def main():
    rows = summarize_training()
    probe = summarize_probe()
    timing = collapse_timing()
    plot_fixed_probe()
    lines = [
        '# Door critic training diagnosis', '',
        'Five paired seeds, 500k online transitions each, unchanged Phase 2 Door task and expert data.',
        'Training telemetry is sampled after every 32-environment vector step (four SAC gradient updates).',
        'Policy entropy is a Monte Carlo estimate of the tanh-squashed action entropy on the final training minibatch.',
        'The fixed probe uses the same 256 expert transitions at every Phase 2 checkpoint.', '',
        '## Online training windows', '',
        '| Variant | Environment steps | Critic loss | Q mean | Q variance | Twin Q disagreement | Actor loss | Policy entropy | Rolling training success |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for variant in ('diag_A', 'diag_B'):
        for low, high in WINDOWS:
            x = mean_window(rows, variant, low)
            lines.append(f'| {variant} | {low:,}–{min(high,500000):,} | '
                         f"{x['critic_loss']:.3f} | {x['q_mean']:.3f} | {x['q_variance']:.3f} | "
                         f"{x['q_disagreement']:.3f} | {x['actor_loss']:.3f} | "
                         f"{x['policy_entropy']:.3f} | {x['online_success_rolling256']:.3f} |")
    lines += ['', '## Fixed expert-state checkpoint probe', '',
              '| Variant | Steps | Expert Q | Q variance | Policy−expert Q | Twin disagreement | Expert TD absolute error | Action gradient norm | Gradient toward expert cosine |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for variant in ('A', 'B'):
        for step in (0, 100000, 200000, 300000, 400000, 500000):
            x = probe[(variant, step)]
            lines.append(f"| {variant} | {step:,} | {x['q_expert_mean']:.3f} | "
                         f"{x['q_expert_variance']:.3f} | {x['q_policy_minus_expert']:.3f} | "
                         f"{x['q_twin_disagreement']:.3f} | {x['expert_td_abs']:.3f} | "
                         f"{x['policy_action_q_grad_l2']:.3f} | "
                         f"{x['q_gradient_toward_expert_cos']:.3f} |")
    lines += ['', '## Seed-level collapse timing (Phase 2 checkpoints)', '',
              '| Seed | Peak step | Peak success | First ≤10% success after peak | First B Q variance >10× A | B Q variance at peak | B Q variance at 500k | B gradient toward expert at 500k |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in timing:
        collapse = row['first_success_le_0_1_after_peak']
        diverge = row['first_B_Qvar_gt_10x_A_Qvar']
        lines.append(f"| {row['seed']} | {row['peak_steps']:,} | {row['peak_success']:.3f} | "
                     f"{collapse if collapse is not None else 'not reached'} | "
                     f"{diverge if diverge is not None else 'not reached'} | "
                     f"{row['B_Qvar_at_peak']:.2f} | {row['B_Qvar_final']:.2f} | "
                     f"{row['B_grad_toward_expert_final']:.3f} |")
    lines += ['', '## Interpretation boundary', '',
              'The fixed probe detects drift or miscalibration on a held-constant expert distribution; it does not measure actual on-policy return.',
              'A more positive predicted policy-versus-expert Q advantage with worse evaluated success is evidence of a ranking mismatch, not proof of its mechanism.',
              'Freezing and switching interventions in the Phase 3 ablation provide stronger intervention evidence.', '']
    (OUT / 'training_diagnosis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {OUT / "training_diagnosis.md"}')


if __name__ == '__main__':
    main()
