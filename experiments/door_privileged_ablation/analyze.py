"""Five-seed Door Phase 3 comparison with paired uncertainty and curves."""

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "door_privileged_ablation"
PHASE2 = ROOT / "results" / "door"
NEW = ("diag_A", "diag_B", "B1", "B2", "B5", "B6", "E0", "E1", "E2", "E3")
OLD = ("A", "B", "C", "D")
THRESHOLDS = (0.5, 0.8, 0.9)


def read_curve(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 21:
        raise ValueError(f"Need exactly 21 evaluation points: {path} has {len(rows)}")
    x = np.array([int(r["env_steps"]) for r in rows])
    y = np.array([float(r["success_rate"]) for r in rows])
    reward = np.array([float(r["mean_return"]) for r in rows])
    if x[0] != 0 or x[-1] != 500000 or np.any(np.diff(x) <= 0):
        raise ValueError(f"Incomplete or nonmonotonic curve: {path}")
    return x, y, reward


def seed_metrics(x, y, reward):
    peak_idx = int(np.argmax(y))
    result = {"auc": float(np.trapz(y, x) / 500000),
              "best_so_far_auc": float(np.trapz(np.maximum.accumulate(y), x) / 500000),
              "final_success": float(y[-1]),
              "late_success_400k_500k": float(y[x >= 400000].mean()),
              "peak_success": float(y[peak_idx]),
              "peak_steps": int(x[peak_idx]), "post_peak_drop": float(y[peak_idx] - y[-1]),
              "reward_auc": float(np.trapz(reward, x) / 500000),
              "final_reward": float(reward[-1])}
    for threshold in THRESHOLDS:
        hit = np.flatnonzero(y >= threshold)
        retained = [int(x[i]) for i in hit if i < len(y)-1 and np.all(y[i:] >= threshold)]
        result[f"first_{threshold}"] = int(x[hit[0]]) if len(hit) else None
        result[f"retained_{threshold}"] = retained[0] if retained else None
    return result


def bootstrap(values, seed=20260929):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, (20000, len(values)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def signflip(values):
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    return sum(abs(float(np.mean(values * signs))) >= observed - 1e-12
               for signs in itertools.product((-1, 1), repeat=len(values))) / 2 ** len(values)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plotted = ("A", "B", "diag_A", "diag_B", "B1", "B2", "B5", "B6", "E0", "E1", "E2", "E3")
    for index, (suffix, ylabel) in enumerate((('success', 'Success rate'), ('reward', 'Episode return'))):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for variant in plotted:
            x = curves[(variant, 0)][0]
            values = np.stack([curves[(variant, seed)][index + 1] for seed in range(5)])
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(5)
            ax.plot(x, mean, label=variant)
            ax.fill_between(x, mean - sem, mean + sem, alpha=.12)
        ax.set(xlabel='Online environment steps', ylabel=ylabel)
        if suffix == 'success': ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
        ax.legend(ncol=5, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f'{suffix}_curve.png', dpi=180)
        plt.close(fig)
    groups = {
        'critic': [('A','Historical A','#777777','--'),('B','Historical B','#aaaaaa','--'),
                   ('diag_A','Diagnostic A','#1f77b4','-'),('diag_B','Full GT','#d62728','-'),
                   ('B1','Angle only','#2ca02c','-'),('B2','Angle + velocity','#9467bd','-'),
                   ('B5','Freeze Q','#8c564b','-'),('B6','Switch Q','#ff7f0e','-')],
        'auxiliary': [('diag_A','Diagnostic A','#777777','--'),('E0','No GT auxiliary','#1f77b4','-'),
                      ('E1','Continuous GT','#d62728','-'),('E2','GT first 100k','#2ca02c','-'),
                      ('E3','GT in BC only','#ff7f0e','-')],
    }
    for group, variants in groups.items():
        for index, (suffix, ylabel) in enumerate((('success','Success rate'),('reward','Episode return'))):
            fig, ax = plt.subplots(figsize=(8.4,4.8))
            for variant,label,color,style in variants:
                x=curves[(variant,0)][0]
                values=np.stack([curves[(variant,seed)][index+1] for seed in range(5)])
                mean=values.mean(axis=0)
                sem=values.std(axis=0,ddof=1)/np.sqrt(5)
                ax.plot(x,mean,label=label,color=color,linestyle=style,lw=2.2)
                if style!='--': ax.fill_between(x,mean-sem,mean+sem,color=color,alpha=.10)
            ax.set(xlabel='Online environment steps',ylabel=ylabel)
            if suffix=='success': ax.set_ylim(0,1)
            ax.grid(alpha=.22)
            ax.legend(ncol=2,fontsize=8,loc='upper right')
            fig.tight_layout()
            fig.savefig(OUT/f'{suffix}_curve_{group}.png',dpi=180)
            plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves = {}
    per_seed = []
    for variant in OLD + NEW:
        source = PHASE2 if variant in OLD else OUT
        for seed in range(5):
            x, y, reward = read_curve(source / f'eval_{variant}_seed{seed}.csv')
            curves[(variant, seed)] = (x, y, reward)
            per_seed.append({'variant': variant, 'seed': seed, **seed_metrics(x, y, reward)})
    write_csv(OUT / 'seed_metrics.csv', per_seed)
    by_variant = defaultdict(list)
    for row in per_seed: by_variant[row['variant']].append(row)
    aggregates = []
    for variant, rows in by_variant.items():
        auc = [r['auc'] for r in rows]
        best_auc = [r['best_so_far_auc'] for r in rows]
        final = [r['final_success'] for r in rows]
        late = [r['late_success_400k_500k'] for r in rows]
        drop = [r['post_peak_drop'] for r in rows]
        aggregates.append({'variant': variant, 'seeds': len(rows),
                           'auc_mean': float(np.mean(auc)), 'auc_ci95_low': bootstrap(auc)[0],
                           'auc_ci95_high': bootstrap(auc)[1],
                           'best_so_far_auc_mean':float(np.mean(best_auc)),
                           'final_success_mean': float(np.mean(final)),
                           'final_ci95_low': bootstrap(final)[0],
                           'final_ci95_high': bootstrap(final)[1],
                           'late_success_mean':float(np.mean(late)),
                           'late_ci95_low':bootstrap(late)[0],
                           'late_ci95_high':bootstrap(late)[1],
                           'post_peak_drop_mean': float(np.mean(drop)),
                           'reward_auc_mean': float(np.mean([r['reward_auc'] for r in rows]))})
    write_csv(OUT / 'aggregate_metrics.csv', aggregates)
    comparisons = []
    for baseline, contender in (('A','B'), ('diag_A','diag_B'), ('diag_A','B1'),
                                ('diag_A','B2'), ('diag_A','B5'), ('diag_A','B6'),
                                ('E0','E1'), ('E0','E2'), ('E1','E2'),
                                ('E0','E3'), ('E1','E3'), ('E2','E3'),
                                ('diag_A','E1'), ('diag_A','E2'), ('diag_B','E1'),
                                ('diag_B','B1'), ('diag_B','B2'), ('diag_B','B5'),
                                ('diag_B','B6')):
        for metric in ('auc', 'best_so_far_auc', 'final_success',
                       'late_success_400k_500k', 'post_peak_drop'):
            a = np.array([by_variant[baseline][seed][metric] for seed in range(5)])
            b = np.array([by_variant[contender][seed][metric] for seed in range(5)])
            diff = b - a
            ci = bootstrap(diff)
            comparisons.append({'comparison': f'{contender}_minus_{baseline}', 'metric': metric,
                                'mean_difference': float(diff.mean()), 'ci95_low': ci[0],
                                'ci95_high': ci[1], 'exact_signflip_p': signflip(diff),
                                'seed_differences': ';'.join(f'{v:.6f}' for v in diff)})
    write_csv(OUT / 'paired_effects.csv', comparisons)
    steps = []
    for variant in OLD + NEW:
        for threshold in THRESHOLDS:
            hits = [r[f'first_{threshold}'] for r in by_variant[variant]]
            stable = [r[f'retained_{threshold}'] for r in by_variant[variant]]
            steps.append({'variant':variant, 'threshold':threshold,
                          'seeds_reaching':sum(v is not None for v in hits),
                          'median_first_steps_if_reached':float(np.median([v for v in hits if v is not None])) if any(v is not None for v in hits) else '',
                          'seeds_retaining':sum(v is not None for v in stable),
                          'median_retained_steps_if_reached':float(np.median([v for v in stable if v is not None])) if any(v is not None for v in stable) else ''})
    write_csv(OUT / 'thresholds.csv', steps)
    summary = {'protocol': 'door_phase3_500k_5seed', 'aggregate': aggregates,
               'comparisons': comparisons, 'thresholds': steps,
               'aliases': {'B3': 'diag_B', 'B4': 'diag_B'}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot(curves)
    print(f'Analyzed {len(per_seed)} complete curves; outputs in {OUT}')


if __name__ == '__main__':
    main()
