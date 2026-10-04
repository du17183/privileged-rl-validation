"""Episode-level ranking metrics shared by frozen Q and quality models."""

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr


def success_ranking_accuracy(score, success):
    """Probability a successful episode scores above a failed one (ties=.5)."""
    score = np.asarray(score, dtype=float)
    success = np.asarray(success, dtype=bool)
    n_pos = int(success.sum())
    n_neg = int((~success).sum())
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum = rankdata(score, method="average")[success].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def safe_corr(score, target, method):
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(score) < 3 or np.std(score) < 1e-10 or np.std(target) < 1e-10:
        return float("nan")
    return float((pearsonr if method == "pearson" else spearmanr)(score, target).statistic)


def evaluate(score, discounted_return, episode_return, success, source):
    score = np.asarray(score, dtype=float).reshape(-1)
    discounted_return = np.asarray(discounted_return, dtype=float).reshape(-1)
    episode_return = np.asarray(episode_return, dtype=float).reshape(-1)
    success = np.asarray(success, dtype=bool).reshape(-1)
    source = np.asarray(source)
    if not (len(score) == len(discounted_return) == len(episode_return) == len(success) == len(source)):
        raise ValueError("Ranking arrays have different episode counts")
    if not np.all(np.isfinite(score)):
        raise ValueError("Nonfinite ranking score")
    result = {"episodes": len(score), "success_prevalence": float(success.mean()),
              "pearson_discounted_return": safe_corr(score, discounted_return, "pearson"),
              "spearman_discounted_return": safe_corr(score, discounted_return, "spearman"),
              "pearson_episode_return": safe_corr(score, episode_return, "pearson"),
              "spearman_episode_return": safe_corr(score, episode_return, "spearman"),
              "success_pairwise_accuracy": success_ranking_accuracy(score, success)}
    order = np.argsort(-score, kind="stable")
    for fraction in (10, 20, 50):
        top = order[:max(1, round(len(score) * fraction / 100))]
        result[f"top{fraction}_success"] = float(success[top].mean())
        result[f"top{fraction}_mean_return"] = float(episode_return[top].mean())
    within_auc = []
    within_spearman = []
    for name in np.unique(source):
        mask = source == name
        auc = success_ranking_accuracy(score[mask], success[mask])
        rho = safe_corr(score[mask], discounted_return[mask], "spearman")
        if np.isfinite(auc):
            within_auc.append(auc)
        if np.isfinite(rho):
            within_spearman.append(rho)
    result["mixed_success_source_groups"] = len(within_auc)
    result["within_source_pairwise_accuracy_mean"] = float(np.mean(within_auc)) if within_auc else float("nan")
    result["within_source_spearman_mean"] = float(np.mean(within_spearman)) if within_spearman else float("nan")
    return result
