"""
Seed-level statistics (docs/RESEARCH_DESIGN_v2.md §7). The trained seed is the unit of
replication: every function here takes one score per seed.
"""

import numpy as np


def bootstrap_ci(scores, n_resamples: int = 10_000, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI of the mean over seeds."""
    x = np.asarray(scores, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_resamples, x.size), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def permutation_test(a, b, n_resamples: int = 20_000, seed: int = 0) -> float:
    """Two-sided permutation test for a difference in means between two sets of seed scores."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_resamples):
        perm = rng.permutation(pooled)
        count += abs(perm[:a.size].mean() - perm[a.size:].mean()) >= observed - 1e-12
    return float((count + 1) / (n_resamples + 1))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    names = sorted(p_values, key=p_values.get)
    m = len(names)
    adjusted, running = {}, 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i) * p_values[name]))
        adjusted[name] = running
    return adjusted


def verdict(scores, pass_mean: float = 0.5, fail_mean: float = 0.2, seed: int = 0) -> str:
    """Pre-registered rule: pass if mean >= 0.5 and CI excludes 0; fail if CI includes 0 or
    mean < 0.2; otherwise partial."""
    mean, lo, hi = bootstrap_ci(scores, seed=seed)
    if np.isnan(mean):
        return "n/a"
    excludes_zero = lo > 0 or hi < 0
    if mean >= pass_mean and excludes_zero:
        return "pass"
    if not excludes_zero or mean < fail_mean:
        return "fail"
    return "partial"
