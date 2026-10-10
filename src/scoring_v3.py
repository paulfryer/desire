"""
The pre-registered scoring rule for the held-out-need experiment (docs/HELDOUT_DESIGN.md §6).

One rule, applied to every arm and every control. No seed is excluded. Positive and negative
claims need the same strength of evidence: a quantity counts as "present" when its 95% CI lies
above DELTA, and "small" when its CI lies below DELTA. CIs are Student-t intervals over seeds.
"""

import itertools
import math

import numpy as np
from scipy import stats

DELTA = 0.2


def t_interval(values) -> tuple[float, float, float]:
    """Mean and 95% Student-t interval over seeds."""
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    if x.size < 2 or np.allclose(x, x[0]):
        return mean, mean, mean
    half = stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size)
    return mean, mean - half, mean + half


def heldout_verdict(ri_h, ri_spec_h, ri_dir_h, si_h) -> dict:
    """Zero-shot revaluation verdict for one arm, from per-seed values (§6).

    1. negligible        upper CI of RI_H < DELTA
    2. if lower CI of RI_H > DELTA ("revaluation present"):
         transfers        lower CIs of RI_spec_H, RI_dir_H and SI_H all > 0
         without signature  some check is positively small (upper CI < DELTA); names which
         signature inconclusive   otherwise
    3. inconclusive      otherwise
    """
    _, lo, hi = t_interval(ri_h)
    if hi < DELTA:
        return {"verdict": "negligible", "failed_checks": []}
    if lo <= DELTA:
        return {"verdict": "inconclusive", "failed_checks": []}
    checks = {"specific": ri_spec_h, "directional": ri_dir_h, "seeking": si_h}
    ci = {name: t_interval(v) for name, v in checks.items()}
    if all(c[1] > 0 for c in ci.values()):
        return {"verdict": "transfers", "failed_checks": []}
    failed = [name for name, c in ci.items() if c[2] < DELTA]
    if failed:
        return {"verdict": "revaluation without wanting signature", "failed_checks": failed}
    return {"verdict": "revaluation present, signature inconclusive", "failed_checks": []}


def magnitude_verdict(values, present: str, absent: str) -> str:
    """Generic: 'present' if lower CI > DELTA, 'absent' if upper CI < DELTA, else inconclusive."""
    _, lo, hi = t_interval(values)
    if lo > DELTA:
        return present
    if hi < DELTA:
        return absent
    return "inconclusive"


def trained_verdict(ri_t) -> str:
    return magnitude_verdict(ri_t, "learned", "negligible")


def gap_verdict(delta) -> str:
    """Transfer gap Δ = RI_T - RI_H, per seed."""
    _, lo, hi = t_interval(delta)
    if hi < DELTA:
        return "complete"
    if lo > DELTA:
        return "incomplete"
    return "unclear"


def seed_transfers(ri_h, ri_spec_h, ri_dir_h, si_h) -> bool:
    """Per-seed classification for the secondary transfer count."""
    return ri_h >= 0.5 and ri_spec_h >= 0.5 and ri_dir_h >= 0.5 and si_h >= DELTA


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def sign_flip_test(diffs, n_random: int = 100_000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test of mean(diffs) = 0 (exact for n <= 16)."""
    d = np.asarray(diffs, dtype=float)
    observed = abs(d.mean())
    if d.size <= 16:
        flips = np.array(list(itertools.product((1, -1), repeat=d.size)))
    else:
        flips = np.random.default_rng(seed).choice((1, -1), size=(n_random, d.size))
    null = np.abs((flips * d).mean(axis=1))
    return float((null >= observed - 1e-12).mean())


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - rank) * p))
        adjusted[name] = running
    return adjusted


def brier(prior: dict[str, float], outcome: str) -> float:
    return float(sum((p - (c == outcome)) ** 2 for c, p in prior.items()))
