"""
Statistical power analysis and enhanced statistical methods.

Provides power analysis, bootstrap confidence intervals, and Bayesian statistics
for rigorous evaluation of predicate test results.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class PowerAnalysisResult:
    """Result from power analysis computation."""

    observed_power: float
    required_n_per_group: int
    effect_size: float
    alpha: float
    target_power: float
    achieved_power_at_current_n: float
    current_n_per_group: int
    is_adequately_powered: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_power": self.observed_power,
            "required_n_per_group": self.required_n_per_group,
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "achieved_power_at_current_n": self.achieved_power_at_current_n,
            "current_n_per_group": self.current_n_per_group,
            "is_adequately_powered": self.is_adequately_powered,
        }


@dataclass
class BootstrapCIResult:
    """Result from bootstrap confidence interval computation."""

    estimate: float
    ci_lower: float
    ci_upper: float
    confidence: float
    n_bootstrap: int
    standard_error: float
    bootstrap_distribution: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence": self.confidence,
            "n_bootstrap": self.n_bootstrap,
            "standard_error": self.standard_error,
        }


@dataclass
class BayesFactorResult:
    """Result from Bayes factor computation."""

    bf10: float  # Evidence for H1 over H0
    bf01: float  # Evidence for H0 over H1
    interpretation: str
    posterior_odds: float
    effect_size: float
    prior_scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bf10": self.bf10,
            "bf01": self.bf01,
            "interpretation": self.interpretation,
            "posterior_odds": self.posterior_odds,
            "effect_size": self.effect_size,
            "prior_scale": self.prior_scale,
        }


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Uses pooled standard deviation for independent samples.
    Handles edge cases with zero variance.

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Handle zero variance edge cases
    if var1 == 0 and var2 == 0:
        # Both groups have zero variance
        if np.mean(group1) == np.mean(group2):
            return 0.0
        else:
            # Return a large effect size with sign
            return np.sign(np.mean(group1) - np.mean(group2)) * 10.0

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Avoid division by zero
    if pooled_std < 1e-10:
        pooled_std = 1e-10

    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return float(d)


def compute_power_analysis(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> PowerAnalysisResult:
    """
    Compute statistical power analysis for two-group comparison.

    Uses Cohen's d effect size and normal approximation for power.

    Args:
        group1: First group data
        group2: Second group data
        alpha: Significance level
        target_power: Desired statistical power

    Returns:
        PowerAnalysisResult with power metrics
    """
    n1, n2 = len(group1), len(group2)
    current_n = min(n1, n2)

    # Compute effect size
    d = compute_cohens_d(group1, group2)
    abs_d = abs(d)

    # Handle zero or very small effect sizes
    if abs_d < 0.001:
        return PowerAnalysisResult(
            observed_power=alpha,  # Power equals alpha when effect size is 0
            required_n_per_group=10000,  # Very large N needed
            effect_size=d,
            alpha=alpha,
            target_power=target_power,
            achieved_power_at_current_n=alpha,
            current_n_per_group=current_n,
            is_adequately_powered=False,
        )

    # Critical z-value for alpha (two-tailed)
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    # Compute achieved power at current sample size
    # Using normal approximation for two-sample t-test
    se = np.sqrt(2 / current_n) if current_n > 0 else np.inf
    ncp = abs_d / se  # Non-centrality parameter
    achieved_power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)

    # Compute required N for target power
    z_beta = stats.norm.ppf(target_power)
    required_n = int(np.ceil(2 * ((z_alpha + z_beta) / abs_d) ** 2))
    required_n = max(required_n, 2)  # Minimum 2 per group

    return PowerAnalysisResult(
        observed_power=achieved_power,
        required_n_per_group=required_n,
        effect_size=d,
        alpha=alpha,
        target_power=target_power,
        achieved_power_at_current_n=achieved_power,
        current_n_per_group=current_n,
        is_adequately_powered=achieved_power >= target_power,
    )


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
    return_distribution: bool = False,
) -> BootstrapCIResult:
    """
    Compute bootstrap confidence interval for a statistic.

    Uses the percentile method for CI estimation.

    Args:
        data: Sample data
        statistic_fn: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)
        seed: Random seed for reproducibility
        return_distribution: Whether to return full bootstrap distribution

    Returns:
        BootstrapCIResult with CI bounds and statistics
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    # Point estimate
    estimate = float(statistic_fn(data))

    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample_idx = rng.integers(0, n, size=n)
        resample = data[resample_idx]
        bootstrap_stats[i] = statistic_fn(resample)

    # Percentile CI
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    # Standard error
    se = float(np.std(bootstrap_stats))

    return BootstrapCIResult(
        estimate=estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        standard_error=se,
        bootstrap_distribution=bootstrap_stats if return_distribution else None,
    )


def bootstrap_difference_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapCIResult:
    """
    Compute bootstrap CI for the difference between two groups.

    Args:
        group1: First group data
        group2: Second group data
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed

    Returns:
        BootstrapCIResult for the mean difference
    """
    rng = np.random.default_rng(seed)
    n1, n2 = len(group1), len(group2)

    # Point estimate of difference
    estimate = float(np.mean(group1) - np.mean(group2))

    # Bootstrap resampling
    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample1 = group1[rng.integers(0, n1, size=n1)]
        resample2 = group2[rng.integers(0, n2, size=n2)]
        bootstrap_diffs[i] = np.mean(resample1) - np.mean(resample2)

    # Percentile CI
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2)))

    return BootstrapCIResult(
        estimate=estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        standard_error=float(np.std(bootstrap_diffs)),
    )


def compute_bayes_factor(
    group1: np.ndarray,
    group2: np.ndarray,
    prior_scale: float = 0.707,
) -> BayesFactorResult:
    """
    Compute Bayes factor for two-group comparison.

    Uses JZS (Jeffreys-Zellner-Siow) prior with specified scale.
    Approximates BF using Savage-Dickey density ratio.

    Args:
        group1: First group data
        group2: Second group data
        prior_scale: Scale of Cauchy prior on effect size (default: sqrt(2)/2)

    Returns:
        BayesFactorResult with BF10, BF01, and interpretation
    """
    # Compute effect size and standard error
    d = compute_cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    n = (n1 + n2) / 2  # Average sample size
    se = np.sqrt(1/n1 + 1/n2)

    # Handle edge cases
    if abs(d) < 1e-10:
        return BayesFactorResult(
            bf10=1.0,
            bf01=1.0,
            interpretation="No evidence either way",
            posterior_odds=1.0,
            effect_size=d,
            prior_scale=prior_scale,
        )

    # Approximate BF using t-statistic based approach
    # This is a simplified approximation of the JZS BF
    t_stat = d / se
    n1 + n2 - 2

    # Compute BF10 using BIC approximation
    # BF10 ≈ sqrt(n) * exp(-t^2/2) for large n (simplified)
    # More accurate: use integration over prior

    # Use numerical integration for more accurate BF
    bf10 = _compute_jzs_bf(t_stat, n, prior_scale)

    bf01 = 1 / bf10 if bf10 > 0 else np.inf

    # Interpret BF according to Jeffreys scale
    interpretation = _interpret_bayes_factor(bf10)

    return BayesFactorResult(
        bf10=bf10,
        bf01=bf01,
        interpretation=interpretation,
        posterior_odds=bf10,  # Assuming equal prior odds
        effect_size=d,
        prior_scale=prior_scale,
    )


def _compute_jzs_bf(t_stat: float, n: float, prior_scale: float) -> float:
    """
    Compute JZS Bayes factor using numerical integration.

    BF10 = integral of likelihood ratio weighted by prior.
    """
    from scipy import integrate

    # Likelihood under H1 (integrated over effect size with Cauchy prior)
    def integrand(delta):
        # Cauchy prior on effect size
        prior = stats.cauchy.pdf(delta, loc=0, scale=prior_scale)
        # Likelihood ratio for given delta
        lr = np.exp(-0.5 * n * (t_stat / np.sqrt(n) - delta) ** 2)
        return lr * prior

    # Numerical integration
    try:
        integral, _ = integrate.quad(integrand, -np.inf, np.inf)
    except Exception:
        # Fallback to simpler approximation
        integral = 1.0

    # Likelihood under H0 (delta = 0)
    h0_likelihood = np.exp(-0.5 * t_stat ** 2)

    # BF10 = H1 likelihood / H0 likelihood
    if h0_likelihood > 0:
        bf10 = integral / h0_likelihood
    else:
        bf10 = np.inf if integral > 0 else 1.0

    # Bound BF to reasonable range
    bf10 = np.clip(bf10, 1e-10, 1e10)

    return float(bf10)


def _interpret_bayes_factor(bf10: float) -> str:
    """Interpret Bayes factor according to Jeffreys' scale."""
    if bf10 > 100:
        return "Extreme evidence for H1"
    elif bf10 > 30:
        return "Very strong evidence for H1"
    elif bf10 > 10:
        return "Strong evidence for H1"
    elif bf10 > 3:
        return "Moderate evidence for H1"
    elif bf10 > 1:
        return "Anecdotal evidence for H1"
    elif bf10 > 1/3:
        return "Anecdotal evidence for H0"
    elif bf10 > 1/10:
        return "Moderate evidence for H0"
    elif bf10 > 1/30:
        return "Strong evidence for H0"
    elif bf10 > 1/100:
        return "Very strong evidence for H0"
    else:
        return "Extreme evidence for H0"


def compute_probability_of_superiority(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """
    Compute probability that a random value from group1 exceeds group2.

    P(X1 > X2) - a non-parametric effect size measure.
    Also known as the Common Language Effect Size (CLES).

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Probability in [0, 1]
    """
    n1, n2 = len(group1), len(group2)

    if n1 == 0 or n2 == 0:
        return 0.5

    # Count all pairwise comparisons
    count_superior = 0
    count_ties = 0

    for x1 in group1:
        for x2 in group2:
            if x1 > x2:
                count_superior += 1
            elif x1 == x2:
                count_ties += 1

    # P(X1 > X2) + 0.5 * P(X1 = X2)
    total_pairs = n1 * n2
    prob = (count_superior + 0.5 * count_ties) / total_pairs

    return float(prob)


def compute_hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Hedges' g effect size (bias-corrected Cohen's d).

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Hedges' g effect size
    """
    d = compute_cohens_d(group1, group2)
    n = len(group1) + len(group2)

    # Bias correction factor (approximation for small samples)
    if n <= 2:
        return d

    correction = 1 - 3 / (4 * (n - 2) - 1)
    return float(d * correction)


def compute_glass_delta(
    group1: np.ndarray,
    group2: np.ndarray,
    control_group: int = 2,
) -> float:
    """
    Compute Glass's delta effect size.

    Uses only the control group's standard deviation.

    Args:
        group1: First group data
        group2: Second group data
        control_group: Which group to use as control (1 or 2)

    Returns:
        Glass's delta effect size
    """
    if control_group == 1:
        control_std = np.std(group1, ddof=1)
    else:
        control_std = np.std(group2, ddof=1)

    if control_std < 1e-10:
        control_std = 1e-10

    delta = (np.mean(group1) - np.mean(group2)) / control_std
    return float(delta)


def compute_comprehensive_statistics(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Compute comprehensive statistical comparison between two groups.

    Includes:
    - Traditional frequentist tests (t-test, Mann-Whitney)
    - Effect sizes (Cohen's d, Hedges' g, probability of superiority)
    - Power analysis
    - Bootstrap confidence intervals
    - Bayes factor

    Args:
        group1: First group data
        group2: Second group data
        alpha: Significance level
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Dictionary with all statistical results
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    # Frequentist tests
    t_stat, t_pvalue = stats.ttest_ind(group1, group2)
    u_stat, u_pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    # Effect sizes
    cohens_d = compute_cohens_d(group1, group2)
    hedges_g = compute_hedges_g(group1, group2)
    prob_superiority = compute_probability_of_superiority(group1, group2)

    # Power analysis
    power_result = compute_power_analysis(group1, group2, alpha=alpha)

    # Bootstrap CI for difference
    bootstrap_ci = bootstrap_difference_ci(
        group1, group2, n_bootstrap=n_bootstrap, seed=seed
    )

    # Bayes factor
    bf_result = compute_bayes_factor(group1, group2)

    return {
        "frequentist": {
            "t_test": {
                "statistic": float(t_stat),
                "p_value": float(t_pvalue),
                "significant": t_pvalue < alpha,
            },
            "mann_whitney": {
                "statistic": float(u_stat),
                "p_value": float(u_pvalue),
                "significant": u_pvalue < alpha,
            },
        },
        "effect_sizes": {
            "cohens_d": cohens_d,
            "hedges_g": hedges_g,
            "probability_of_superiority": prob_superiority,
            "interpretation": _interpret_cohens_d(cohens_d),
        },
        "power_analysis": power_result.to_dict(),
        "bootstrap_ci": bootstrap_ci.to_dict(),
        "bayesian": bf_result.to_dict(),
        "descriptive": {
            "group1_mean": float(np.mean(group1)),
            "group1_std": float(np.std(group1)),
            "group1_n": len(group1),
            "group2_mean": float(np.mean(group2)),
            "group2_std": float(np.std(group2)),
            "group2_n": len(group2),
            "mean_difference": float(np.mean(group1) - np.mean(group2)),
        },
    }


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    direction = "positive" if d > 0 else "negative" if d < 0 else "zero"
    return f"{magnitude} {direction} effect"
