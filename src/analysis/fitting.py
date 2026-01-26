"""
Curve fitting utilities for behavioral analysis.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass
class FitResult:
    """Result from a curve fitting operation."""

    model_name: str
    parameters: dict
    r_squared: float
    residuals: np.ndarray
    predicted: np.ndarray
    success: bool
    error_message: str | None = None


def fit_exponential_decay(
    x: np.ndarray,
    y: np.ndarray,
    p0: tuple[float, float, float] | None = None,
) -> FitResult:
    """
    Fit exponential decay: y = a * exp(-b * x) + c

    Args:
        x: Independent variable (e.g., time)
        y: Dependent variable (e.g., action intensity)
        p0: Initial parameters (a, b, c)

    Returns:
        FitResult with fitted parameters and diagnostics
    """
    if len(x) < 3 or len(y) < 3:
        return FitResult(
            model_name="exponential_decay",
            parameters={},
            r_squared=0.0,
            residuals=np.array([]),
            predicted=np.array([]),
            success=False,
            error_message="Insufficient data points",
        )

    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c

    # Default initial parameters
    if p0 is None:
        a0 = y[0] - y[-1] if y[0] > y[-1] else 1.0
        b0 = 1.0 / (x[-1] - x[0]) if x[-1] != x[0] else 0.1
        c0 = y[-1]
        p0 = (a0, b0, c0)

    try:
        popt, pcov = optimize.curve_fit(
            exp_decay, x, y, p0=p0,
            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=5000,
        )

        predicted = exp_decay(x, *popt)
        residuals = y - predicted

        # Compute R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return FitResult(
            model_name="exponential_decay",
            parameters={"a": popt[0], "b": popt[1], "c": popt[2]},
            r_squared=r_squared,
            residuals=residuals,
            predicted=predicted,
            success=True,
        )

    except (RuntimeError, ValueError, optimize.OptimizeWarning) as e:
        return FitResult(
            model_name="exponential_decay",
            parameters={},
            r_squared=0.0,
            residuals=np.array([]),
            predicted=np.array([]),
            success=False,
            error_message=str(e),
        )


def fit_power_law_decay(
    x: np.ndarray,
    y: np.ndarray,
    p0: tuple[float, float, float] | None = None,
) -> FitResult:
    """
    Fit power law decay: y = a * (x + 1)^(-b) + c

    Power law decay is slower than exponential and may indicate
    more persistent seeking behavior.

    Args:
        x: Independent variable (e.g., time)
        y: Dependent variable (e.g., action intensity)
        p0: Initial parameters (a, b, c)

    Returns:
        FitResult with fitted parameters and diagnostics
    """
    if len(x) < 3 or len(y) < 3:
        return FitResult(
            model_name="power_law_decay",
            parameters={},
            r_squared=0.0,
            residuals=np.array([]),
            predicted=np.array([]),
            success=False,
            error_message="Insufficient data points",
        )

    def power_decay(t, a, b, c):
        return a * np.power(t + 1, -b) + c

    # Default initial parameters
    if p0 is None:
        a0 = y[0] - y[-1] if y[0] > y[-1] else 1.0
        b0 = 1.0
        c0 = y[-1]
        p0 = (a0, b0, c0)

    try:
        popt, pcov = optimize.curve_fit(
            power_decay, x, y, p0=p0,
            bounds=([0, 0, -np.inf], [np.inf, 10, np.inf]),
            maxfev=5000,
        )

        predicted = power_decay(x, *popt)
        residuals = y - predicted

        # Compute R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return FitResult(
            model_name="power_law_decay",
            parameters={"a": popt[0], "b": popt[1], "c": popt[2]},
            r_squared=r_squared,
            residuals=residuals,
            predicted=predicted,
            success=True,
        )

    except (RuntimeError, ValueError, optimize.OptimizeWarning) as e:
        return FitResult(
            model_name="power_law_decay",
            parameters={},
            r_squared=0.0,
            residuals=np.array([]),
            predicted=np.array([]),
            success=False,
            error_message=str(e),
        )


def fit_linear(x: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Fit linear model: y = a * x + b

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        FitResult with fitted parameters
    """
    if len(x) < 2 or len(y) < 2:
        return FitResult(
            model_name="linear",
            parameters={},
            r_squared=0.0,
            residuals=np.array([]),
            predicted=np.array([]),
            success=False,
            error_message="Insufficient data points",
        )

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        predicted = slope * x + intercept
        residuals = y - predicted

        return FitResult(
            model_name="linear",
            parameters={
                "slope": slope,
                "intercept": intercept,
                "p_value": p_value,
                "std_err": std_err,
            },
            r_squared=r_value ** 2,
            residuals=residuals,
            predicted=predicted,
            success=True,
        )

    except Exception as e:
        return FitResult(
            model_name="linear",
            parameters={},
            r_squared=0.0,
            residuals=np.array([]),
            predicted=np.array([]),
            success=False,
            error_message=str(e),
        )


def compare_fits(
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Compare exponential vs power law decay fits.

    This comparison is important for extinction behavior analysis:
    - Exponential decay suggests reward-driven behavior
    - Power law decay may suggest desire-like persistence

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        Dictionary with comparison results
    """
    exp_fit = fit_exponential_decay(x, y)
    power_fit = fit_power_law_decay(x, y)
    linear_fit = fit_linear(x, y)

    # Determine best fit by R-squared
    fits = [
        ("exponential", exp_fit),
        ("power_law", power_fit),
        ("linear", linear_fit),
    ]

    successful_fits = [(name, fit) for name, fit in fits if fit.success]

    if not successful_fits:
        return {
            "best_model": None,
            "exponential": exp_fit,
            "power_law": power_fit,
            "linear": linear_fit,
            "comparison": "No successful fits",
        }

    best_name, best_fit = max(successful_fits, key=lambda x: x[1].r_squared)

    # AIC-like comparison (simplified, using residual sum of squares)
    def aic_proxy(fit: FitResult, n_params: int) -> float:
        if not fit.success or len(fit.residuals) == 0:
            return float('inf')
        n = len(fit.residuals)
        rss = np.sum(fit.residuals ** 2)
        return n * np.log(rss / n) + 2 * n_params

    aic_exp = aic_proxy(exp_fit, 3)
    aic_power = aic_proxy(power_fit, 3)
    aic_linear = aic_proxy(linear_fit, 2)

    return {
        "best_model": best_name,
        "best_r_squared": best_fit.r_squared,
        "exponential": {
            "r_squared": exp_fit.r_squared,
            "parameters": exp_fit.parameters,
            "aic_proxy": aic_exp,
            "success": exp_fit.success,
        },
        "power_law": {
            "r_squared": power_fit.r_squared,
            "parameters": power_fit.parameters,
            "aic_proxy": aic_power,
            "success": power_fit.success,
        },
        "linear": {
            "r_squared": linear_fit.r_squared,
            "parameters": linear_fit.parameters,
            "aic_proxy": aic_linear,
            "success": linear_fit.success,
        },
        "decay_type_indicator": _classify_decay_type(exp_fit, power_fit),
    }


def _classify_decay_type(
    exp_fit: FitResult,
    power_fit: FitResult,
) -> str:
    """
    Classify the decay type based on fit comparison.

    Returns interpretation of which decay pattern better fits the data.
    """
    if not exp_fit.success and not power_fit.success:
        return "undetermined"

    if not exp_fit.success:
        return "power_law_only"

    if not power_fit.success:
        return "exponential_only"

    # Compare R-squared values
    r2_diff = power_fit.r_squared - exp_fit.r_squared

    if r2_diff > 0.05:
        return "power_law_preferred"
    elif r2_diff < -0.05:
        return "exponential_preferred"
    else:
        return "indeterminate"


def compute_half_life(fit: FitResult) -> float | None:
    """
    Compute the half-life from a decay fit.

    Half-life is the time for the value to decay to 50% of initial.

    Args:
        fit: FitResult from exponential or power law fit

    Returns:
        Half-life value, or None if cannot be computed
    """
    if not fit.success:
        return None

    if fit.model_name == "exponential_decay":
        b = fit.parameters.get("b", 0)
        if b > 0:
            return np.log(2) / b
        return None

    elif fit.model_name == "power_law_decay":
        b = fit.parameters.get("b", 0)
        if b > 0:
            # For power law: 0.5 = (t + 1)^(-b)
            # t = 2^(1/b) - 1
            return np.power(2, 1 / b) - 1
        return None

    return None
