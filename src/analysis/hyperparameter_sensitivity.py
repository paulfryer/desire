"""
Hyperparameter sensitivity analysis for experimental robustness.

Provides tools for systematic evaluation of how results change
with different hyperparameter settings.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.tests.base_test import BasePredicateTest


@dataclass
class SensitivityResult:
    """Result from sensitivity analysis."""

    parameter_name: str
    parameter_values: list[Any]
    scores: list[float]
    baseline_value: Any
    baseline_score: float
    sensitivity_coefficient: float  # d(score)/d(param)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "parameter_values": self.parameter_values,
            "scores": self.scores,
            "baseline_value": self.baseline_value,
            "baseline_score": self.baseline_score,
            "sensitivity_coefficient": self.sensitivity_coefficient,
        }


@dataclass
class GridSearchResult:
    """Result from grid search sensitivity analysis."""

    parameter_grid: dict[str, list[Any]]
    results: list[dict[str, Any]]
    best_params: dict[str, Any]
    best_score: float
    worst_params: dict[str, Any]
    worst_score: float
    score_variance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_grid": self.parameter_grid,
            "n_combinations": len(self.results),
            "best_params": self.best_params,
            "best_score": self.best_score,
            "worst_params": self.worst_params,
            "worst_score": self.worst_score,
            "score_variance": self.score_variance,
        }


class HyperparameterSensitivityAnalysis:
    """
    Hyperparameter sensitivity analysis for experimental robustness.

    Evaluates how sensitive results are to:
    - Number of trials
    - Episode lengths
    - Test-specific parameters
    - Agent hyperparameters
    """

    def __init__(
        self,
        baseline_config: dict[str, Any],
        seed: int | None = None,
    ):
        """
        Initialize sensitivity analysis.

        Args:
            baseline_config: Baseline configuration
            seed: Random seed
        """
        self.baseline_config = baseline_config
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def analyze_single_parameter(
        self,
        test_fn: Callable[[dict[str, Any]], float],
        parameter_name: str,
        parameter_values: list[Any],
    ) -> SensitivityResult:
        """
        Analyze sensitivity to a single parameter.

        Args:
            test_fn: Function that takes config and returns score
            parameter_name: Name of parameter to vary
            parameter_values: Values to test

        Returns:
            SensitivityResult
        """
        scores = []

        for value in parameter_values:
            config = self.baseline_config.copy()
            config[parameter_name] = value

            score = test_fn(config)
            scores.append(score)

        # Find baseline
        baseline_idx = None
        baseline_value = self.baseline_config.get(parameter_name)
        if baseline_value in parameter_values:
            baseline_idx = parameter_values.index(baseline_value)
            baseline_score = scores[baseline_idx]
        else:
            baseline_score = np.mean(scores)

        # Compute sensitivity coefficient (approximate derivative)
        if len(parameter_values) > 1:
            try:
                param_array = np.array([float(v) for v in parameter_values])
                score_array = np.array(scores)
                sensitivity = np.polyfit(param_array, score_array, 1)[0]
            except (ValueError, TypeError):
                # Non-numeric parameters
                sensitivity = np.std(scores)
        else:
            sensitivity = 0.0

        return SensitivityResult(
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            scores=scores,
            baseline_value=baseline_value,
            baseline_score=float(baseline_score),
            sensitivity_coefficient=float(sensitivity),
        )

    def grid_search_sensitivity(
        self,
        test_fn: Callable[[dict[str, Any]], float],
        parameter_grid: dict[str, list[Any]],
        n_repeats: int = 1,
    ) -> GridSearchResult:
        """
        Run grid search over parameter combinations.

        Args:
            test_fn: Function that takes config and returns score
            parameter_grid: Dictionary of parameter -> values
            n_repeats: Repeats per combination

        Returns:
            GridSearchResult
        """
        # Generate all combinations
        param_names = list(parameter_grid.keys())
        param_values = [parameter_grid[name] for name in param_names]
        combinations = list(product(*param_values))

        results = []

        for combo in combinations:
            config = self.baseline_config.copy()
            for name, value in zip(param_names, combo, strict=False):
                config[name] = value

            # Run with repeats
            scores = []
            for _ in range(n_repeats):
                score = test_fn(config)
                scores.append(score)

            mean_score = np.mean(scores)
            std_score = np.std(scores) if n_repeats > 1 else 0.0

            results.append({
                "params": dict(zip(param_names, combo, strict=False)),
                "mean_score": float(mean_score),
                "std_score": float(std_score),
            })

        # Find best and worst
        sorted_results = sorted(results, key=lambda x: x["mean_score"], reverse=True)
        best = sorted_results[0]
        worst = sorted_results[-1]

        all_scores = [r["mean_score"] for r in results]
        score_variance = float(np.var(all_scores))

        return GridSearchResult(
            parameter_grid=parameter_grid,
            results=results,
            best_params=best["params"],
            best_score=best["mean_score"],
            worst_params=worst["params"],
            worst_score=worst["mean_score"],
            score_variance=score_variance,
        )

    def bootstrap_sensitivity(
        self,
        test_fn: Callable[[dict[str, Any]], float],
        n_bootstrap: int = 100,
    ) -> dict[str, float]:
        """
        Bootstrap sensitivity to random seed.

        Args:
            test_fn: Function that takes config and returns score
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with bootstrap statistics
        """
        scores = []

        for i in range(n_bootstrap):
            config = self.baseline_config.copy()
            config["seed"] = self.seed + i if self.seed else i

            score = test_fn(config)
            scores.append(score)

        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "ci_lower": float(np.percentile(scores, 2.5)),
            "ci_upper": float(np.percentile(scores, 97.5)),
            "n_samples": n_bootstrap,
        }


def run_sensitivity_analysis(
    test: BasePredicateTest,
    agent: BaseAgent,
    parameters_to_test: dict[str, list[Any]],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run comprehensive sensitivity analysis.

    Args:
        test: Predicate test to analyze
        agent: Agent to test
        parameters_to_test: Parameters and their values
        output_dir: Directory to save results

    Returns:
        Dictionary of analysis results
    """
    baseline_config = {
        "n_trials": test.n_trials,
        "baseline_episodes": test.baseline_episodes,
        "test_episodes": test.test_episodes,
        "seed": test.seed,
    }

    analysis = HyperparameterSensitivityAnalysis(baseline_config)

    def run_test(config: dict[str, Any]) -> float:
        test.n_trials = config.get("n_trials", test.n_trials)
        test.baseline_episodes = config.get("baseline_episodes", test.baseline_episodes)
        test.test_episodes = config.get("test_episodes", test.test_episodes)
        test.seed = config.get("seed", test.seed)

        result = test.run(agent)
        return result.score

    results = {
        "test_name": test.name,
        "agent_name": agent.name,
        "baseline_config": baseline_config,
        "single_parameter_analysis": {},
        "grid_search": None,
    }

    # Single parameter analysis
    for param_name, param_values in parameters_to_test.items():
        sensitivity_result = analysis.analyze_single_parameter(
            run_test, param_name, param_values
        )
        results["single_parameter_analysis"][param_name] = sensitivity_result.to_dict()

    # Grid search if multiple parameters
    if len(parameters_to_test) > 1:
        grid_result = analysis.grid_search_sensitivity(run_test, parameters_to_test)
        results["grid_search"] = grid_result.to_dict()

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"sensitivity_{test.name}_{agent.name}.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def analyze_sample_size_requirements(
    test: BasePredicateTest,
    agent: BaseAgent,
    trial_sizes: list[int] = None,
    episode_sizes: list[int] = None,
    n_repeats: int = 5,
) -> dict[str, Any]:
    """
    Analyze sample size requirements for stable results.

    Args:
        test: Predicate test
        agent: Agent to test
        trial_sizes: Trial counts to test
        episode_sizes: Episode counts to test
        n_repeats: Repeats per size

    Returns:
        Sample size analysis results
    """
    if trial_sizes is None:
        trial_sizes = [5, 10, 20, 50]
    if episode_sizes is None:
        episode_sizes = [10, 20, 50, 100]

    results = {
        "trial_size_analysis": [],
        "episode_size_analysis": [],
    }

    # Analyze trial size
    for n_trials in trial_sizes:
        scores = []
        for _ in range(n_repeats):
            test.n_trials = n_trials
            result = test.run(agent)
            scores.append(result.score)

        results["trial_size_analysis"].append({
            "n_trials": n_trials,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "cv": float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0,
        })

    # Analyze episode size
    for n_episodes in episode_sizes:
        scores = []
        for _ in range(n_repeats):
            test.baseline_episodes = n_episodes
            test.test_episodes = n_episodes
            result = test.run(agent)
            scores.append(result.score)

        results["episode_size_analysis"].append({
            "n_episodes": n_episodes,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "cv": float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0,
        })

    return results
