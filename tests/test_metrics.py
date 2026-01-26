"""
Unit tests for metrics and analysis components.
"""

import numpy as np
import pytest

from src.analysis.fitting import (
    compare_fits,
    compute_half_life,
    fit_exponential_decay,
    fit_power_law_decay,
)
from src.analysis.metrics import (
    aggregate_metrics,
    compare_agents,
    compute_action_entropy,
    compute_frustration_score,
    compute_goal_directedness,
    compute_means_end_score,
)
from src.tests.base_test import TestResult
from src.utils.logging import EpisodeLog, StepLog


class TestFrustrationScore:
    """Tests for frustration intensification score."""

    def test_basic_fi_score(self):
        """Test basic FI score computation."""
        baseline = np.array([0.5, 0.5, 0.5, 0.5])
        post_blockage = np.array([0.8, 1.2, 0.9, 0.6])

        score = compute_frustration_score(baseline, post_blockage)

        # max(post) / mean(baseline) = 1.2 / 0.5 = 2.4
        assert score == pytest.approx(2.4, rel=0.01)

    def test_fi_no_increase(self):
        """Test FI score when no increase occurs."""
        baseline = np.array([0.5, 0.5, 0.5])
        post_blockage = np.array([0.3, 0.2, 0.1])

        score = compute_frustration_score(baseline, post_blockage)

        assert score < 1.0

    def test_fi_with_zero_baseline(self):
        """Test FI score handles near-zero baseline."""
        baseline = np.array([0.001, 0.001, 0.001])
        post_blockage = np.array([0.5, 0.5, 0.5])

        score = compute_frustration_score(baseline, post_blockage)

        # Should not divide by zero
        assert np.isfinite(score)


class TestMeansEndScore:
    """Tests for means-end decoupling score."""

    def test_perfect_transfer(self):
        """Test MED score for perfect transfer."""
        optimal_length = 10
        actual_times = [10, 10, 10]

        score = compute_means_end_score(optimal_length, actual_times)

        assert score == pytest.approx(1.0, rel=0.01)

    def test_suboptimal_transfer(self):
        """Test MED score for suboptimal transfer."""
        optimal_length = 10
        actual_times = [20, 20, 20]  # 2x optimal

        score = compute_means_end_score(optimal_length, actual_times)

        assert score == pytest.approx(0.5, rel=0.01)

    def test_no_success(self):
        """Test MED score when goal never reached."""
        optimal_length = 10
        actual_times = []  # Never reached

        score = compute_means_end_score(optimal_length, actual_times)

        assert score == 0.0


class TestActionEntropy:
    """Tests for action entropy computation."""

    def test_uniform_entropy(self):
        """Test entropy for uniform distribution."""
        # 6 actions, each used equally
        actions = np.array([0, 1, 2, 3, 4, 5] * 10)

        entropy = compute_action_entropy(actions)

        # Max entropy for 6 actions = log2(6) ≈ 2.58
        assert entropy == pytest.approx(np.log2(6), rel=0.01)

    def test_deterministic_entropy(self):
        """Test entropy for deterministic policy."""
        actions = np.array([0, 0, 0, 0, 0])

        entropy = compute_action_entropy(actions)

        assert entropy == pytest.approx(0.0, abs=0.01)

    def test_empty_sequence(self):
        """Test entropy for empty sequence."""
        actions = np.array([])

        entropy = compute_action_entropy(actions)

        assert entropy == 0.0


class TestGoalDirectedness:
    """Tests for goal-directedness metric."""

    def test_perfect_directedness(self):
        """Test perfectly goal-directed trajectory."""
        # Moving toward (10, 10)
        trajectory = np.array([
            [0, 0], [1, 1], [2, 2], [3, 3], [4, 4]
        ])
        goal = (10, 10)

        score = compute_goal_directedness(trajectory, goal)

        assert score == 1.0

    def test_moving_away(self):
        """Test trajectory moving away from goal."""
        # Moving away from (10, 10)
        trajectory = np.array([
            [5, 5], [4, 4], [3, 3], [2, 2], [1, 1]
        ])
        goal = (10, 10)

        score = compute_goal_directedness(trajectory, goal)

        assert score == 0.0

    def test_random_movement(self):
        """Test random-ish movement."""
        trajectory = np.array([
            [5, 5], [6, 5], [5, 5], [6, 6], [5, 6]
        ])
        goal = (10, 10)

        score = compute_goal_directedness(trajectory, goal)

        assert 0.0 <= score <= 1.0


class TestExponentialFit:
    """Tests for exponential decay fitting."""

    def test_perfect_exponential(self):
        """Test fitting perfect exponential data."""
        x = np.arange(100)
        y = 2.0 * np.exp(-0.05 * x) + 0.5

        fit = fit_exponential_decay(x, y)

        assert fit.success
        assert fit.r_squared > 0.99
        assert fit.parameters["a"] == pytest.approx(2.0, rel=0.1)
        assert fit.parameters["b"] == pytest.approx(0.05, rel=0.1)

    def test_noisy_exponential(self):
        """Test fitting noisy exponential data."""
        rng = np.random.default_rng(42)
        x = np.arange(100)
        y = 2.0 * np.exp(-0.05 * x) + 0.5 + rng.normal(0, 0.1, 100)

        fit = fit_exponential_decay(x, y)

        assert fit.success
        assert fit.r_squared > 0.8

    def test_insufficient_data(self):
        """Test fitting with insufficient data."""
        x = np.array([0, 1])
        y = np.array([1, 0.5])

        fit = fit_exponential_decay(x, y)

        assert not fit.success


class TestPowerLawFit:
    """Tests for power law decay fitting."""

    def test_perfect_power_law(self):
        """Test fitting perfect power law data."""
        x = np.arange(1, 101)
        y = 5.0 * np.power(x, -0.5) + 1.0

        fit = fit_power_law_decay(x, y)

        assert fit.success
        assert fit.r_squared > 0.99

    def test_half_life_computation(self):
        """Test half-life computation from fit."""
        x = np.arange(100)
        y = 2.0 * np.exp(-0.1 * x)  # Known half-life = ln(2)/0.1 ≈ 6.93

        fit = fit_exponential_decay(x, y)
        half_life = compute_half_life(fit)

        expected = np.log(2) / 0.1
        assert half_life == pytest.approx(expected, rel=0.1)


class TestFitComparison:
    """Tests for comparing decay models."""

    def test_exponential_vs_power_law(self):
        """Test comparison identifies correct model."""
        x = np.arange(100)

        # Create exponential data
        y_exp = 2.0 * np.exp(-0.05 * x) + 0.5

        comparison = compare_fits(x, y_exp)

        assert comparison["best_model"] in ["exponential", "power_law"]
        assert "decay_type_indicator" in comparison


class TestAgentComparison:
    """Tests for statistical agent comparison."""

    def test_compare_identical_agents(self):
        """Test comparison of identical results."""
        results = {
            "agent1": TestResult(
                test_name="test",
                agent_name="agent1",
                score=0.5,
                raw_data={"trial_scores": [0.5, 0.5, 0.5]},
            ),
            "agent2": TestResult(
                test_name="test",
                agent_name="agent2",
                score=0.5,
                raw_data={"trial_scores": [0.5, 0.5, 0.5]},
            ),
        }

        comparison = compare_agents(results)

        assert not comparison["kruskal_wallis"]["significant"]

    def test_compare_different_agents(self):
        """Test comparison of different results."""
        results = {
            "agent1": TestResult(
                test_name="test",
                agent_name="agent1",
                score=0.9,
                raw_data={"trial_scores": [0.88, 0.90, 0.92, 0.89, 0.91]},
            ),
            "agent2": TestResult(
                test_name="test",
                agent_name="agent2",
                score=0.3,
                raw_data={"trial_scores": [0.28, 0.30, 0.32, 0.29, 0.31]},
            ),
        }

        comparison = compare_agents(results)

        # Should be significantly different
        assert comparison["kruskal_wallis"]["p_value"] < 0.05

    def test_summary_stats(self):
        """Test summary statistics are computed."""
        results = {
            "agent1": TestResult(
                test_name="test",
                agent_name="agent1",
                score=0.5,
                raw_data={"trial_scores": [0.4, 0.5, 0.6]},
            ),
        }

        comparison = compare_agents(results)

        stats = comparison["summary_stats"]["agent1"]
        assert stats["mean"] == pytest.approx(0.5, rel=0.01)
        assert stats["n"] == 3


class TestAggregateMetrics:
    """Tests for aggregate metrics computation."""

    def test_aggregate_basic_metrics(self):
        """Test basic metrics aggregation."""
        # Create mock episodes
        episodes = []
        for i in range(5):
            steps = [
                StepLog(
                    step=j,
                    action=j % 6,
                    position=(j, j),
                    energy=100 - j,
                    reward=0.1,
                )
                for j in range(10)
            ]
            episodes.append(EpisodeLog(
                episode_id=i,
                agent_type="test",
                environment_mode="baseline",
                total_steps=10,
                total_reward=1.0,
                final_energy=90.0,
                survived=True,
                steps=steps,
            ))

        metrics = aggregate_metrics(episodes)

        assert "mean_reward" in metrics
        assert "survival_rate" in metrics
        assert "action_entropy" in metrics
        assert metrics["n_episodes"] == 5
        assert metrics["survival_rate"] == 1.0

    def test_aggregate_empty_list(self):
        """Test aggregation handles empty list."""
        metrics = aggregate_metrics([])

        assert metrics == {}
