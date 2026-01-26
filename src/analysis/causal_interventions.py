"""
Causal intervention analysis for understanding agent behavior.

Provides tools for ablation studies, state interventions, and
counterfactual analysis to understand causal mechanisms.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.utils.logging import EpisodeLog, StepLog


@dataclass
class InterventionResult:
    """Result from a causal intervention."""

    intervention_type: str
    baseline_score: float
    intervened_score: float
    effect_size: float
    significance: Optional[float] = None
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_type": self.intervention_type,
            "baseline_score": self.baseline_score,
            "intervened_score": self.intervened_score,
            "effect_size": self.effect_size,
            "significance": self.significance,
            "details": self.details or {},
        }


class CausalInterventionAnalysis:
    """
    Causal intervention analysis for understanding agent representations.

    Supports:
    - Ablation: Zeroing out components
    - Amplification: Scaling up components
    - Noise injection: Adding noise to components
    - State swapping: Replacing components from different conditions
    """

    def __init__(
        self,
        env: ForagingGridEnv,
        n_episodes: int = 20,
        seed: Optional[int] = None,
    ):
        """
        Initialize causal intervention analysis.

        Args:
            env: Environment for testing
            n_episodes: Episodes per condition
            seed: Random seed
        """
        self.env = env
        self.n_episodes = n_episodes
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def intervene_on_internal_state(
        self,
        agent: BaseAgent,
        intervention_fn: Callable[[np.ndarray], np.ndarray],
        intervention_name: str = "custom",
    ) -> InterventionResult:
        """
        Apply intervention to agent's internal state.

        Args:
            agent: Agent to intervene on
            intervention_fn: Function that transforms internal state
            intervention_name: Name of the intervention

        Returns:
            InterventionResult with baseline and intervention scores
        """
        # Run baseline episodes
        baseline_rewards = self._run_episodes(agent, intervene=False)

        # Run intervention episodes
        intervention_rewards = self._run_episodes_with_intervention(
            agent, intervention_fn
        )

        baseline_mean = np.mean(baseline_rewards)
        intervention_mean = np.mean(intervention_rewards)

        effect_size = intervention_mean - baseline_mean

        return InterventionResult(
            intervention_type=intervention_name,
            baseline_score=float(baseline_mean),
            intervened_score=float(intervention_mean),
            effect_size=float(effect_size),
            details={
                "baseline_std": float(np.std(baseline_rewards)),
                "intervention_std": float(np.std(intervention_rewards)),
                "n_episodes": self.n_episodes,
            },
        )

    def _run_episodes(
        self,
        agent: BaseAgent,
        intervene: bool = False,
    ) -> List[float]:
        """Run episodes and collect rewards."""
        rewards = []

        for ep in range(self.n_episodes):
            seed = self.seed + ep if self.seed else None
            obs, _ = self.env.reset(seed=seed)
            agent.reset()

            total_reward = 0.0

            for _ in range(self.env.max_steps):
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

                obs = next_obs

            rewards.append(total_reward)

        return rewards

    def _run_episodes_with_intervention(
        self,
        agent: BaseAgent,
        intervention_fn: Callable[[np.ndarray], np.ndarray],
    ) -> List[float]:
        """Run episodes with intervention on internal state."""
        rewards = []

        # Store original get_internal_state method
        original_method = agent.get_internal_state

        def intervened_get_internal_state():
            state = original_method()
            return intervention_fn(state)

        # Monkey-patch the method
        agent.get_internal_state = intervened_get_internal_state

        try:
            for ep in range(self.n_episodes):
                seed = self.seed + ep if self.seed else None
                obs, _ = self.env.reset(seed=seed)
                agent.reset()

                total_reward = 0.0

                for _ in range(self.env.max_steps):
                    action = agent.act(obs)
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    total_reward += reward

                    if terminated or truncated:
                        break

                    obs = next_obs

                rewards.append(total_reward)
        finally:
            # Restore original method
            agent.get_internal_state = original_method

        return rewards

    def ablation_analysis(
        self,
        agent: BaseAgent,
        component_indices: Optional[List[int]] = None,
    ) -> List[InterventionResult]:
        """
        Perform ablation analysis by zeroing components.

        Args:
            agent: Agent to analyze
            component_indices: Specific components to ablate (default: all)

        Returns:
            List of intervention results
        """
        # Get internal state dimension
        sample_state = agent.get_internal_state()
        n_components = len(sample_state)

        if component_indices is None:
            component_indices = list(range(n_components))

        results = []

        for idx in component_indices:
            def ablate_component(state, i=idx):
                ablated = state.copy()
                ablated[i] = 0.0
                return ablated

            result = self.intervene_on_internal_state(
                agent, ablate_component, f"ablate_component_{idx}"
            )
            results.append(result)

        return results

    def noise_injection_analysis(
        self,
        agent: BaseAgent,
        noise_levels: List[float] = None,
    ) -> List[InterventionResult]:
        """
        Analyze robustness to noise in internal state.

        Args:
            agent: Agent to analyze
            noise_levels: Noise standard deviations to test

        Returns:
            List of intervention results
        """
        if noise_levels is None:
            noise_levels = [0.1, 0.5, 1.0, 2.0]

        results = []

        for noise_std in noise_levels:
            def add_noise(state, std=noise_std):
                noise = self._rng.normal(0, std, state.shape)
                return state + noise

            result = self.intervene_on_internal_state(
                agent, add_noise, f"noise_std_{noise_std}"
            )
            results.append(result)

        return results

    def amplification_analysis(
        self,
        agent: BaseAgent,
        scales: List[float] = None,
    ) -> List[InterventionResult]:
        """
        Analyze effect of amplifying internal state.

        Args:
            agent: Agent to analyze
            scales: Scaling factors to test

        Returns:
            List of intervention results
        """
        if scales is None:
            scales = [0.5, 1.5, 2.0, 5.0]

        results = []

        for scale in scales:
            def amplify(state, s=scale):
                return state * s

            result = self.intervene_on_internal_state(
                agent, amplify, f"amplify_{scale}x"
            )
            results.append(result)

        return results

    def identify_critical_components(
        self,
        agent: BaseAgent,
        n_components: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Identify most critical components through systematic ablation.

        Args:
            agent: Agent to analyze
            n_components: Number of components to identify

        Returns:
            List of (component_idx, importance_score) tuples
        """
        # Run ablation on all components
        results = self.ablation_analysis(agent)

        # Compute importance scores (larger negative effect = more important)
        importance_scores = []
        for idx, result in enumerate(results):
            importance = -result.effect_size  # Negative effect means important
            importance_scores.append((idx, importance))

        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)

        return importance_scores[:n_components]

    def test_counterfactual_activation(
        self,
        agent: BaseAgent,
        observation: Dict[str, np.ndarray],
        goal_distances: List[float],
    ) -> Dict[str, List[float]]:
        """
        Test counterfactual: what if agent were at different distances?

        Args:
            agent: Agent to analyze
            observation: Base observation
            goal_distances: Distances to simulate

        Returns:
            Dictionary with activation at each distance
        """
        activations = []
        agent.reset()

        for distance in goal_distances:
            # Modify observation to simulate different distances
            modified_obs = self._simulate_distance(observation, distance)

            # Get agent's internal response
            response = agent.get_internal_response(modified_obs)
            activation = np.linalg.norm(response)
            activations.append(activation)

        return {
            "distances": goal_distances,
            "activations": activations,
            "correlation": float(np.corrcoef(goal_distances, activations)[0, 1]),
        }

    def _simulate_distance(
        self,
        observation: Dict[str, np.ndarray],
        target_distance: float,
    ) -> Dict[str, np.ndarray]:
        """Modify observation to simulate target distance (simplified)."""
        # This is a simplified simulation - in practice would need
        # more sophisticated observation manipulation
        modified = {k: v.copy() for k, v in observation.items()}
        # Could modify local_view to simulate different resource distances
        return modified


def run_causal_analysis(
    agent: BaseAgent,
    env: ForagingGridEnv,
    n_episodes: int = 20,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive causal analysis on an agent.

    Args:
        agent: Agent to analyze
        env: Environment
        n_episodes: Episodes per condition
        seed: Random seed

    Returns:
        Dictionary of analysis results
    """
    analysis = CausalInterventionAnalysis(
        env=env,
        n_episodes=n_episodes,
        seed=seed,
    )

    results = {
        "agent_name": agent.name,
    }

    # Ablation analysis
    print("Running ablation analysis...")
    ablation_results = analysis.ablation_analysis(agent)
    results["ablation"] = [r.to_dict() for r in ablation_results]

    # Noise injection
    print("Running noise injection analysis...")
    noise_results = analysis.noise_injection_analysis(agent)
    results["noise_injection"] = [r.to_dict() for r in noise_results]

    # Identify critical components
    print("Identifying critical components...")
    critical = analysis.identify_critical_components(agent)
    results["critical_components"] = critical

    return results
