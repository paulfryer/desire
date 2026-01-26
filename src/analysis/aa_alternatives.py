"""
Alternative Anticipatory Activation (AA) measurement methods.

Provides multiple approaches for measuring anticipatory activation in agents,
enabling comparison across different measurement methodologies.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.utils.logging import EpisodeLog


@dataclass
class AAMeasurementSuite:
    """Suite of AA measurements using different methods."""

    internal_state_norm: float
    value_function: float
    gradient_magnitude: float
    policy_entropy: float
    belief_precision: float | None = None

    # Raw data for analysis
    near_samples: int = 0
    far_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "internal_state_norm": self.internal_state_norm,
            "value_function": self.value_function,
            "gradient_magnitude": self.gradient_magnitude,
            "policy_entropy": self.policy_entropy,
            "belief_precision": self.belief_precision,
            "near_samples": self.near_samples,
            "far_samples": self.far_samples,
        }


@dataclass
class AADistanceProfile:
    """AA measurements across different distances to goal."""

    distances: np.ndarray
    activations: np.ndarray
    value_estimates: np.ndarray
    entropy_values: np.ndarray
    gradient_magnitudes: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "distances": self.distances.tolist(),
            "activations": self.activations.tolist(),
            "value_estimates": self.value_estimates.tolist(),
            "entropy_values": self.entropy_values.tolist(),
            "gradient_magnitudes": self.gradient_magnitudes.tolist(),
        }


def compute_alternative_aa_scores(
    episodes: list[EpisodeLog],
    agent: BaseAgent,
    methods: list[str] = None,
    near_threshold: int = 3,
    far_threshold: int = 8,
) -> AAMeasurementSuite:
    """
    Compute AA scores using multiple measurement methods.

    Args:
        episodes: Episode logs with internal states
        agent: The agent (for computing value function, gradients)
        methods: Which methods to use (default: all available)
        near_threshold: Distance considered "near" goal
        far_threshold: Distance considered "far" from goal

    Returns:
        AAMeasurementSuite with scores from each method
    """
    if methods is None:
        methods = ["internal_state", "value", "gradient", "entropy"]

    # Collect measurements by distance
    near_data = {"activations": [], "values": [], "gradients": [], "entropies": []}
    far_data = {"activations": [], "values": [], "gradients": [], "entropies": []}

    for episode in episodes:
        for step in episode.steps:
            info = step.info
            internal_state = step.internal_state
            distance = info.get("nearest_resource_distance")

            if distance is None or internal_state is None:
                continue

            # Internal state norm (activation magnitude)
            activation = np.linalg.norm(internal_state)

            # Value function estimate (if available)
            value = _get_value_estimate(agent, step, info)

            # Gradient magnitude (if available)
            gradient = _get_gradient_magnitude(agent, step, info)

            # Policy entropy
            entropy = _get_policy_entropy(agent, step, info)

            # Categorize by distance
            if distance <= near_threshold:
                near_data["activations"].append(activation)
                near_data["values"].append(value)
                near_data["gradients"].append(gradient)
                near_data["entropies"].append(entropy)
            elif distance >= far_threshold:
                far_data["activations"].append(activation)
                far_data["values"].append(value)
                far_data["gradients"].append(gradient)
                far_data["entropies"].append(entropy)

    # Compute AA scores for each method
    def compute_ratio(near_vals: list[float], far_vals: list[float]) -> float:
        if not near_vals or not far_vals:
            return 1.0
        near_mean = np.mean(near_vals)
        far_mean = np.mean(far_vals)
        if far_mean < 0.001:
            far_mean = 0.001
        return near_mean / far_mean

    internal_state_aa = compute_ratio(near_data["activations"], far_data["activations"])
    value_aa = compute_ratio(near_data["values"], far_data["values"])
    gradient_aa = compute_ratio(near_data["gradients"], far_data["gradients"])

    # For entropy, lower entropy near goal = more anticipation
    # So we invert: far_entropy / near_entropy
    entropy_near = near_data["entropies"]
    entropy_far = far_data["entropies"]
    if entropy_near and entropy_far:
        near_mean = np.mean(entropy_near)
        far_mean = np.mean(entropy_far)
        if near_mean < 0.001:
            near_mean = 0.001
        entropy_aa = far_mean / near_mean
    else:
        entropy_aa = 1.0

    return AAMeasurementSuite(
        internal_state_norm=internal_state_aa,
        value_function=value_aa,
        gradient_magnitude=gradient_aa,
        policy_entropy=entropy_aa,
        near_samples=len(near_data["activations"]),
        far_samples=len(far_data["activations"]),
    )


def compute_aa_distance_profile(
    episodes: list[EpisodeLog],
    agent: BaseAgent,
    max_distance: int = 15,
    bin_size: int = 1,
) -> AADistanceProfile:
    """
    Compute AA measurements as a function of distance to goal.

    Args:
        episodes: Episode logs
        agent: The agent
        max_distance: Maximum distance to include
        bin_size: Size of distance bins

    Returns:
        AADistanceProfile with measurements at each distance
    """
    n_bins = max_distance // bin_size + 1
    distance_bins = {i: {"activations": [], "values": [], "gradients": [], "entropies": []}
                     for i in range(n_bins)}

    for episode in episodes:
        for step in episode.steps:
            info = step.info
            internal_state = step.internal_state
            distance = info.get("nearest_resource_distance")

            if distance is None or internal_state is None:
                continue

            if distance > max_distance:
                continue

            bin_idx = int(distance) // bin_size

            activation = np.linalg.norm(internal_state)
            value = _get_value_estimate(agent, step, info)
            gradient = _get_gradient_magnitude(agent, step, info)
            entropy = _get_policy_entropy(agent, step, info)

            distance_bins[bin_idx]["activations"].append(activation)
            distance_bins[bin_idx]["values"].append(value)
            distance_bins[bin_idx]["gradients"].append(gradient)
            distance_bins[bin_idx]["entropies"].append(entropy)

    # Aggregate bins
    distances = []
    activations = []
    values = []
    gradients = []
    entropies = []

    for i in range(n_bins):
        if distance_bins[i]["activations"]:
            distances.append(i * bin_size + bin_size / 2)
            activations.append(np.mean(distance_bins[i]["activations"]))
            values.append(np.mean(distance_bins[i]["values"]))
            gradients.append(np.mean(distance_bins[i]["gradients"]))
            entropies.append(np.mean(distance_bins[i]["entropies"]))

    return AADistanceProfile(
        distances=np.array(distances),
        activations=np.array(activations),
        value_estimates=np.array(values),
        entropy_values=np.array(entropies),
        gradient_magnitudes=np.array(gradients),
    )


def _get_value_estimate(agent: BaseAgent, step: Any, info: dict) -> float:
    """Get value function estimate from agent or step info."""
    # Check if value stored in info
    if "value_estimate" in info:
        return float(info["value_estimate"])

    # Try agent method
    if hasattr(agent, "get_value_function_estimate"):
        try:
            # Would need observation - reconstruct from step
            return 0.0  # Placeholder
        except Exception:
            pass

    # Default: use internal state norm as proxy
    if step.internal_state is not None:
        return float(np.linalg.norm(step.internal_state))
    return 0.0


def _get_gradient_magnitude(agent: BaseAgent, step: Any, info: dict) -> float:
    """Get gradient magnitude from agent or step info."""
    # Check if stored in info
    if "gradient_magnitude" in info:
        return float(info["gradient_magnitude"])

    # Try agent method
    if hasattr(agent, "get_gradient_magnitude"):
        try:
            return 0.0  # Placeholder
        except Exception:
            pass

    # Default: use change in internal state as proxy
    if step.internal_state is not None:
        return float(np.std(step.internal_state))
    return 0.0


def _get_policy_entropy(agent: BaseAgent, step: Any, info: dict) -> float:
    """Get policy entropy from step info."""
    if "policy_entropy" in info:
        return float(info["policy_entropy"])

    # Compute from action probabilities if available
    if "action_probs" in info:
        probs = np.array(info["action_probs"])
        probs = probs + 1e-10  # Avoid log(0)
        return float(-np.sum(probs * np.log(probs)))

    # Default: maximum entropy for uniform distribution
    return 1.79  # log(6) for 6 actions


def compute_aa_correlation_analysis(
    episodes: list[EpisodeLog],
    variables: list[str] = None,
) -> dict[str, float]:
    """
    Compute correlations between distance and various AA measures.

    Args:
        episodes: Episode logs
        variables: Which variables to correlate (default: all)

    Returns:
        Dictionary of correlation coefficients
    """
    if variables is None:
        variables = ["activation", "value", "energy", "steps_to_goal"]

    distances = []
    activations = []
    energies = []

    for episode in episodes:
        for step in episode.steps:
            distance = step.info.get("nearest_resource_distance")
            if distance is None:
                continue

            distances.append(distance)

            if step.internal_state is not None:
                activations.append(np.linalg.norm(step.internal_state))
            else:
                activations.append(0.0)

            energies.append(step.energy)

    if len(distances) < 10:
        return {"error": "Insufficient data"}

    distances = np.array(distances)
    activations = np.array(activations)
    energies = np.array(energies)

    correlations = {}

    # Distance-activation correlation
    if "activation" in variables:
        corr = np.corrcoef(distances, activations)[0, 1]
        correlations["distance_activation"] = float(corr) if not np.isnan(corr) else 0.0

    # Distance-energy correlation
    if "energy" in variables:
        corr = np.corrcoef(distances, energies)[0, 1]
        correlations["distance_energy"] = float(corr) if not np.isnan(corr) else 0.0

    # Activation-energy partial correlation
    if "activation" in variables and "energy" in variables:
        # Simple partial correlation approximation
        r_da = correlations.get("distance_activation", 0)
        r_de = correlations.get("distance_energy", 0)
        r_ae = np.corrcoef(activations, energies)[0, 1]
        if not np.isnan(r_ae):
            denom = np.sqrt((1 - r_de**2) * (1 - r_ae**2))
            if denom > 0.001:
                partial = (r_da - r_de * r_ae) / denom
                correlations["distance_activation_partial"] = float(partial)

    return correlations


def compare_aa_methods(
    episodes: list[EpisodeLog],
    agent: BaseAgent,
    near_threshold: int = 3,
    far_threshold: int = 8,
) -> dict[str, Any]:
    """
    Compare different AA measurement methods and assess convergence.

    Args:
        episodes: Episode logs
        agent: The agent
        near_threshold: Distance threshold for "near"
        far_threshold: Distance threshold for "far"

    Returns:
        Comparison results with method agreement metrics
    """
    suite = compute_alternative_aa_scores(
        episodes, agent,
        near_threshold=near_threshold,
        far_threshold=far_threshold,
    )

    # Collect all AA scores
    scores = {
        "internal_state": suite.internal_state_norm,
        "value_function": suite.value_function,
        "gradient": suite.gradient_magnitude,
        "entropy": suite.policy_entropy,
    }

    # Check agreement
    values = list(scores.values())
    mean_score = np.mean(values)
    std_score = np.std(values)

    # All methods agree on direction (>1 or <1)?
    all_above_1 = all(v > 1.0 for v in values)
    all_below_1 = all(v < 1.0 for v in values)
    direction_agreement = all_above_1 or all_below_1

    return {
        "scores": scores,
        "mean_across_methods": float(mean_score),
        "std_across_methods": float(std_score),
        "direction_agreement": direction_agreement,
        "coefficient_of_variation": float(std_score / mean_score) if mean_score > 0 else 0.0,
        "samples": {
            "near": suite.near_samples,
            "far": suite.far_samples,
        },
    }
