"""
Temporal dynamics analysis for Anticipatory Activation.

Analyzes how activation patterns evolve over time as agents approach goals,
including onset timing, ramp rates, and trajectory characteristics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy import stats

from src.utils.logging import EpisodeLog


@dataclass
class ActivationTrajectory:
    """Trajectory of activation as agent approaches goal."""

    timestamps: np.ndarray  # Step numbers
    activations: np.ndarray  # Activation magnitudes
    distances: np.ndarray  # Distances to goal
    derivatives: np.ndarray  # d(activation)/dt
    goal_position: Tuple[int, int] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_steps": len(self.timestamps),
            "mean_activation": float(np.mean(self.activations)),
            "max_activation": float(np.max(self.activations)),
            "mean_derivative": float(np.mean(self.derivatives)),
            "goal_position": self.goal_position,
        }


@dataclass
class AnticipationTimingResult:
    """Timing characteristics of anticipatory activation."""

    onset_distance: float  # Distance where activation starts increasing
    peak_distance: float  # Distance at peak activation
    ramp_duration: int  # Steps from onset to peak
    ramp_rate: float  # Average rate of activation increase
    peak_activation: float  # Maximum activation value
    baseline_activation: float  # Activation far from goal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "onset_distance": self.onset_distance,
            "peak_distance": self.peak_distance,
            "ramp_duration": self.ramp_duration,
            "ramp_rate": self.ramp_rate,
            "peak_activation": self.peak_activation,
            "baseline_activation": self.baseline_activation,
        }


def analyze_activation_trajectory(
    episode: EpisodeLog,
    goal_position: Optional[Tuple[int, int]] = None,
) -> ActivationTrajectory:
    """
    Extract activation trajectory from an episode.

    Args:
        episode: Episode log with internal states
        goal_position: Position of goal (inferred if not provided)

    Returns:
        ActivationTrajectory for the episode
    """
    timestamps = []
    activations = []
    distances = []

    for step in episode.steps:
        if step.internal_state is None:
            continue

        timestamps.append(step.step)
        activations.append(np.linalg.norm(step.internal_state))

        # Get distance from info or compute
        distance = step.info.get("nearest_resource_distance")
        if distance is not None:
            distances.append(distance)
        elif goal_position is not None:
            pos = np.array(step.position)
            goal = np.array(goal_position)
            distances.append(float(np.linalg.norm(pos - goal)))
        else:
            distances.append(0.0)

    timestamps = np.array(timestamps)
    activations = np.array(activations)
    distances = np.array(distances)

    # Compute derivatives (change in activation per step)
    if len(activations) > 1:
        derivatives = np.gradient(activations)
    else:
        derivatives = np.zeros_like(activations)

    # Infer goal position from minimum distance
    if goal_position is None and len(distances) > 0:
        min_dist_idx = np.argmin(distances)
        goal_position = episode.steps[min_dist_idx].position

    return ActivationTrajectory(
        timestamps=timestamps,
        activations=activations,
        distances=distances,
        derivatives=derivatives,
        goal_position=goal_position,
    )


def compute_anticipation_timing(
    trajectories: List[ActivationTrajectory],
    baseline_distance: float = 10.0,
    onset_threshold: float = 1.5,  # Multiples of baseline std
) -> AnticipationTimingResult:
    """
    Compute timing characteristics of anticipatory activation.

    Args:
        trajectories: List of activation trajectories
        baseline_distance: Distance considered "far" for baseline
        onset_threshold: Threshold for detecting onset (in std units)

    Returns:
        AnticipationTimingResult with timing metrics
    """
    # Collect all (distance, activation) pairs
    all_distances = []
    all_activations = []

    for traj in trajectories:
        for dist, act in zip(traj.distances, traj.activations):
            all_distances.append(dist)
            all_activations.append(act)

    all_distances = np.array(all_distances)
    all_activations = np.array(all_activations)

    # Compute baseline (far from goal)
    far_mask = all_distances >= baseline_distance
    if np.sum(far_mask) > 0:
        baseline_activation = np.mean(all_activations[far_mask])
        baseline_std = np.std(all_activations[far_mask])
    else:
        baseline_activation = np.mean(all_activations)
        baseline_std = np.std(all_activations)

    if baseline_std < 0.001:
        baseline_std = 0.001

    # Find onset: first distance where activation exceeds threshold
    threshold = baseline_activation + onset_threshold * baseline_std

    # Bin by distance and find where activation consistently exceeds threshold
    distance_bins = np.arange(0, baseline_distance + 1, 1)
    binned_activations = []

    for i in range(len(distance_bins) - 1):
        mask = (all_distances >= distance_bins[i]) & (all_distances < distance_bins[i + 1])
        if np.sum(mask) > 0:
            binned_activations.append(np.mean(all_activations[mask]))
        else:
            binned_activations.append(baseline_activation)

    binned_activations = np.array(binned_activations)

    # Find onset (scanning from far to near)
    onset_distance = baseline_distance
    for i in range(len(binned_activations) - 1, -1, -1):
        if binned_activations[i] < threshold:
            onset_distance = distance_bins[i + 1] if i + 1 < len(distance_bins) else baseline_distance
            break

    # Find peak
    peak_idx = np.argmax(binned_activations)
    peak_distance = (distance_bins[peak_idx] + distance_bins[peak_idx + 1]) / 2 if peak_idx < len(distance_bins) - 1 else 0
    peak_activation = float(binned_activations[peak_idx])

    # Compute ramp duration and rate
    ramp_duration = int(onset_distance - peak_distance)

    if ramp_duration > 0:
        ramp_rate = (peak_activation - baseline_activation) / ramp_duration
    else:
        ramp_rate = 0.0

    return AnticipationTimingResult(
        onset_distance=float(onset_distance),
        peak_distance=float(peak_distance),
        ramp_duration=ramp_duration,
        ramp_rate=float(ramp_rate),
        peak_activation=peak_activation,
        baseline_activation=float(baseline_activation),
    )


def compute_activation_dynamics_features(
    trajectory: ActivationTrajectory,
) -> Dict[str, float]:
    """
    Compute dynamic features of an activation trajectory.

    Args:
        trajectory: Single activation trajectory

    Returns:
        Dictionary of dynamic features
    """
    activations = trajectory.activations
    distances = trajectory.distances
    derivatives = trajectory.derivatives

    if len(activations) < 3:
        return {
            "temporal_autocorrelation": 0.0,
            "approach_correlation": 0.0,
            "acceleration": 0.0,
            "jerk": 0.0,
            "smoothness": 0.0,
        }

    # Temporal autocorrelation (lag-1)
    if len(activations) > 1:
        autocorr = np.corrcoef(activations[:-1], activations[1:])[0, 1]
        autocorr = float(autocorr) if not np.isnan(autocorr) else 0.0
    else:
        autocorr = 0.0

    # Correlation between distance decrease and activation increase
    dist_changes = np.diff(distances)
    act_changes = np.diff(activations)
    if len(dist_changes) > 1:
        approach_corr = np.corrcoef(dist_changes, act_changes)[0, 1]
        approach_corr = float(approach_corr) if not np.isnan(approach_corr) else 0.0
    else:
        approach_corr = 0.0

    # Acceleration (second derivative)
    if len(derivatives) > 1:
        acceleration = np.gradient(derivatives)
        mean_accel = float(np.mean(np.abs(acceleration)))
    else:
        mean_accel = 0.0

    # Jerk (third derivative)
    if len(derivatives) > 2:
        accel = np.gradient(derivatives)
        jerk = np.gradient(accel)
        mean_jerk = float(np.mean(np.abs(jerk)))
    else:
        mean_jerk = 0.0

    # Smoothness (inverse of total variation)
    total_variation = np.sum(np.abs(np.diff(activations)))
    if total_variation > 0:
        smoothness = 1.0 / total_variation
    else:
        smoothness = float('inf')

    return {
        "temporal_autocorrelation": autocorr,
        "approach_correlation": approach_corr,
        "acceleration": mean_accel,
        "jerk": mean_jerk,
        "smoothness": smoothness if smoothness != float('inf') else 1000.0,
    }


def detect_activation_events(
    trajectory: ActivationTrajectory,
    threshold_std: float = 2.0,
    min_duration: int = 3,
) -> List[Dict[str, Any]]:
    """
    Detect significant activation events in trajectory.

    Args:
        trajectory: Activation trajectory
        threshold_std: Threshold in standard deviations
        min_duration: Minimum event duration in steps

    Returns:
        List of detected events with characteristics
    """
    activations = trajectory.activations
    distances = trajectory.distances
    timestamps = trajectory.timestamps

    if len(activations) < min_duration:
        return []

    # Compute threshold
    mean_act = np.mean(activations)
    std_act = np.std(activations)
    threshold = mean_act + threshold_std * std_act

    # Find regions above threshold
    above_threshold = activations > threshold

    # Find contiguous regions
    events = []
    in_event = False
    event_start = 0

    for i, is_above in enumerate(above_threshold):
        if is_above and not in_event:
            in_event = True
            event_start = i
        elif not is_above and in_event:
            in_event = False
            event_duration = i - event_start

            if event_duration >= min_duration:
                event_activations = activations[event_start:i]
                event_distances = distances[event_start:i]

                events.append({
                    "start_step": int(timestamps[event_start]),
                    "end_step": int(timestamps[i - 1]),
                    "duration": event_duration,
                    "peak_activation": float(np.max(event_activations)),
                    "mean_activation": float(np.mean(event_activations)),
                    "start_distance": float(event_distances[0]),
                    "end_distance": float(event_distances[-1]),
                })

    # Check if episode ended during event
    if in_event:
        event_duration = len(activations) - event_start
        if event_duration >= min_duration:
            event_activations = activations[event_start:]
            event_distances = distances[event_start:]

            events.append({
                "start_step": int(timestamps[event_start]),
                "end_step": int(timestamps[-1]),
                "duration": event_duration,
                "peak_activation": float(np.max(event_activations)),
                "mean_activation": float(np.mean(event_activations)),
                "start_distance": float(event_distances[0]),
                "end_distance": float(event_distances[-1]),
            })

    return events


def compute_phase_space_metrics(
    trajectories: List[ActivationTrajectory],
) -> Dict[str, float]:
    """
    Compute phase space characteristics of activation dynamics.

    Args:
        trajectories: List of activation trajectories

    Returns:
        Phase space metrics
    """
    # Collect all (activation, derivative) pairs
    all_activations = []
    all_derivatives = []

    for traj in trajectories:
        all_activations.extend(traj.activations.tolist())
        all_derivatives.extend(traj.derivatives.tolist())

    activations = np.array(all_activations)
    derivatives = np.array(all_derivatives)

    if len(activations) < 10:
        return {
            "phase_spread": 0.0,
            "attractor_strength": 0.0,
            "limit_cycle_tendency": 0.0,
        }

    # Phase space spread (covariance)
    phase_cov = np.cov(activations, derivatives)
    phase_spread = np.trace(phase_cov)

    # Attractor strength (tendency to return to mean)
    deviations = activations - np.mean(activations)
    if len(deviations) > 1:
        # Correlation between deviation and subsequent change
        attractor_corr = np.corrcoef(deviations[:-1], derivatives[:-1])[0, 1]
        attractor_strength = -float(attractor_corr) if not np.isnan(attractor_corr) else 0.0
    else:
        attractor_strength = 0.0

    # Limit cycle tendency (cyclic patterns)
    if len(activations) > 10:
        # Use autocorrelation to detect periodicity
        autocorr = np.correlate(activations - np.mean(activations),
                                activations - np.mean(activations),
                                mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        # Find first peak after lag 0
        peaks = signal.find_peaks(autocorr, distance=5)[0]
        if len(peaks) > 0:
            limit_cycle_tendency = float(autocorr[peaks[0]])
        else:
            limit_cycle_tendency = 0.0
    else:
        limit_cycle_tendency = 0.0

    return {
        "phase_spread": float(phase_spread),
        "attractor_strength": attractor_strength,
        "limit_cycle_tendency": limit_cycle_tendency,
    }
