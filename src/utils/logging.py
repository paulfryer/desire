"""
Logging utilities for trajectory recording and analysis.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np


@dataclass
class StepLog:
    """Log entry for a single step."""

    step: int
    action: int
    position: tuple[int, int]
    energy: float
    reward: float
    observation: Optional[np.ndarray] = None
    internal_state: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "step": self.step,
            "action": self.action,
            "position": list(self.position),
            "energy": self.energy,
            "reward": self.reward,
            "info": self.info,
        }

        if self.observation is not None:
            d["observation"] = self.observation.tolist()
        if self.internal_state is not None:
            d["internal_state"] = self.internal_state.tolist()

        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepLog":
        """Create from dictionary."""
        observation = None
        if "observation" in data:
            observation = np.array(data["observation"])

        internal_state = None
        if "internal_state" in data:
            internal_state = np.array(data["internal_state"])

        return cls(
            step=data["step"],
            action=data["action"],
            position=tuple(data["position"]),
            energy=data["energy"],
            reward=data["reward"],
            observation=observation,
            internal_state=internal_state,
            info=data.get("info", {}),
        )


@dataclass
class EpisodeLog:
    """Log entry for a complete episode."""

    episode_id: int
    agent_type: str
    environment_mode: str
    total_steps: int
    total_reward: float
    final_energy: float
    survived: bool
    steps: List[StepLog] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "agent_type": self.agent_type,
            "environment_mode": self.environment_mode,
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "final_energy": self.final_energy,
            "survived": self.survived,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeLog":
        """Create from dictionary."""
        steps = [StepLog.from_dict(s) for s in data.get("steps", [])]

        return cls(
            episode_id=data["episode_id"],
            agent_type=data["agent_type"],
            environment_mode=data["environment_mode"],
            total_steps=data["total_steps"],
            total_reward=data["total_reward"],
            final_energy=data["final_energy"],
            survived=data["survived"],
            steps=steps,
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )

    def get_trajectory(self) -> np.ndarray:
        """Get position trajectory as numpy array."""
        return np.array([s.position for s in self.steps])

    def get_energy_curve(self) -> np.ndarray:
        """Get energy values over time."""
        return np.array([s.energy for s in self.steps])

    def get_reward_curve(self) -> np.ndarray:
        """Get cumulative rewards over time."""
        rewards = np.array([s.reward for s in self.steps])
        return np.cumsum(rewards)

    def get_action_sequence(self) -> np.ndarray:
        """Get sequence of actions taken."""
        return np.array([s.action for s in self.steps])

    def get_internal_states(self) -> Optional[np.ndarray]:
        """Get internal states if recorded."""
        states = [s.internal_state for s in self.steps if s.internal_state is not None]
        if not states:
            return None
        return np.array(states)


class TrajectoryLogger:
    """
    Logger for recording agent trajectories.

    Handles recording, saving, and loading of episode data.
    """

    def __init__(
        self,
        output_dir: Path,
        agent_type: str,
        environment_mode: str,
        experiment_name: str = "experiment",
        save_observations: bool = False,
        save_internal_states: bool = True,
    ):
        """
        Initialize the trajectory logger.

        Args:
            output_dir: Directory for saving logs
            agent_type: Type of agent being logged
            environment_mode: Environment mode
            experiment_name: Name of the experiment
            save_observations: Whether to save full observations
            save_internal_states: Whether to save internal states
        """
        self.output_dir = Path(output_dir)
        self.agent_type = agent_type
        self.environment_mode = environment_mode
        self.experiment_name = experiment_name
        self.save_observations = save_observations
        self.save_internal_states = save_internal_states

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Current episode tracking
        self.current_episode: Optional[EpisodeLog] = None
        self.episode_count = 0

        # All episodes
        self.episodes: List[EpisodeLog] = []

    def start_episode(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start logging a new episode."""
        self.current_episode = EpisodeLog(
            episode_id=self.episode_count,
            agent_type=self.agent_type,
            environment_mode=self.environment_mode,
            total_steps=0,
            total_reward=0.0,
            final_energy=0.0,
            survived=True,
            metadata=metadata or {},
        )

    def log_step(
        self,
        step: int,
        action: int,
        position: tuple[int, int],
        energy: float,
        reward: float,
        observation: Optional[np.ndarray] = None,
        internal_state: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single step."""
        if self.current_episode is None:
            raise RuntimeError("No episode started. Call start_episode() first.")

        step_log = StepLog(
            step=step,
            action=action,
            position=position,
            energy=energy,
            reward=reward,
            observation=observation if self.save_observations else None,
            internal_state=internal_state if self.save_internal_states else None,
            info=info or {},
        )

        self.current_episode.steps.append(step_log)
        self.current_episode.total_steps = step + 1
        self.current_episode.total_reward += reward

    def end_episode(self, final_energy: float, survived: bool) -> EpisodeLog:
        """End the current episode and return the log."""
        if self.current_episode is None:
            raise RuntimeError("No episode started. Call start_episode() first.")

        self.current_episode.final_energy = final_energy
        self.current_episode.survived = survived

        episode = self.current_episode
        self.episodes.append(episode)
        self.episode_count += 1
        self.current_episode = None

        return episode

    def save_episode(self, episode: EpisodeLog, filename: Optional[str] = None) -> Path:
        """Save a single episode to file."""
        if filename is None:
            filename = f"episode_{episode.episode_id:05d}.json"

        path = self.output_dir / filename

        with open(path, 'w') as f:
            json.dump(episode.to_dict(), f, indent=2)

        return path

    def save_all(self, filename: Optional[str] = None) -> Path:
        """Save all episodes to a single file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_{self.agent_type}_{timestamp}.json"

        path = self.output_dir / filename

        data = {
            "experiment_name": self.experiment_name,
            "agent_type": self.agent_type,
            "environment_mode": self.environment_mode,
            "num_episodes": len(self.episodes),
            "episodes": [e.to_dict() for e in self.episodes],
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        return path

    @classmethod
    def load(cls, path: Path) -> List[EpisodeLog]:
        """Load episodes from file."""
        path = Path(path)

        with open(path, 'r') as f:
            data = json.load(f)

        if "episodes" in data:
            return [EpisodeLog.from_dict(e) for e in data["episodes"]]
        else:
            # Single episode file
            return [EpisodeLog.from_dict(data)]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all logged episodes."""
        if not self.episodes:
            return {"num_episodes": 0}

        total_rewards = [e.total_reward for e in self.episodes]
        total_steps = [e.total_steps for e in self.episodes]
        survival_rate = sum(1 for e in self.episodes if e.survived) / len(self.episodes)

        return {
            "num_episodes": len(self.episodes),
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "mean_steps": float(np.mean(total_steps)),
            "std_steps": float(np.std(total_steps)),
            "survival_rate": survival_rate,
        }


def setup_logging(
    output_dir: str,
    agent_type: str,
    environment_mode: str,
    experiment_name: str = "experiment",
) -> TrajectoryLogger:
    """
    Convenience function to set up logging.

    Args:
        output_dir: Directory for logs
        agent_type: Type of agent
        environment_mode: Environment mode
        experiment_name: Experiment name

    Returns:
        Configured TrajectoryLogger
    """
    return TrajectoryLogger(
        output_dir=Path(output_dir) / "logs",
        agent_type=agent_type,
        environment_mode=environment_mode,
        experiment_name=experiment_name,
    )
