"""
Configuration management for the Desire research project.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvironmentConfig:
    """Configuration for the environment."""

    grid_size: int = 20
    max_steps: int = 1000
    initial_energy: float = 100.0
    max_energy: float = 100.0
    energy_decay_rate: float = 1.0
    local_view_size: int = 5
    mode: str = "baseline"
    render_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_size": self.grid_size,
            "max_steps": self.max_steps,
            "initial_energy": self.initial_energy,
            "max_energy": self.max_energy,
            "energy_decay_rate": self.energy_decay_rate,
            "local_view_size": self.local_view_size,
            "mode": self.mode,
            "render_mode": self.render_mode,
        }


@dataclass
class AgentConfig:
    """Configuration for agents."""

    agent_type: str = "heuristic"  # heuristic, rl, active_inference

    # RL-specific
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.01

    # Active inference specific
    planning_horizon: int = 3
    num_states: int = 100  # Discretized state space size
    policy_precision: float = 4.0

    # Common
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "planning_horizon": self.planning_horizon,
            "num_states": self.num_states,
            "policy_precision": self.policy_precision,
            "hidden_sizes": self.hidden_sizes,
        }


@dataclass
class TrainingConfig:
    """Configuration for training."""

    total_timesteps: int = 100000
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    save_freq: int = 10000
    log_freq: int = 1000
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    checkpoint_keep_last: int = 5  # Keep last 5 checkpoints
    resume_from: str | None = None  # Path to resume checkpoint

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_timesteps": self.total_timesteps,
            "eval_freq": self.eval_freq,
            "n_eval_episodes": self.n_eval_episodes,
            "save_freq": self.save_freq,
            "log_freq": self.log_freq,
            "seed": self.seed,
            "device": self.device,
            "checkpoint_keep_last": self.checkpoint_keep_last,
            "resume_from": self.resume_from,
        }


@dataclass
class TestConfig:
    """Configuration for predicate tests."""

    n_trials: int = 10
    baseline_episodes: int = 50
    test_episodes: int = 50
    warmup_steps: int = 100

    # Frustration test
    blockage_step: int = 100
    post_blockage_steps: int = 200

    # Means-end test
    training_episodes: int = 100
    transfer_episodes: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "baseline_episodes": self.baseline_episodes,
            "test_episodes": self.test_episodes,
            "warmup_steps": self.warmup_steps,
            "blockage_step": self.blockage_step,
            "post_blockage_steps": self.post_blockage_steps,
            "training_episodes": self.training_episodes,
            "transfer_episodes": self.transfer_episodes,
        }


@dataclass
class Config:
    """Master configuration containing all sub-configurations."""

    experiment_name: str = "desire_experiment"
    output_dir: str = "outputs"

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    test: TestConfig = field(default_factory=TestConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "environment": self.environment.to_dict(),
            "agent": self.agent.to_dict(),
            "training": self.training.to_dict(),
            "test": self.test.to_dict(),
        }

    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            experiment_name=data.get("experiment_name", "desire_experiment"),
            output_dir=data.get("output_dir", "outputs"),
            environment=EnvironmentConfig(**data.get("environment", {})),
            agent=AgentConfig(**data.get("agent", {})),
            training=TrainingConfig(**data.get("training", {})),
            test=TestConfig(**data.get("test", {})),
        )


def load_config(path: Path) -> Config:
    """
    Load configuration from YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Config object with loaded settings
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return Config.from_dict(data)


def merge_configs(base: Config, override: dict[str, Any]) -> Config:
    """
    Merge override settings into base config.

    Args:
        base: Base configuration
        override: Dictionary of override values

    Returns:
        New Config with merged values
    """
    base_dict = base.to_dict()

    def deep_merge(d1: dict, d2: dict) -> dict:
        result = d1.copy()
        for key, value in d2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = deep_merge(base_dict, override)
    return Config.from_dict(merged)
