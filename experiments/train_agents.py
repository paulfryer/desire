"""
Training script for all agent types.

Usage:
    python experiments/train_agents.py --config experiments/configs/default.yaml
    python experiments/train_agents.py --config experiments/configs/quick_test.yaml
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode
from src.agents.base_agent import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.utils.config import Config, load_config
from src.utils.logging import TrajectoryLogger, setup_logging


def create_agents(config: Config, env: ForagingGridEnv) -> Dict[str, BaseAgent]:
    """Create all agent types for training."""
    agents = {}

    # 1. Heuristic agent (no training needed)
    agents["heuristic"] = HeuristicAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=config.training.seed,
    )

    # 2. RL agent (PPO)
    try:
        from src.agents.rl_agent import RLAgent, HAS_SB3
        if HAS_SB3:
            agents["rl_ppo"] = RLAgent(
                observation_space=env.observation_space,
                action_space=env.action_space,
                learning_rate=config.agent.learning_rate,
                gamma=config.agent.gamma,
                n_steps=config.agent.n_steps,
                batch_size=config.agent.batch_size,
                n_epochs=config.agent.n_epochs,
                clip_range=config.agent.clip_range,
                ent_coef=config.agent.ent_coef,
                device=config.training.device,
                seed=config.training.seed,
            )
        else:
            print("Warning: stable-baselines3 not available, using dummy RL agent")
            from src.agents.rl_agent import DummyRLAgent
            agents["rl_ppo"] = DummyRLAgent(
                observation_space=env.observation_space,
                action_space=env.action_space,
                seed=config.training.seed,
            )
    except ImportError as e:
        print(f"Warning: Could not create RL agent: {e}")

    # 3. Active Inference agent
    try:
        from src.agents.active_inference import ActiveInferenceAgent, HAS_PYMDP
        if HAS_PYMDP:
            agents["active_inference"] = ActiveInferenceAgent(
                observation_space=env.observation_space,
                action_space=env.action_space,
                planning_horizon=config.agent.planning_horizon,
                policy_precision=config.agent.policy_precision,
                grid_size=config.environment.grid_size,
                seed=config.training.seed,
            )
        else:
            print("Warning: pymdp not available, using dummy active inference agent")
            from src.agents.active_inference import DummyActiveInferenceAgent
            agents["active_inference"] = DummyActiveInferenceAgent(
                observation_space=env.observation_space,
                action_space=env.action_space,
                seed=config.training.seed,
            )
    except ImportError as e:
        print(f"Warning: Could not create Active Inference agent: {e}")

    return agents


def train_rl_agent(
    agent,
    config: Config,
    output_dir: Path,
) -> Dict[str, List[float]]:
    """Train the RL agent using stable-baselines3."""
    from src.agents.rl_agent import HAS_SB3
    if not HAS_SB3:
        print("Skipping RL training (stable-baselines3 not available)")
        return {"rewards": []}

    print("\nTraining RL agent...")

    # Create training environment
    env = ForagingGridEnv(
        grid_size=config.environment.grid_size,
        max_steps=config.environment.max_steps,
        initial_energy=config.environment.initial_energy,
        mode=EnvironmentMode.BASELINE,
        seed=config.training.seed,
    )

    # Create model
    agent.create_model(env)

    # Training callback for logging
    training_rewards = []

    class RewardCallback:
        def __init__(self):
            self.episode_rewards = []
            self.current_reward = 0

        def __call__(self, locals_dict, globals_dict):
            # Track rewards
            if 'rewards' in locals_dict:
                self.current_reward += np.sum(locals_dict['rewards'])
            if 'dones' in locals_dict and any(locals_dict['dones']):
                self.episode_rewards.append(self.current_reward)
                training_rewards.append(self.current_reward)
                self.current_reward = 0
            return True

    callback = RewardCallback()

    # Train
    agent.train(
        total_timesteps=config.training.total_timesteps,
        callback=callback,
    )

    # Save model
    model_path = output_dir / "models" / "rl_ppo.zip"
    agent.save(str(model_path))
    print(f"Saved RL model to {model_path}")

    env.close()

    return {"rewards": training_rewards}


def evaluate_agent(
    agent: BaseAgent,
    config: Config,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """Evaluate an agent on the baseline environment."""
    env = ForagingGridEnv(
        grid_size=config.environment.grid_size,
        max_steps=config.environment.max_steps,
        initial_energy=config.environment.initial_energy,
        mode=EnvironmentMode.BASELINE,
        seed=config.training.seed,
    )

    total_rewards = []
    total_steps = []
    survival_count = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=config.training.seed + ep)
        agent.reset()

        episode_reward = 0
        step = 0
        done = False

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        total_steps.append(step)
        if not terminated:  # Survived (truncated due to time limit)
            survival_count += 1

    env.close()

    return {
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_steps": float(np.mean(total_steps)),
        "survival_rate": survival_count / n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Train agents for Desire research")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=None,
        help="Specific agents to train (default: all)",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Config file not found: {config_path}, using defaults")
        config = Config()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.output_dir) / f"training_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Configuration: {config.experiment_name}")

    # Set random seed
    np.random.seed(config.training.seed)

    # Create environment for agent initialization
    env = ForagingGridEnv(
        grid_size=config.environment.grid_size,
        max_steps=config.environment.max_steps,
        mode=EnvironmentMode.BASELINE,
        seed=config.training.seed,
    )

    # Create agents
    agents = create_agents(config, env)

    # Filter agents if specified
    if args.agents:
        agents = {k: v for k, v in agents.items() if k in args.agents}

    print(f"\nAgents to train: {list(agents.keys())}")

    # Training results
    training_results = {}
    evaluation_results = {}

    # Train and evaluate each agent
    for agent_name, agent in agents.items():
        print(f"\n{'='*50}")
        print(f"Processing agent: {agent_name}")
        print(f"{'='*50}")

        # Train if needed
        if agent_name == "rl_ppo" and hasattr(agent, 'create_model'):
            training_data = train_rl_agent(agent, config, output_dir)
            training_results[agent_name] = training_data
        else:
            print(f"Agent {agent_name} does not require training")

        # Evaluate
        print(f"Evaluating {agent_name}...")
        eval_results = evaluate_agent(
            agent, config,
            n_episodes=config.training.n_eval_episodes,
        )
        evaluation_results[agent_name] = eval_results

        print(f"  Mean reward: {eval_results['mean_reward']:.2f}")
        print(f"  Survival rate: {eval_results['survival_rate']:.2%}")

    # Save results
    results = {
        "config": config.to_dict(),
        "training": training_results,
        "evaluation": evaluation_results,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)

    for agent_name, eval_res in evaluation_results.items():
        print(f"\n{agent_name}:")
        print(f"  Mean reward: {eval_res['mean_reward']:.2f} (+/- {eval_res['std_reward']:.2f})")
        print(f"  Mean steps: {eval_res['mean_steps']:.0f}")
        print(f"  Survival rate: {eval_res['survival_rate']:.2%}")

    env.close()


if __name__ == "__main__":
    main()
