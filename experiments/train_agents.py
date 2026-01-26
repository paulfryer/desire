"""
Training script for all agent types.

Usage:
    python experiments/train_agents.py --config experiments/configs/default.yaml
    python experiments/train_agents.py --config experiments/configs/quick_test.yaml

Resume from latest checkpoint:
    python experiments/train_agents.py --resume latest --resume-dir outputs/training_XXXXXX

Resume from specific checkpoint:
    python experiments/train_agents.py --resume outputs/training_XXXXXX/checkpoints/rl_ppo_50000_steps.zip
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.agents.base_agent import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode
from src.utils.callbacks import CheckpointCallbackWithCleanup
from src.utils.checkpoint import find_latest_checkpoint, get_resume_timestep
from src.utils.config import Config, load_config


def create_agents(config: Config, env: ForagingGridEnv) -> dict[str, BaseAgent]:
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
        from src.agents.rl_agent import HAS_SB3, RLAgent
        if not HAS_SB3:
            raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")
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
    except ImportError as e:
        raise ImportError(f"Could not create RL agent: {e}") from e

    # 3. Active Inference agent (using fast implementation)
    try:
        from src.agents.active_inference_fast import HAS_PYMDP, ActiveInferenceFast
        if not HAS_PYMDP:
            raise ImportError("pymdp is required. Install with: pip install inferactively-pymdp")
        agents["active_inference"] = ActiveInferenceFast(
            observation_space=env.observation_space,
            action_space=env.action_space,
            planning_horizon=config.agent.planning_horizon,
            policy_precision=config.agent.policy_precision,
            grid_size=config.environment.grid_size,
            beam_width=20,
            seed=config.training.seed,
        )
    except ImportError as e:
        raise ImportError(f"Could not create Active Inference agent: {e}") from e

    return agents


def train_rl_agent(
    agent,
    config: Config,
    output_dir: Path,
    resume_checkpoint: Path | None = None,
) -> dict[str, list[float]]:
    """
    Train the RL agent using stable-baselines3.

    Args:
        agent: The RL agent to train
        config: Training configuration
        output_dir: Output directory for models and checkpoints
        resume_checkpoint: Optional path to checkpoint for resuming training

    Returns:
        Dictionary with training rewards
    """
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

    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Handle resume vs new training
    reset_num_timesteps = True
    total_timesteps = config.training.total_timesteps

    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        agent.load(str(resume_checkpoint), env=env)
        resume_timestep = get_resume_timestep(resume_checkpoint)
        remaining_timesteps = total_timesteps - resume_timestep
        if remaining_timesteps <= 0:
            print(f"Training already complete (checkpoint at {resume_timestep} steps)")
            env.close()
            return {"rewards": []}
        total_timesteps = remaining_timesteps
        reset_num_timesteps = False
        print(f"Resuming from timestep {resume_timestep}, {total_timesteps} steps remaining")
    else:
        # Create new model
        agent.create_model(env)

    # Training callback for logging
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    training_rewards = []

    class RewardCallback(BaseCallback):
        def __init__(self, verbose: int = 0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.current_reward = 0

        def _on_step(self) -> bool:
            # Track rewards
            if 'rewards' in self.locals:
                self.current_reward += np.sum(self.locals['rewards'])
            if 'dones' in self.locals and any(self.locals['dones']):
                self.episode_rewards.append(self.current_reward)
                training_rewards.append(self.current_reward)
                self.current_reward = 0
            return True

    reward_callback = RewardCallback()

    # Checkpoint callback with cleanup
    checkpoint_callback = CheckpointCallbackWithCleanup(
        save_freq=config.training.save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="rl_ppo",
        keep_last=config.training.checkpoint_keep_last,
        verbose=1,
    )

    # Combine callbacks
    callbacks = CallbackList([reward_callback, checkpoint_callback])

    # Train
    agent.train(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )

    # Save final model
    model_path = output_dir / "models" / "rl_ppo.zip"
    agent.save(str(model_path))
    print(f"Saved RL model to {model_path}")

    env.close()

    return {"rewards": training_rewards}


def evaluate_agent(
    agent: BaseAgent,
    config: Config,
    n_episodes: int = 10,
) -> dict[str, float]:
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training: 'latest' for latest checkpoint, or path to specific checkpoint",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Directory to resume from (required when --resume=latest)",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Config file not found: {config_path}, using defaults")
        config = Config()

    # Handle resume logic
    resume_checkpoint = None
    if args.resume:
        if args.resume.lower() == "latest":
            if not args.resume_dir:
                parser.error("--resume-dir is required when --resume=latest")
            checkpoint_dir = Path(args.resume_dir) / "checkpoints"
            resume_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if resume_checkpoint is None:
                parser.error(f"No checkpoints found in {checkpoint_dir}")
            print(f"Found latest checkpoint: {resume_checkpoint}")
        else:
            resume_checkpoint = Path(args.resume)
            if not resume_checkpoint.exists():
                parser.error(f"Checkpoint not found: {resume_checkpoint}")

    # Set output directory
    if args.resume_dir:
        # Resume into the same directory
        output_dir = Path(args.resume_dir)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.output_dir) / f"training_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Configuration: {config.experiment_name}")
    if resume_checkpoint:
        print(f"Resuming from: {resume_checkpoint}")

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
            training_data = train_rl_agent(
                agent, config, output_dir,
                resume_checkpoint=resume_checkpoint,
            )
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
