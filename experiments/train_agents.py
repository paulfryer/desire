"""
Training script for all agent types.

Usage:
    python experiments/train_agents.py --config experiments/configs/default.yaml
    python experiments/train_agents.py --config experiments/configs/quick_test.yaml

Resume from latest checkpoint:
    python experiments/train_agents.py --resume latest --resume-dir outputs/training_XXXXXX

Resume from specific checkpoint:
    python experiments/train_agents.py --resume outputs/training_XXXXXX/checkpoints/rl_ppo_50000_steps.zip

Train specific agents:
    python experiments/train_agents.py --agents rl_ppo rl_dqn rl_a2c
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


# List of all trainable RL agents using stable-baselines3
SB3_AGENTS = ["rl_ppo", "rl_dqn", "rl_a2c"]

# Agents with custom training loops
CUSTOM_TRAINING_AGENTS = ["model_based_rl", "decision_transformer"]


def create_agents(config: Config, env: ForagingGridEnv, agent_list: list[str] | None = None) -> dict[str, BaseAgent]:
    """
    Create agent instances for training.

    Args:
        config: Configuration object
        env: Environment for agent initialization
        agent_list: Optional list of specific agents to create

    Returns:
        Dictionary of agent name -> agent instance
    """
    agents = {}

    # Default to all agents if none specified
    if agent_list is None:
        agent_list = ["heuristic", "rl_ppo", "rl_dqn", "rl_a2c", "active_inference"]

    # 1. Heuristic agent (no training needed)
    if "heuristic" in agent_list:
        agents["heuristic"] = HeuristicAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=config.training.seed,
        )

    # 2. RL agent (PPO)
    if "rl_ppo" in agent_list:
        try:
            from src.agents.rl_agent import HAS_SB3, RLAgent
            if not HAS_SB3:
                print("Warning: stable-baselines3 not available, skipping rl_ppo")
            else:
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
            print(f"Warning: Could not create rl_ppo: {e}")

    # 3. DQN agent
    if "rl_dqn" in agent_list:
        try:
            from src.agents.rl_dqn import DQNAgent, HAS_SB3
            if not HAS_SB3:
                print("Warning: stable-baselines3 not available, skipping rl_dqn")
            else:
                agents["rl_dqn"] = DQNAgent(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    learning_rate=config.agent.learning_rate,
                    gamma=config.agent.gamma,
                    device=config.training.device,
                    seed=config.training.seed,
                )
        except ImportError as e:
            print(f"Warning: Could not create rl_dqn: {e}")

    # 4. A2C agent
    if "rl_a2c" in agent_list:
        try:
            from src.agents.rl_a2c import A2CAgent, HAS_SB3
            if not HAS_SB3:
                print("Warning: stable-baselines3 not available, skipping rl_a2c")
            else:
                agents["rl_a2c"] = A2CAgent(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    learning_rate=config.agent.learning_rate,
                    gamma=config.agent.gamma,
                    n_steps=config.agent.n_steps,
                    ent_coef=config.agent.ent_coef,
                    device=config.training.device,
                    seed=config.training.seed,
                )
        except ImportError as e:
            print(f"Warning: Could not create rl_a2c: {e}")

    # 5. Active Inference agent (using fast implementation)
    if "active_inference" in agent_list:
        try:
            from src.agents.active_inference_fast import HAS_PYMDP, ActiveInferenceFast
            if not HAS_PYMDP:
                print("Warning: pymdp not available, skipping active_inference")
            else:
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
            print(f"Warning: Could not create active_inference: {e}")

    return agents


def train_sb3_agent(
    agent,
    agent_name: str,
    config: Config,
    output_dir: Path,
    resume_checkpoint: Path | None = None,
) -> dict[str, list[float]]:
    """
    Train a stable-baselines3 based agent with checkpointing.

    Args:
        agent: The agent to train (must have create_model, train, save, load methods)
        agent_name: Name of the agent (for checkpoint naming)
        config: Training configuration
        output_dir: Output directory for models and checkpoints
        resume_checkpoint: Optional path to checkpoint for resuming training

    Returns:
        Dictionary with training rewards
    """
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    print(f"\nTraining {agent_name}...")

    # Create training environment
    env = ForagingGridEnv(
        grid_size=config.environment.grid_size,
        max_steps=config.environment.max_steps,
        initial_energy=config.environment.initial_energy,
        mode=EnvironmentMode.BASELINE,
        seed=config.training.seed,
    )

    # Create checkpoint directory for this agent
    checkpoint_dir = output_dir / "checkpoints" / agent_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Handle resume vs new training
    reset_num_timesteps = True
    total_timesteps = config.training.total_timesteps

    if resume_checkpoint is not None and agent_name in str(resume_checkpoint):
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
        name_prefix=agent_name,
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
    model_path = output_dir / "models" / f"{agent_name}.zip"
    agent.save(str(model_path))
    print(f"Saved {agent_name} model to {model_path}")

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


def find_agent_checkpoint(checkpoint_base_dir: Path, agent_name: str) -> Path | None:
    """Find the latest checkpoint for a specific agent."""
    agent_checkpoint_dir = checkpoint_base_dir / agent_name
    if agent_checkpoint_dir.exists():
        return find_latest_checkpoint(agent_checkpoint_dir)
    # Also check the old flat structure for backwards compatibility
    return find_latest_checkpoint(checkpoint_base_dir, pattern=f"{agent_name}_*_steps.zip")


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
        help="Specific agents to train (default: all). Options: heuristic, rl_ppo, rl_dqn, rl_a2c, active_inference",
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
    resume_mode = False
    checkpoint_base_dir = None
    if args.resume:
        resume_mode = True
        if args.resume.lower() == "latest":
            if not args.resume_dir:
                parser.error("--resume-dir is required when --resume=latest")
            checkpoint_base_dir = Path(args.resume_dir) / "checkpoints"
            print(f"Will resume from latest checkpoints in: {checkpoint_base_dir}")
        else:
            # Specific checkpoint path provided
            checkpoint_base_dir = Path(args.resume).parent

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
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Configuration: {config.experiment_name}")
    if resume_mode:
        print(f"Resume mode: enabled")

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
    agents = create_agents(config, env, args.agents)

    print(f"\nAgents to process: {list(agents.keys())}")

    # Training results
    training_results = {}
    evaluation_results = {}

    # Train and evaluate each agent
    for agent_name, agent in agents.items():
        print(f"\n{'='*50}")
        print(f"Processing agent: {agent_name}")
        print(f"{'='*50}")

        # Determine if this agent needs training
        if agent_name in SB3_AGENTS and hasattr(agent, 'create_model'):
            # Find checkpoint for this agent if resuming
            resume_checkpoint = None
            if resume_mode and checkpoint_base_dir:
                resume_checkpoint = find_agent_checkpoint(checkpoint_base_dir, agent_name)
                if resume_checkpoint:
                    print(f"Found checkpoint: {resume_checkpoint}")
                else:
                    print(f"No checkpoint found for {agent_name}, starting fresh")

            training_data = train_sb3_agent(
                agent, agent_name, config, output_dir,
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
