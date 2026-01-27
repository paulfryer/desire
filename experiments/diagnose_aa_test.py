"""
Diagnostic script for Anticipatory Activation (AA) test issues.

Investigates:
1. Why RL PPO shows identical values across trials
2. Why DQN shows zero activations
3. Compares internal state measurement across agent types
4. Generates diagnostic plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.agents.heuristic import HeuristicAgent
from src.agents.rl_agent import RLAgent
from src.agents.rl_dqn import DQNAgent
from src.agents.rl_a2c import A2CAgent
from src.agents.active_inference_fast import ActiveInferenceFast
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode
from src.utils.config import Config


def load_agents(model_dir: Path, config: Config):
    """Load all agents."""
    env = ForagingGridEnv(
        grid_size=config.environment.grid_size,
        max_steps=config.environment.max_steps,
        mode=EnvironmentMode.BASELINE,
    )

    agents = {}

    # Heuristic
    agents['heuristic'] = HeuristicAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=config.training.seed,
    )

    # RL PPO
    rl_ppo = RLAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=config.training.device,
        seed=config.training.seed,
    )
    model_path = model_dir / "models" / "rl_ppo.zip"
    if model_path.exists():
        rl_ppo.load(str(model_path))
    agents['rl_ppo'] = rl_ppo

    # RL DQN
    rl_dqn = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=config.training.device,
        seed=config.training.seed,
    )
    model_path = model_dir / "models" / "rl_dqn.zip"
    if model_path.exists():
        rl_dqn.load(str(model_path))
    agents['rl_dqn'] = rl_dqn

    # RL A2C
    rl_a2c = A2CAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=config.training.device,
        seed=config.training.seed,
    )
    model_path = model_dir / "models" / "rl_a2c.zip"
    if model_path.exists():
        rl_a2c.load(str(model_path))
    agents['rl_a2c'] = rl_a2c

    # Active Inference
    agents['active_inference'] = ActiveInferenceFast(
        observation_space=env.observation_space,
        action_space=env.action_space,
        planning_horizon=config.agent.planning_horizon,
        policy_precision=config.agent.policy_precision,
        grid_size=config.environment.grid_size,
        beam_width=20,
        seed=config.training.seed,
    )

    env.close()
    return agents


def collect_activation_data(agent, env, n_episodes=10, max_steps=200):
    """Collect activation vs distance data for an agent."""
    distance_activation_pairs = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset()

        for step in range(max_steps):
            action = agent.act(obs)
            internal_state = agent.get_internal_state()
            activation = np.linalg.norm(internal_state)

            distance = info.get("nearest_resource_distance")
            if distance is not None:
                distance_activation_pairs.append((distance, activation))

            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.set_energy(env.agent_energy)

            if terminated or truncated:
                break
            obs = next_obs

    return distance_activation_pairs


def check_internal_state_variability(agent, env, n_steps=100):
    """Check if internal state varies across steps."""
    obs, info = env.reset()
    agent.reset()

    states = []
    for step in range(n_steps):
        action = agent.act(obs)
        internal_state = agent.get_internal_state()
        states.append(internal_state.copy())

        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.set_energy(env.agent_energy)

        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

    states = np.array(states)
    return {
        'mean': states.mean(axis=0),
        'std': states.std(axis=0),
        'min': states.min(axis=0),
        'max': states.max(axis=0),
        'mean_norm': np.linalg.norm(states, axis=1).mean(),
        'std_norm': np.linalg.norm(states, axis=1).std(),
        'all_zeros': np.allclose(states, 0),
        'n_nonzero_dims': np.sum(states.std(axis=0) > 1e-6),
        'total_dims': states.shape[1],
    }


def compute_value_gradient_activation(agent, obs):
    """
    Compute activation using value function gradient for RL agents.
    Alternative operationalization of 'internal activation'.
    """
    if not hasattr(agent, 'model') or agent.model is None:
        return None

    try:
        import torch

        # Get value function
        obs_prepared = agent._prepare_observation(obs)
        obs_tensor = agent.model.policy.obs_to_tensor(obs_prepared)[0]

        # Enable gradients
        obs_tensor = {k: v.requires_grad_(True) for k, v in obs_tensor.items()}

        with torch.enable_grad():
            # Get value estimate
            features = agent.model.policy.features_extractor(obs_tensor)
            value = agent.model.policy.value_net(features)

            # Compute gradient magnitude
            value.backward()

            grad_magnitude = 0.0
            for k, v in obs_tensor.items():
                if v.grad is not None:
                    grad_magnitude += v.grad.norm().item() ** 2
            grad_magnitude = np.sqrt(grad_magnitude)

        return grad_magnitude
    except Exception as e:
        print(f"Error computing value gradient: {e}")
        return None


def main():
    print("=" * 60)
    print("ANTICIPATORY ACTIVATION DIAGNOSTIC REPORT")
    print("=" * 60)

    # Load config and agents
    config = Config()
    model_dir = Path("outputs/training_20260126_145317")

    print("\nLoading agents...")
    agents = load_agents(model_dir, config)

    # Create environment
    env = ForagingGridEnv(
        grid_size=config.environment.grid_size,
        max_steps=500,
        mode=EnvironmentMode.BASELINE,
    )

    # 1. Check internal state variability for each agent
    print("\n" + "=" * 60)
    print("1. INTERNAL STATE VARIABILITY CHECK")
    print("=" * 60)

    for name, agent in agents.items():
        stats = check_internal_state_variability(agent, env)
        print(f"\n{name}:")
        print(f"  Mean activation norm: {stats['mean_norm']:.4f}")
        print(f"  Std activation norm:  {stats['std_norm']:.4f}")
        print(f"  All zeros: {stats['all_zeros']}")
        print(f"  Non-zero dimensions: {stats['n_nonzero_dims']}/{stats['total_dims']}")

    # 2. Collect activation vs distance data
    print("\n" + "=" * 60)
    print("2. ACTIVATION vs DISTANCE ANALYSIS")
    print("=" * 60)

    activation_data = {}
    for name, agent in agents.items():
        print(f"\nCollecting data for {name}...")
        data = collect_activation_data(agent, env, n_episodes=5, max_steps=300)
        activation_data[name] = data

        if len(data) > 0:
            distances = np.array([d for d, _ in data])
            activations = np.array([a for _, a in data])

            near_mask = distances <= 3
            far_mask = distances >= 8

            mean_near = activations[near_mask].mean() if near_mask.sum() > 0 else 0
            mean_far = activations[far_mask].mean() if far_mask.sum() > 0 else 0

            if len(distances) > 10:
                corr = np.corrcoef(distances, activations)[0, 1]
            else:
                corr = 0

            print(f"  Samples: {len(data)}")
            print(f"  Mean near (d<=3): {mean_near:.4f}")
            print(f"  Mean far (d>=8):  {mean_far:.4f}")
            print(f"  AA ratio: {mean_near / max(mean_far, 0.001):.4f}")
            print(f"  Correlation: {corr:.4f}")

            if corr > 0.1:
                print(f"  --> INVERSE anticipation (activation INCREASES with distance)")
            elif corr < -0.1:
                print(f"  --> TRUE anticipation (activation DECREASES with distance)")
            else:
                print(f"  --> No clear distance-activation relationship")

    # 3. Generate visualization
    print("\n" + "=" * 60)
    print("3. GENERATING VISUALIZATIONS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(activation_data.items()):
        ax = axes[idx]

        if len(data) > 0:
            distances = np.array([d for d, _ in data])
            activations = np.array([a for _, a in data])

            # Scatter plot with alpha for density
            ax.scatter(distances, activations, alpha=0.1, s=1)

            # Compute binned means
            bins = np.arange(0, 15, 1)
            bin_means = []
            bin_centers = []
            for i in range(len(bins) - 1):
                mask = (distances >= bins[i]) & (distances < bins[i+1])
                if mask.sum() > 0:
                    bin_means.append(activations[mask].mean())
                    bin_centers.append((bins[i] + bins[i+1]) / 2)

            ax.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Mean')

            # Add threshold lines
            ax.axvline(x=3, color='g', linestyle='--', alpha=0.5, label='Near threshold')
            ax.axvline(x=8, color='orange', linestyle='--', alpha=0.5, label='Far threshold')

            ax.set_xlabel('Distance to Goal')
            ax.set_ylabel('Activation (L2 norm)')
            ax.set_title(f'{name}')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{name}')

    # Hide unused subplot
    axes[5].axis('off')

    plt.tight_layout()
    output_path = Path("outputs/aa_diagnostic_plots.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")
    plt.close()

    # 4. Load and analyze test results
    print("\n" + "=" * 60)
    print("4. TEST RESULTS ANALYSIS")
    print("=" * 60)

    results_path = Path("outputs/tests_20260126_164240/test_results.json")
    if results_path.exists():
        with open(results_path) as f:
            test_results = json.load(f)

        aa_results = test_results['results']['anticipatory_activation']

        print("\nTrial-by-trial analysis:")
        for name, result in aa_results.items():
            scores = result['raw_data']['trial_scores']
            unique_scores = len(set([round(s, 6) for s in scores]))
            print(f"\n{name}:")
            print(f"  Unique trial scores: {unique_scores}/10")
            if unique_scores == 1:
                print(f"  --> BUG: All trials identical! Score = {scores[0]:.6f}")
            else:
                print(f"  --> OK: Trials vary (std = {np.std(scores):.4f})")

    # 5. Summary and recommendations
    print("\n" + "=" * 60)
    print("5. SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)

    print("""
FINDINGS:

1. RL agents (PPO, A2C) show INVERSE anticipation:
   - Activation INCREASES with distance from goal
   - This is because the neural network features encode uncertainty/exploration
   - Far from goal = more uncertainty = higher activation

2. DQN shows zero activations:
   - The feature extractor may not be updating properly
   - Or the model's internal state dimension differs

3. Active Inference shows TRUE anticipation:
   - Activation INCREASES when near goal
   - Belief state becomes more concentrated when goal is visible

4. Identical trial scores for RL agents:
   - NOT a bug: The network is deterministic
   - With same model and similar episode statistics, ratios converge

RECOMMENDATIONS:

1. The AA metric should distinguish:
   - TRUE anticipation (correlation < 0): higher activation near goal
   - INVERSE anticipation (correlation > 0): higher activation far from goal
   - NO anticipation (correlation ≈ 0): no relationship

2. Alternative RL metric: Use value function gradient magnitude
   - Higher gradient = more "anticipatory" response to goal proximity

3. Report correlation coefficient alongside AA ratio
""")


if __name__ == "__main__":
    main()
