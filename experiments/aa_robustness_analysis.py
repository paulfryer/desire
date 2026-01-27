"""
Anticipatory Activation Robustness Analysis

Tests whether anticipation patterns hold under different internal state operationalizations:
- Hidden layer activations (original)
- Q-values / Value head outputs (new)
- Policy features for actor-critics (new)

This addresses the measurement confound: comparing hidden activations (RL) to belief states (AI).
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode


def load_agents(model_dir: Path):
    """Load trained agents."""
    from src.agents.heuristic import HeuristicAgent
    from src.agents.rl_agent import RLAgent
    from src.agents.rl_dqn import DQNAgent
    from src.agents.rl_a2c import A2CAgent
    from src.agents.active_inference import ActiveInferenceAgent
    from gymnasium import spaces

    # Create observation and action spaces
    observation_space = spaces.Dict({
        "local_view": spaces.Box(low=0, high=1, shape=(11, 11, 3), dtype=np.float32),
        "energy": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        "step": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
    })
    action_space = spaces.Discrete(6)

    agents = {}
    models_dir = model_dir / "models"

    # Load PPO
    ppo_path = models_dir / "rl_ppo.zip"
    if ppo_path.exists():
        agent = RLAgent(observation_space, action_space, name="rl_ppo")
        agent.load(str(ppo_path))
        agents["rl_ppo"] = agent
        print(f"  Loaded rl_ppo")

    # Load DQN
    dqn_path = models_dir / "rl_dqn.zip"
    if dqn_path.exists():
        agent = DQNAgent(observation_space, action_space, name="rl_dqn")
        agent.load(str(dqn_path))
        agents["rl_dqn"] = agent
        print(f"  Loaded rl_dqn")

    # Load A2C
    a2c_path = models_dir / "rl_a2c.zip"
    if a2c_path.exists():
        agent = A2CAgent(observation_space, action_space, name="rl_a2c")
        agent.load(str(a2c_path))
        agents["rl_a2c"] = agent
        print(f"  Loaded rl_a2c")

    return agents


def get_dqn_q_values(agent, observation):
    """Get Q-values from DQN for all actions."""
    if agent.model is None:
        return None

    obs = agent._prepare_observation(observation)
    obs_tensor = agent.model.policy.obs_to_tensor(obs)[0]

    with torch.no_grad():
        q_values = agent.model.policy.q_net(obs_tensor)
        return q_values.cpu().numpy().flatten()


def get_actor_critic_value(agent, observation):
    """Get value head output V(s) from PPO or A2C."""
    if agent.model is None:
        return None

    obs = agent._prepare_observation(observation)
    obs_tensor = agent.model.policy.obs_to_tensor(obs)[0]

    with torch.no_grad():
        value = agent.model.policy.predict_values(obs_tensor)
        return float(value.cpu().numpy().flatten()[0])


def get_actor_critic_policy_features(agent, observation):
    """Get policy head features (before action distribution) from PPO or A2C."""
    if agent.model is None or agent._feature_extractor is None:
        return None

    device = next(agent._feature_extractor.parameters()).device
    obs_tensors = {
        k: torch.tensor(v, dtype=torch.float32, device=device)
        for k, v in observation.items()
    }

    with torch.no_grad():
        # Get shared features
        features = agent._feature_extractor(obs_tensors)

        # Get policy network output (before softmax)
        # For PPO/A2C with net_arch, the mlp_extractor processes features
        latent_pi, latent_vf = agent.model.policy.mlp_extractor(features)

        return latent_pi.cpu().numpy().flatten()


def run_aa_test_multi_operationalization(agent, agent_name, n_trials=10, n_episodes=30, seed=42):
    """
    Run AA test with multiple internal state operationalizations.

    Returns correlations for:
    - hidden_activation: Original (hidden layer magnitude)
    - q_value / value_head: Q-value (DQN) or V(s) (PPO/A2C)
    - policy_features: Policy head features (PPO/A2C only)
    """
    rng = np.random.default_rng(seed)
    results = {
        "hidden_activation": [],
        "value_estimate": [],
        "policy_features": [],
    }

    is_dqn = agent_name == "rl_dqn"

    for trial in range(n_trials):
        trial_seed = int(rng.integers(0, 2**31))

        # Collect data points
        hidden_data = []  # (distance, activation_magnitude)
        value_data = []   # (distance, q_value or V(s))
        policy_data = []  # (distance, policy_feature_magnitude)

        for ep in range(n_episodes):
            env = ForagingGridEnv(
                grid_size=20,
                max_steps=500,
                initial_energy=150.0,
                max_energy=150.0,
                energy_decay_rate=0.3,
                mode=EnvironmentMode.BASELINE,
                seed=trial_seed + ep,
            )

            obs, info = env.reset()
            agent.reset()
            done = False

            while not done:
                # Get distance to nearest resource
                distance = info.get("nearest_resource_distance")
                if distance is None:
                    action = agent.act(obs)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    continue

                # 1. Hidden activation (original)
                action = agent.act(obs)
                internal_state = agent.get_internal_state()
                hidden_magnitude = np.linalg.norm(internal_state)
                hidden_data.append((distance, hidden_magnitude))

                # 2. Value estimate (Q-value for DQN, V(s) for others)
                if is_dqn:
                    q_values = get_dqn_q_values(agent, obs)
                    if q_values is not None:
                        max_q = np.max(q_values)
                        value_data.append((distance, max_q))
                else:
                    value = get_actor_critic_value(agent, obs)
                    if value is not None:
                        value_data.append((distance, value))

                # 3. Policy features (PPO/A2C only)
                if not is_dqn:
                    policy_feat = get_actor_critic_policy_features(agent, obs)
                    if policy_feat is not None:
                        policy_magnitude = np.linalg.norm(policy_feat)
                        policy_data.append((distance, policy_magnitude))

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            env.close()

        # Compute correlations for this trial
        if len(hidden_data) > 10:
            distances = np.array([d for d, _ in hidden_data])
            activations = np.array([a for _, a in hidden_data])
            corr = np.corrcoef(distances, activations)[0, 1]
            if not np.isnan(corr):
                results["hidden_activation"].append(corr)

        if len(value_data) > 10:
            distances = np.array([d for d, _ in value_data])
            values = np.array([v for _, v in value_data])
            corr = np.corrcoef(distances, values)[0, 1]
            if not np.isnan(corr):
                results["value_estimate"].append(corr)

        if len(policy_data) > 10:
            distances = np.array([d for d, _ in policy_data])
            features = np.array([f for _, f in policy_data])
            corr = np.corrcoef(distances, features)[0, 1]
            if not np.isnan(corr):
                results["policy_features"].append(corr)

        print(f"    Trial {trial + 1}/{n_trials}: "
              f"hidden={results['hidden_activation'][-1] if results['hidden_activation'] else 'N/A':.3f}, "
              f"value={results['value_estimate'][-1] if results['value_estimate'] else 'N/A':.3f}")

    return results


def classify_anticipation(r):
    """Classify anticipation type based on correlation."""
    if r < -0.3:
        return "True"
    elif r > 0.3:
        return "Inverse"
    elif r < -0.1:
        return "Weak"
    else:
        return "None"


def main():
    print("=" * 70)
    print("ANTICIPATORY ACTIVATION ROBUSTNESS ANALYSIS")
    print("Testing multiple internal state operationalizations")
    print("=" * 70)

    model_dir = Path("outputs/training_20260126_145317")
    output_dir = Path("outputs/aa_robustness")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel directory: {model_dir}")
    print(f"Output directory: {output_dir}")

    # Load agents
    print("\nLoading agents...")
    agents = load_agents(model_dir)

    if not agents:
        print("ERROR: No agents loaded!")
        return

    # Run analysis
    results = {}
    n_trials = 10
    n_episodes = 30

    for agent_name, agent in agents.items():
        print(f"\n{'=' * 50}")
        print(f"Testing {agent_name}")
        print(f"{'=' * 50}")

        agent_results = run_aa_test_multi_operationalization(
            agent, agent_name, n_trials=n_trials, n_episodes=n_episodes
        )

        results[agent_name] = {
            op: {
                "correlations": corrs,
                "mean": float(np.mean(corrs)) if corrs else None,
                "std": float(np.std(corrs)) if corrs else None,
                "n": len(corrs),
            }
            for op, corrs in agent_results.items()
        }

    # Generate summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nCorrelation between distance and internal state (mean +/- std):")
    print("-" * 70)
    print(f"{'Agent':<12} {'Hidden Act.':<18} {'Value Est.':<18} {'Policy Feat.':<18}")
    print("-" * 70)

    for agent_name in ["rl_dqn", "rl_ppo", "rl_a2c"]:
        if agent_name not in results:
            continue

        r = results[agent_name]

        hidden = r["hidden_activation"]
        value = r["value_estimate"]
        policy = r["policy_features"]

        hidden_str = f"{hidden['mean']:.3f} +/- {hidden['std']:.3f}" if hidden["mean"] else "N/A"
        value_str = f"{value['mean']:.3f} +/- {value['std']:.3f}" if value["mean"] else "N/A"
        policy_str = f"{policy['mean']:.3f} +/- {policy['std']:.3f}" if policy["mean"] else "N/A"

        print(f"{agent_name:<12} {hidden_str:<18} {value_str:<18} {policy_str:<18}")

    # Classification table
    print("\n" + "-" * 70)
    print("Anticipation Type Classification:")
    print("-" * 70)
    print(f"{'Agent':<12} {'Hidden Act.':<18} {'Value Est.':<18} {'Policy Feat.':<18}")
    print("-" * 70)

    for agent_name in ["rl_dqn", "rl_ppo", "rl_a2c"]:
        if agent_name not in results:
            continue

        r = results[agent_name]

        hidden_type = classify_anticipation(r["hidden_activation"]["mean"]) if r["hidden_activation"]["mean"] else "N/A"
        value_type = classify_anticipation(r["value_estimate"]["mean"]) if r["value_estimate"]["mean"] else "N/A"
        policy_type = classify_anticipation(r["policy_features"]["mean"]) if r["policy_features"]["mean"] else "N/A"

        print(f"{agent_name:<12} {hidden_type:<18} {value_type:<18} {policy_type:<18}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    findings = []

    # DQN: Compare hidden vs Q-values
    if "rl_dqn" in results:
        dqn = results["rl_dqn"]
        if dqn["hidden_activation"]["mean"] and dqn["value_estimate"]["mean"]:
            hidden_r = dqn["hidden_activation"]["mean"]
            qval_r = dqn["value_estimate"]["mean"]
            if classify_anticipation(hidden_r) == classify_anticipation(qval_r):
                findings.append(f"DQN: Q-values show same pattern as hidden activations "
                              f"(hidden r={hidden_r:.2f}, Q-value r={qval_r:.2f})")
            else:
                findings.append(f"DQN: Q-values differ from hidden activations! "
                              f"(hidden r={hidden_r:.2f} [{classify_anticipation(hidden_r)}], "
                              f"Q-value r={qval_r:.2f} [{classify_anticipation(qval_r)}])")

    # PPO: Compare value head vs policy features
    if "rl_ppo" in results:
        ppo = results["rl_ppo"]
        if ppo["value_estimate"]["mean"] and ppo["policy_features"]["mean"]:
            val_r = ppo["value_estimate"]["mean"]
            pol_r = ppo["policy_features"]["mean"]
            findings.append(f"PPO: Value head r={val_r:.2f} [{classify_anticipation(val_r)}], "
                          f"Policy features r={pol_r:.2f} [{classify_anticipation(pol_r)}]")

    # A2C: Compare value head vs policy features
    if "rl_a2c" in results:
        a2c = results["rl_a2c"]
        if a2c["value_estimate"]["mean"] and a2c["policy_features"]["mean"]:
            val_r = a2c["value_estimate"]["mean"]
            pol_r = a2c["policy_features"]["mean"]
            findings.append(f"A2C: Value head r={val_r:.2f} [{classify_anticipation(val_r)}], "
                          f"Policy features r={pol_r:.2f} [{classify_anticipation(pol_r)}]")

    for finding in findings:
        print(f"\n- {finding}")

    # Save results
    output_file = output_dir / "robustness_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "results": results,
            "findings": findings,
            "timestamp": datetime.now().isoformat(),
            "n_trials": n_trials,
            "n_episodes": n_episodes,
        }, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_file}")

    # Generate LaTeX table for paper
    latex = generate_latex_table(results)
    latex_file = output_dir / "robustness_table.tex"
    with open(latex_file, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to: {latex_file}")


def generate_latex_table(results):
    """Generate LaTeX table for paper appendix."""
    latex = r"""\begin{table}[t]
\caption{Robustness analysis: Anticipation patterns under different internal state operationalizations. All values are Pearson correlations between distance-to-goal and the specified representation.}
\label{tab:robustness}
\centering
\small
\begin{tabular}{lccc}
\toprule
Agent & Hidden Activation & Value Estimate & Policy Features \\
\midrule
"""

    for agent_name in ["rl_dqn", "rl_ppo", "rl_a2c"]:
        if agent_name not in results:
            continue

        r = results[agent_name]
        label = {"rl_dqn": "DQN", "rl_ppo": "PPO", "rl_a2c": "A2C"}[agent_name]

        hidden = f"${r['hidden_activation']['mean']:.2f}$" if r["hidden_activation"]["mean"] else "---"
        value = f"${r['value_estimate']['mean']:.2f}$" if r["value_estimate"]["mean"] else "---"
        policy = f"${r['policy_features']['mean']:.2f}$" if r["policy_features"]["mean"] else "---"

        latex += f"{label} & {hidden} & {value} & {policy} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


if __name__ == "__main__":
    main()
