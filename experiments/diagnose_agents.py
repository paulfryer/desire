"""
Diagnostic script to observe agent behavior in detail.
"""

from collections import Counter

import numpy as np

from src.agents.active_inference import HAS_PYMDP, ActiveInferenceAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.rl_agent import HAS_SB3, RLAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "CONSUME"]


def diagnose_agent(agent, env, n_steps=100, verbose=True):
    """Run agent and collect diagnostic information."""
    obs, info = env.reset()
    agent.reset()

    actions = []
    positions = []
    energies = []
    internal_states = []

    for _step in range(n_steps):
        action = agent.act(obs)
        actions.append(action)
        positions.append(obs["position"].copy())
        energies.append(obs["energy"][0])
        internal_states.append(agent.get_internal_state().copy())

        obs, reward, terminated, truncated, info = env.step(action)
        agent.set_energy(env.agent_energy)

        if terminated or truncated:
            break

    # Analysis
    action_counts = Counter(actions)
    unique_positions = len({tuple(p) for p in positions})

    if verbose:
        print(f"\n{'='*50}")
        print(f"Agent: {agent.name}")
        print(f"{'='*50}")
        print(f"Steps taken: {len(actions)}")
        print(f"Unique positions visited: {unique_positions}")
        print("\nAction distribution:")
        for a, name in enumerate(ACTION_NAMES):
            count = action_counts.get(a, 0)
            pct = 100 * count / len(actions) if actions else 0
            bar = "#" * int(pct / 5)
            print(f"  {name:8s}: {count:4d} ({pct:5.1f}%) {bar}")

        print(f"\nFirst 20 actions: {[ACTION_NAMES[a] for a in actions[:20]]}")

        # Check for repetitive patterns
        if len(actions) >= 10:
            # Check if same action repeated
            repeats = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i-1])
            print(f"\nConsecutive same-action: {repeats}/{len(actions)-1} ({100*repeats/(len(actions)-1):.1f}%)")

            # Check for short cycles
            for cycle_len in [2, 3, 4]:
                if len(actions) >= cycle_len * 3:
                    pattern = actions[:cycle_len]
                    matches = sum(1 for i in range(len(actions) - cycle_len)
                                if actions[i:i+cycle_len] == pattern)
                    if matches > len(actions) / cycle_len / 2:
                        print(f"  Possible {cycle_len}-step cycle detected: {[ACTION_NAMES[a] for a in pattern]}")

        # Energy trajectory
        print(f"\nEnergy: {energies[0]:.1f} -> {energies[-1]:.1f}")

        # Internal state analysis
        internal_states = np.array(internal_states)
        print(f"\nInternal state dimension: {internal_states.shape[1]}")

        # Check for internal state variation
        state_std = np.std(internal_states, axis=0)
        varying_dims = np.sum(state_std > 0.01)
        print(f"Dimensions with variation (std > 0.01): {varying_dims}/{internal_states.shape[1]}")

        if varying_dims < internal_states.shape[1] * 0.1:
            print("  WARNING: Very little internal state variation!")

    return {
        "actions": actions,
        "positions": positions,
        "energies": energies,
        "internal_states": internal_states,
        "action_counts": dict(action_counts),
        "unique_positions": unique_positions,
    }


def main():
    print("Agent Diagnostic Tool")
    print("=" * 60)

    # Create environment
    env = ForagingGridEnv(
        grid_size=15,
        max_steps=200,
        initial_energy=100.0,
        mode=EnvironmentMode.BASELINE,
        seed=42,
    )

    # Test each agent type
    print("\n" + "=" * 60)
    print("HEURISTIC AGENT")
    print("=" * 60)
    heuristic = HeuristicAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=42,
    )
    diagnose_agent(heuristic, env, n_steps=100)

    if HAS_SB3:
        print("\n" + "=" * 60)
        print("RL AGENT (untrained)")
        print("=" * 60)
        rl_agent = RLAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=42,
        )
        diagnose_agent(rl_agent, env, n_steps=100)

    if HAS_PYMDP:
        print("\n" + "=" * 60)
        print("ACTIVE INFERENCE AGENT")
        print("=" * 60)
        ai_agent = ActiveInferenceAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            planning_horizon=2,
            policy_precision=4.0,
            grid_size=15,
            seed=42,
        )
        diagnose_agent(ai_agent, env, n_steps=100)

        # Additional pymdp-specific diagnostics
        print("\n--- Active Inference Internals ---")
        obs, _ = env.reset()
        ai_agent.reset()

        # Check discretization
        disc_obs = ai_agent._discretize_observation(obs)
        print(f"Discretized observation: {disc_obs}")
        print(f"  Position state: {disc_obs[0]} (of {ai_agent.num_position_states})")
        print(f"  Energy state: {disc_obs[1]} (of {ai_agent.num_energy_states})")
        print(f"  Resource visible: {disc_obs[2]}")

        # Check beliefs after one action
        action = ai_agent.act(obs)
        print(f"\nFirst action chosen: {ACTION_NAMES[action]}")

        if ai_agent.beliefs is not None:
            print("\nBeliefs after first action:")
            for i, b in enumerate(ai_agent.beliefs):
                print(f"  Factor {i}: shape={b.shape}, max={b.max():.3f}, entropy={-np.sum(b * np.log(b + 1e-10)):.3f}")

    env.close()
    print("\n" + "=" * 60)
    print("Diagnostic complete!")


if __name__ == "__main__":
    main()
