"""
Robustness check: Run AA test with multiple seeds to verify consistency.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode
from src.agents.heuristic import HeuristicAgent
from src.tests.anticipation import AnticipatoryActivationTest
from src.utils.config import Config, load_config


def run_aa_with_seed(config: Config, seed: int, n_trials: int = 10) -> Dict[str, float]:
    """Run AA test for all agents with a specific seed."""
    print(f"\n  Seed {seed}:")

    # Create environment for agent initialization
    env = ForagingGridEnv(
        grid_size=config.environment.grid_size,
        max_steps=config.environment.max_steps,
        mode=EnvironmentMode.BASELINE,
    )

    # Create agents
    agents = {}

    # Heuristic
    agents['heuristic'] = HeuristicAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=seed,
    )

    # RL
    try:
        from src.agents.rl_agent import RLAgent
        agents['rl_ppo'] = RLAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device='auto',
            seed=seed,
        )
    except ImportError:
        pass

    # Active Inference
    try:
        from src.agents.active_inference_fast import ActiveInferenceFast
        agents['active_inference'] = ActiveInferenceFast(
            observation_space=env.observation_space,
            action_space=env.action_space,
            planning_horizon=config.agent.planning_horizon,
            policy_precision=config.agent.policy_precision,
            grid_size=config.environment.grid_size,
            beam_width=20,
            seed=seed,
        )
    except ImportError:
        pass

    env.close()

    # Run AA test
    aa_test = AnticipatoryActivationTest(
        n_trials=n_trials,
        baseline_episodes=config.test.baseline_episodes,
        test_episodes=config.test.test_episodes,
        grid_size=config.environment.grid_size,
        seed=seed,
    )

    results = {}
    for agent_name, agent in agents.items():
        result = aa_test.run(agent)
        results[agent_name] = result.score
        print(f"    {agent_name}: {result.score:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Robustness check with multiple seeds')
    parser.add_argument('--config', type=str, default='experiments/configs/quick_test.yaml')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--output', type=str, default='outputs/robustness_check.json')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = Config()

    print(f"Running robustness check with seeds: {args.seeds}")
    print(f"Trials per seed: {args.n_trials}")

    all_results = {}
    for seed in args.seeds:
        all_results[seed] = run_aa_with_seed(config, seed, args.n_trials)

    # Analyze consistency
    print("\n" + "=" * 50)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 50)

    agents = list(all_results[args.seeds[0]].keys())

    consistency = {}
    for agent in agents:
        scores = [all_results[seed][agent] for seed in args.seeds]
        consistency[agent] = {
            'scores_by_seed': {str(seed): all_results[seed][agent] for seed in args.seeds},
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'range': float(np.max(scores) - np.min(scores)),
        }
        print(f"\n{agent}:")
        print(f"  Scores: {[f'{s:.3f}' for s in scores]}")
        print(f"  Mean: {consistency[agent]['mean']:.3f} (SD: {consistency[agent]['std']:.3f})")
        print(f"  Range: {consistency[agent]['range']:.3f}")

    # Check if AA dissociation holds across all seeds
    aa_dissociation_holds = True
    for seed in args.seeds:
        rl_score = all_results[seed].get('rl_ppo', 0)
        ai_score = all_results[seed].get('active_inference', 0)
        heur_score = all_results[seed].get('heuristic', 0)

        # Check: RL should be much lower than both AI and heuristic
        if rl_score > 0.5 or ai_score < 0.5 or heur_score < 0.5:
            aa_dissociation_holds = False
            break

    print(f"\n{'=' * 50}")
    print(f"AA DISSOCIATION CONSISTENT ACROSS SEEDS: {'YES' if aa_dissociation_holds else 'NO'}")
    print(f"{'=' * 50}")

    # Save results
    output = {
        'seeds': args.seeds,
        'n_trials_per_seed': args.n_trials,
        'results_by_seed': {str(k): v for k, v in all_results.items()},
        'consistency': consistency,
        'aa_dissociation_holds': aa_dissociation_holds,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
