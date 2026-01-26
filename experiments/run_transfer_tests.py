#!/usr/bin/env python3
"""
Run transfer and generalization tests across agent types.

This script tests how well trained agents generalize to novel
environment configurations.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents import (
    ActiveInferenceFast,
    BaseAgent,
    HeuristicAgent,
    RLAgent,
)
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode
from src.tests.transfer_tests import (
    TransferConfig,
    run_transfer_test,
)


def load_trained_agent(
    agent_type: str,
    model_path: Path | None,
    env: ForagingGridEnv,
    seed: int = 42,
) -> BaseAgent:
    """Load a trained agent or create fresh one."""
    if agent_type == "heuristic":
        return HeuristicAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=seed,
        )
    elif agent_type == "rl_ppo":
        agent = RLAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=seed,
        )
        if model_path and model_path.exists():
            agent.load(str(model_path))
        return agent
    elif agent_type == "active_inference":
        return ActiveInferenceFast(
            observation_space=env.observation_space,
            action_space=env.action_space,
            grid_size=env.grid_size,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_custom_transfer_suite(
    include_scaled: bool = True,
    include_partial_obs: bool = True,
) -> tuple:
    """Create customized transfer test suite."""
    source = TransferConfig(
        name="baseline_20",
        mode=EnvironmentMode.BASELINE,
        grid_size=20,
        description="Standard 20x20 baseline",
    )

    targets = []

    # Grid size variations
    targets.append(TransferConfig(
        name="grid_25",
        mode=EnvironmentMode.BASELINE,
        grid_size=25,
        description="25x25 grid",
    ))

    targets.append(TransferConfig(
        name="grid_15",
        mode=EnvironmentMode.BASELINE,
        grid_size=15,
        description="15x15 grid",
    ))

    # Mode variations
    targets.append(TransferConfig(
        name="extinction",
        mode=EnvironmentMode.EXTINCTION,
        grid_size=20,
        description="Resource extinction",
    ))

    targets.append(TransferConfig(
        name="tradeoff",
        mode=EnvironmentMode.TRADEOFF,
        grid_size=20,
        description="Risk-reward tradeoff",
    ))

    # Scaled environments
    if include_scaled:
        targets.append(TransferConfig(
            name="scaled_30",
            mode=EnvironmentMode.BASELINE,
            grid_size=30,
            description="30x30 scaled environment",
        ))

    return source, targets


def run_comprehensive_transfer(
    agent_types: list[str],
    model_dir: Path | None = None,
    output_dir: Path | None = None,
    n_trials: int = 5,
    n_episodes: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run comprehensive transfer tests.

    Args:
        agent_types: List of agent types to test
        model_dir: Directory containing trained models
        output_dir: Output directory for results
        n_trials: Number of trials per test
        n_episodes: Episodes per configuration
        seed: Random seed

    Returns:
        Dictionary of all results
    """
    if output_dir is None:
        output_dir = Path("outputs/transfer_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create transfer suite
    source_config, target_configs = create_custom_transfer_suite()

    # Create reference environment
    ref_env = ForagingGridEnv(
        grid_size=source_config.grid_size,
        max_steps=500,
        mode=source_config.mode,
        seed=seed,
    )

    # Load agents
    agents = []
    for agent_type in agent_types:
        model_path = None
        if model_dir and agent_type == "rl_ppo":
            model_path = model_dir / f"{agent_type}_best.zip"

        agent = load_trained_agent(agent_type, model_path, ref_env, seed)
        agents.append(agent)

    # Run transfer tests
    print(f"Running transfer tests with {len(agents)} agents...")
    results = run_transfer_test(
        agents=agents,
        source_config=source_config,
        target_configs=target_configs,
        n_trials=n_trials,
        n_episodes=n_episodes,
        seed=seed,
    )

    # Compile results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "agent_types": agent_types,
            "n_trials": n_trials,
            "n_episodes": n_episodes,
            "seed": seed,
            "source": source_config.to_dict(),
            "targets": [t.to_dict() for t in target_configs],
        },
        "results": {name: r.to_dict() for name, r in results.items()},
    }

    # Save results
    results_path = output_dir / f"transfer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRANSFER TEST SUMMARY")
    print("=" * 60)

    for agent_name, result in results.items():
        print(f"\n{agent_name}:")
        print(f"  Source score: {result.raw_data['source_score']:.2f}")
        print(f"  Mean relative transfer: {result.raw_data['mean_relative_transfer']:.2%}")
        print(f"  Mean generalization gap: {result.raw_data['mean_generalization_gap']:.2f}")

        print("  Per-target:")
        for tr in result.raw_data['transfer_results']:
            target = tr['target']['name']
            rel_tf = tr['relative_transfer']
            print(f"    {target}: {rel_tf:.2%}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run transfer tests")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["heuristic", "rl_ppo", "active_inference"],
        help="Agent types to test",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/transfer_tests"),
        help="Output directory",
    )
    parser.add_argument("--n-trials", type=int, default=5, help="Number of trials")
    parser.add_argument("--n-episodes", type=int, default=20, help="Episodes per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_comprehensive_transfer(
        agent_types=args.agents,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
