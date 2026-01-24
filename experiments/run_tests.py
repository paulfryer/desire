"""
Run predicate tests on trained agents.

Usage:
    python experiments/run_tests.py --config experiments/configs/default.yaml
    python experiments/run_tests.py --model-dir outputs/training_xxx
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode
from src.agents.base_agent import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.tests.frustration import FrustrationIntensificationTest, run_frustration_test
from src.tests.means_end import MeansEndDecouplingTest, run_means_end_test
from src.tests.base_test import TestResult
from src.analysis.metrics import compare_agents
from src.utils.config import Config, load_config


def load_agents(config: Config, model_dir: Optional[Path] = None) -> Dict[str, BaseAgent]:
    """Load all agent types, including trained models if available."""
    # Create a dummy env for observation/action spaces
    env = ForagingGridEnv(
        grid_size=config.environment.grid_size,
        max_steps=config.environment.max_steps,
        mode=EnvironmentMode.BASELINE,
    )

    agents = {}

    # 1. Heuristic agent
    agents["heuristic"] = HeuristicAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=config.training.seed,
    )

    # 2. RL agent
    try:
        from src.agents.rl_agent import RLAgent, HAS_SB3
        if not HAS_SB3:
            raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")
        agent = RLAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=config.training.device,
            seed=config.training.seed,
        )
        # Try to load trained model
        if model_dir:
            model_path = model_dir / "models" / "rl_ppo.zip"
            if model_path.exists():
                agent.load(str(model_path))
                print(f"Loaded RL model from {model_path}")
            else:
                print(f"No trained RL model found at {model_path}")
        agents["rl_ppo"] = agent
    except ImportError as e:
        raise ImportError(f"Could not create RL agent: {e}")

    # 3. Active Inference agent
    try:
        from src.agents.active_inference import ActiveInferenceAgent, HAS_PYMDP
        if not HAS_PYMDP:
            raise ImportError("pymdp is required. Install with: pip install inferactively-pymdp")
        agents["active_inference"] = ActiveInferenceAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            planning_horizon=config.agent.planning_horizon,
            policy_precision=config.agent.policy_precision,
            grid_size=config.environment.grid_size,
            seed=config.training.seed,
        )
    except ImportError as e:
        raise ImportError(f"Could not create Active Inference agent: {e}")

    env.close()
    return agents


def run_all_tests(
    agents: Dict[str, BaseAgent],
    config: Config,
    output_dir: Path,
) -> Dict[str, Dict[str, TestResult]]:
    """Run all predicate tests on all agents."""
    all_results = {}

    # Test 1: Frustration Intensification
    print("\n" + "="*50)
    print("Running Frustration Intensification Test")
    print("="*50)

    fi_test = FrustrationIntensificationTest(
        n_trials=config.test.n_trials,
        baseline_episodes=config.test.baseline_episodes,
        test_episodes=config.test.test_episodes,
        blockage_step=config.test.blockage_step,
        post_blockage_steps=config.test.post_blockage_steps,
        grid_size=config.environment.grid_size,
        seed=config.training.seed,
        output_dir=str(output_dir / "frustration"),
    )

    fi_results = {}
    for agent_name, agent in agents.items():
        print(f"\n  Testing {agent_name}...")
        result = fi_test.run(agent)
        fi_results[agent_name] = result
        print(f"    FI Score: {result.score:.3f}")

    all_results["frustration_intensification"] = fi_results

    # Test 2: Means-End Decoupling
    print("\n" + "="*50)
    print("Running Means-End Decoupling Test")
    print("="*50)

    med_test = MeansEndDecouplingTest(
        n_trials=config.test.n_trials,
        training_episodes=config.test.training_episodes,
        transfer_episodes=config.test.transfer_episodes,
        grid_size=config.environment.grid_size,
        seed=config.training.seed,
        output_dir=str(output_dir / "means_end"),
    )

    med_results = {}
    for agent_name, agent in agents.items():
        print(f"\n  Testing {agent_name}...")
        result = med_test.run(agent)
        med_results[agent_name] = result
        print(f"    MED Score: {result.score:.3f}")

    all_results["means_end_decoupling"] = med_results

    return all_results


def analyze_results(
    results: Dict[str, Dict[str, TestResult]],
) -> Dict[str, Dict]:
    """Perform statistical analysis on results."""
    analysis = {}

    for test_name, test_results in results.items():
        print(f"\n  Analyzing {test_name}...")
        comparison = compare_agents(test_results, test_name)
        analysis[test_name] = comparison

        # Print key findings
        if "kruskal_wallis" in comparison:
            kw = comparison["kruskal_wallis"]
            sig = "SIGNIFICANT" if kw["significant"] else "not significant"
            print(f"    Kruskal-Wallis: H={kw['statistic']:.2f}, p={kw['p_value']:.4f} ({sig})")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Run predicate tests on agents")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--tests",
        type=str,
        nargs="+",
        default=None,
        help="Specific tests to run",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Config file not found: {config_path}, using defaults")
        config = Config()

    # Set directories
    model_dir = Path(args.model_dir) if args.model_dir else None

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.output_dir) / f"tests_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Model directory: {model_dir}")

    # Set random seed
    np.random.seed(config.training.seed)

    # Load agents
    print("\nLoading agents...")
    agents = load_agents(config, model_dir)
    print(f"Loaded agents: {list(agents.keys())}")

    # Run tests
    results = run_all_tests(agents, config, output_dir)

    # Analyze results
    print("\n" + "="*50)
    print("Statistical Analysis")
    print("="*50)
    analysis = analyze_results(results)

    # Save results
    # Convert TestResult objects to dicts for JSON serialization
    results_dict = {
        test_name: {
            agent_name: result.to_dict()
            for agent_name, result in test_results.items()
        }
        for test_name, test_results in results.items()
    }

    output_data = {
        "config": config.to_dict(),
        "results": results_dict,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)

    for test_name, test_results in results.items():
        print(f"\n{test_name.upper().replace('_', ' ')}:")
        for agent_name, result in test_results.items():
            std = result.raw_data.get("std_score", 0)
            print(f"  {agent_name}: {result.score:.3f} (+/- {std:.3f})")

    # Overall verdict
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)

    for test_name, test_analysis in analysis.items():
        if "pairwise_comparisons" in test_analysis:
            print(f"\n{test_name}:")
            for comparison, data in test_analysis["pairwise_comparisons"].items():
                if data["significant"]:
                    agents_compared = comparison.replace("_vs_", " vs ")
                    print(f"  * Significant difference: {agents_compared}")
                    print(f"    Effect size: {data['effect_size']:.3f}")


if __name__ == "__main__":
    main()
