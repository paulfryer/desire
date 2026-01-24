"""
Run predicate tests on trained agents.

Usage:
    python experiments/run_tests.py --config experiments/configs/default.yaml
    python experiments/run_tests.py --model-dir outputs/training_xxx

Debug mode:
    python experiments/run_tests.py --agent heuristic --test frustration --n_trials 1 --verbose
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode
from src.agents.base_agent import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.tests.frustration import FrustrationIntensificationTest
from src.tests.means_end import MeansEndDecouplingTest
from src.tests.base_test import TestResult
from src.analysis.metrics import compare_agents
from src.utils.config import Config, load_config


class TimeoutError(Exception):
    """Raised when a test times out."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out!")


def load_single_agent(
    agent_name: str,
    config: Config,
    model_dir: Optional[Path] = None,
    verbose: bool = False,
) -> BaseAgent:
    """Load a single agent by name."""
    env = ForagingGridEnv(
        grid_size=config.environment.grid_size,
        max_steps=config.environment.max_steps,
        mode=EnvironmentMode.BASELINE,
    )

    if verbose:
        print(f"  Creating {agent_name} agent...")

    if agent_name == "heuristic":
        agent = HeuristicAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=config.training.seed,
        )
    elif agent_name == "rl_ppo":
        from src.agents.rl_agent import RLAgent, HAS_SB3
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required")
        agent = RLAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=config.training.device,
            seed=config.training.seed,
        )
        if model_dir:
            model_path = model_dir / "models" / "rl_ppo.zip"
            if model_path.exists():
                agent.load(str(model_path))
                if verbose:
                    print(f"    Loaded model from {model_path}")
    elif agent_name == "active_inference":
        from src.agents.active_inference import ActiveInferenceAgent, HAS_PYMDP
        if not HAS_PYMDP:
            raise ImportError("pymdp required")
        agent = ActiveInferenceAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            planning_horizon=config.agent.planning_horizon,
            policy_precision=config.agent.policy_precision,
            grid_size=config.environment.grid_size,
            seed=config.training.seed,
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    env.close()
    return agent


def load_agents(
    config: Config,
    model_dir: Optional[Path] = None,
    agent_filter: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, BaseAgent]:
    """Load agent types, optionally filtered."""
    all_agent_names = ["heuristic", "rl_ppo", "active_inference"]

    if agent_filter:
        agent_names = [n for n in agent_filter if n in all_agent_names]
    else:
        agent_names = all_agent_names

    agents = {}
    for name in agent_names:
        try:
            agents[name] = load_single_agent(name, config, model_dir, verbose)
            if verbose:
                print(f"    {name}: OK")
        except Exception as e:
            print(f"    {name}: FAILED - {e}")

    return agents


def run_single_test_debug(
    agent: BaseAgent,
    agent_name: str,
    test_name: str,
    config: Config,
    n_trials: int = 1,
    timeout_sec: int = 60,
    verbose: bool = True,
) -> Optional[TestResult]:
    """Run a single test with verbose debugging output."""

    print(f"\n{'='*60}")
    print(f"DEBUG TEST: {test_name} on {agent_name}")
    print(f"{'='*60}")
    print(f"  Trials: {n_trials}")
    print(f"  Timeout: {timeout_sec}s per trial")

    start_time = time.time()

    if test_name == "frustration":
        # Create minimal test config
        test = FrustrationIntensificationTest(
            n_trials=n_trials,
            baseline_episodes=5,  # Reduced for debugging
            test_episodes=5,
            blockage_step=50,
            post_blockage_steps=50,
            grid_size=config.environment.grid_size,
            seed=config.training.seed,
        )
    elif test_name == "means_end":
        test = MeansEndDecouplingTest(
            n_trials=n_trials,
            training_episodes=10,  # Reduced for debugging
            transfer_episodes=5,
            grid_size=config.environment.grid_size,
            seed=config.training.seed,
        )
    else:
        print(f"  ERROR: Unknown test '{test_name}'")
        return None

    # Run with verbose output
    print(f"\n  Starting test...")

    try:
        # Manual trial loop with timeout checking
        for trial in range(n_trials):
            trial_start = time.time()
            print(f"\n  Trial {trial + 1}/{n_trials}:")

            # Run baseline episodes
            print(f"    Running baseline episodes...")
            baseline_env = test.create_baseline_env(seed=config.training.seed + trial)

            for ep in range(test.baseline_episodes):
                ep_start = time.time()
                obs, info = baseline_env.reset()
                agent.reset()

                step = 0
                done = False
                while not done:
                    action = agent.act(obs)
                    obs, reward, terminated, truncated, info = baseline_env.step(action)
                    agent.set_energy(baseline_env.agent_energy)
                    done = terminated or truncated
                    step += 1

                    # Heartbeat every 100 steps
                    if step % 100 == 0 and verbose:
                        print(f"      Episode {ep+1}, Step {step}...")

                    # Check timeout
                    if time.time() - trial_start > timeout_sec:
                        print(f"    TIMEOUT after {step} steps in episode {ep+1}")
                        baseline_env.close()
                        return None

                if verbose and (ep + 1) % 2 == 0:
                    elapsed = time.time() - ep_start
                    print(f"      Baseline episode {ep+1}/{test.baseline_episodes} done ({step} steps, {elapsed:.1f}s)")

            baseline_env.close()
            print(f"    Baseline complete.")

            # Run test episodes
            print(f"    Running test episodes...")
            test_env = test.create_test_env(seed=config.training.seed + trial + 1000)

            for ep in range(test.test_episodes):
                ep_start = time.time()
                obs, info = test_env.reset()
                agent.reset()

                step = 0
                done = False
                while not done:
                    action = agent.act(obs)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    agent.set_energy(test_env.agent_energy)
                    done = terminated or truncated
                    step += 1

                    if step % 100 == 0 and verbose:
                        print(f"      Episode {ep+1}, Step {step}...")

                    if time.time() - trial_start > timeout_sec:
                        print(f"    TIMEOUT after {step} steps in episode {ep+1}")
                        test_env.close()
                        return None

                if verbose and (ep + 1) % 2 == 0:
                    elapsed = time.time() - ep_start
                    print(f"      Test episode {ep+1}/{test.test_episodes} done ({step} steps, {elapsed:.1f}s)")

            test_env.close()

            trial_elapsed = time.time() - trial_start
            print(f"    Trial {trial + 1} complete in {trial_elapsed:.1f}s")

        total_elapsed = time.time() - start_time
        print(f"\n  All trials complete in {total_elapsed:.1f}s")

        # Now run the actual test to get the score
        print(f"\n  Computing final score...")
        result = test.run(agent)
        print(f"  Score: {result.score:.4f}")
        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_tests(
    agents: Dict[str, BaseAgent],
    config: Config,
    output_dir: Path,
    test_filter: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, TestResult]]:
    """Run predicate tests on agents."""
    all_results = {}

    tests_to_run = test_filter if test_filter else ["frustration", "means_end"]

    # Test 1: Frustration Intensification
    if "frustration" in tests_to_run:
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
            sys.stdout.flush()

            start = time.time()
            result = fi_test.run(agent)
            elapsed = time.time() - start

            fi_results[agent_name] = result
            print(f"    FI Score: {result.score:.3f} ({elapsed:.1f}s)")

        all_results["frustration_intensification"] = fi_results

    # Test 2: Means-End Decoupling
    if "means_end" in tests_to_run:
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
            sys.stdout.flush()

            start = time.time()
            result = med_test.run(agent)
            elapsed = time.time() - start

            med_results[agent_name] = result
            print(f"    MED Score: {result.score:.3f} ({elapsed:.1f}s)")

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
        "--agent",
        type=str,
        nargs="+",
        default=None,
        help="Specific agent(s) to test: heuristic, rl_ppo, active_inference",
    )
    parser.add_argument(
        "--test",
        type=str,
        nargs="+",
        default=None,
        help="Specific test(s) to run: frustration, means_end",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Override number of trials",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per trial in seconds (default: 300)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with progress updates",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with minimal episodes",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Config file not found: {config_path}, using defaults")
        config = Config()

    # Override config with CLI args
    if args.n_trials:
        config.test.n_trials = args.n_trials

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
    if args.verbose:
        print(f"Verbose mode: ON")
    if args.debug:
        print(f"Debug mode: ON (minimal episodes)")

    # Set random seed
    np.random.seed(config.training.seed)

    # Load agents
    print("\nLoading agents...")
    agents = load_agents(config, model_dir, args.agent, args.verbose)
    print(f"Loaded agents: {list(agents.keys())}")

    if not agents:
        print("ERROR: No agents loaded!")
        return

    # Debug mode - run minimal test with verbose output
    if args.debug:
        test_name = args.test[0] if args.test else "frustration"
        for agent_name, agent in agents.items():
            run_single_test_debug(
                agent=agent,
                agent_name=agent_name,
                test_name=test_name,
                config=config,
                n_trials=args.n_trials or 1,
                timeout_sec=args.timeout,
                verbose=True,
            )
        return

    # Run tests
    results = run_all_tests(
        agents, config, output_dir,
        test_filter=args.test,
        verbose=args.verbose,
    )

    if not results:
        print("No test results!")
        return

    # Analyze results
    print("\n" + "="*50)
    print("Statistical Analysis")
    print("="*50)
    analysis = analyze_results(results)

    # Save results
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


if __name__ == "__main__":
    main()
