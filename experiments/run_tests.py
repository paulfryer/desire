"""
Run predicate tests on trained agents.

Usage:
    python experiments/run_tests.py --config experiments/configs/default.yaml
    python experiments/run_tests.py --model-dir outputs/training_xxx

Resume from checkpoint (after crash or interruption):
    python experiments/run_tests.py --model-dir outputs/training_xxx --output-dir outputs/tests_xxx --resume

With GPU temperature monitoring (prevents overheating):
    python experiments/run_tests.py --model-dir outputs/training_xxx --resume --gpu-cooldown --gpu-max-temp 80

Debug mode:
    python experiments/run_tests.py --agent heuristic --test frustration --n_trials 1 --verbose

Supported agents: heuristic, rl_ppo, rl_dqn, rl_a2c, active_inference
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Suppress pynvml deprecation warning from torch.cuda
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")

import numpy as np

from src.agents.base_agent import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.analysis.metrics import compare_agents
from src.environments.foraging_grid import ForagingGridEnv
from src.environments.modes import EnvironmentMode
from src.tests.anticipation import AnticipatoryActivationTest
from src.tests.base_test import TestResult
from src.tests.extinction import ExtinctionResistanceTest
from src.tests.frustration import FrustrationIntensificationTest
from src.tests.means_end import MeansEndDecouplingTest
from src.tests.opportunity_cost import OpportunityCostSensitivityTest
from src.tests.state_dependent import StateDependentValuationTest
from src.utils.config import Config, load_config


class TimeoutError(Exception):
    """Raised when a test times out."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out!")


def load_single_agent(
    agent_name: str,
    config: Config,
    model_dir: Path | None = None,
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
        from src.agents.rl_agent import HAS_SB3, RLAgent
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
    elif agent_name == "rl_dqn":
        from src.agents.rl_dqn import HAS_SB3, DQNAgent
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required")
        agent = DQNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=config.training.device,
            seed=config.training.seed,
        )
        if model_dir:
            model_path = model_dir / "models" / "rl_dqn.zip"
            if model_path.exists():
                agent.load(str(model_path))
                if verbose:
                    print(f"    Loaded model from {model_path}")
    elif agent_name == "rl_a2c":
        from src.agents.rl_a2c import HAS_SB3, A2CAgent
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required")
        agent = A2CAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=config.training.device,
            seed=config.training.seed,
        )
        if model_dir:
            model_path = model_dir / "models" / "rl_a2c.zip"
            if model_path.exists():
                agent.load(str(model_path))
                if verbose:
                    print(f"    Loaded model from {model_path}")
    elif agent_name == "active_inference":
        from src.agents.active_inference_fast import HAS_PYMDP, ActiveInferenceFast
        if not HAS_PYMDP:
            raise ImportError("pymdp required")
        agent = ActiveInferenceFast(
            observation_space=env.observation_space,
            action_space=env.action_space,
            planning_horizon=config.agent.planning_horizon,
            policy_precision=config.agent.policy_precision,
            grid_size=config.environment.grid_size,
            beam_width=20,  # Fast beam search
            seed=config.training.seed,
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    env.close()
    return agent


def load_agents(
    config: Config,
    model_dir: Path | None = None,
    agent_filter: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, BaseAgent]:
    """Load agent types, optionally filtered."""
    all_agent_names = ["heuristic", "rl_ppo", "rl_dqn", "rl_a2c", "active_inference"]

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


def save_test_checkpoint(
    output_dir: Path,
    completed_results: dict[str, dict[str, TestResult]],
) -> None:
    """Save checkpoint of completed test results."""
    checkpoint_path = output_dir / "test_checkpoint.json"
    checkpoint_data = {
        "completed": {
            test_name: {
                agent_name: result.to_dict()
                for agent_name, result in test_results.items()
            }
            for test_name, test_results in completed_results.items()
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2, default=str)


def load_test_checkpoint(output_dir: Path) -> dict[str, dict[str, dict]] | None:
    """Load checkpoint if exists."""
    checkpoint_path = output_dir / "test_checkpoint.json"
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path, "r") as f:
        data = json.load(f)
    return data.get("completed", {})


def run_single_test_debug(
    agent: BaseAgent,
    agent_name: str,
    test_name: str,
    config: Config,
    n_trials: int = 1,
    timeout_sec: int = 60,
    verbose: bool = True,
) -> TestResult | None:
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
    print("\n  Starting test...")

    try:
        # Manual trial loop with timeout checking
        for trial in range(n_trials):
            trial_start = time.time()
            print(f"\n  Trial {trial + 1}/{n_trials}:")

            # Run baseline episodes
            print("    Running baseline episodes...")
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
            print("    Baseline complete.")

            # Run test episodes
            print("    Running test episodes...")
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
        print("\n  Computing final score...")
        result = test.run(agent)
        print(f"  Score: {result.score:.4f}")
        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def result_from_dict(data: dict) -> TestResult:
    """Reconstruct TestResult from checkpoint dict."""
    return TestResult(
        test_name=data["test_name"],
        agent_name=data["agent_name"],
        score=data["score"],
        raw_data=data.get("raw_data", {}),
        metadata=data.get("metadata", {}),
    )


def get_gpu_temperature() -> float | None:
    """Get GPU temperature if available. Returns None if not available."""
    try:
        # Use nvidia-ml-py directly (pynvml is deprecated wrapper)
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTemperature, nvmlShutdown, NVML_TEMPERATURE_GPU
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        nvmlShutdown()
        return float(temp)
    except Exception:
        return None


def wait_for_gpu_cooldown(max_temp: float = 80.0, check_interval: float = 30.0) -> None:
    """Wait for GPU to cool down if temperature is too high."""
    temp = get_gpu_temperature()
    if temp is None:
        return  # No GPU monitoring available

    if temp > max_temp:
        print(f"\n  GPU temperature: {temp}C (above {max_temp}C threshold)")
        print("  Waiting for cooldown...")
        while temp and temp > max_temp - 5:  # Cool to 5 degrees below threshold
            time.sleep(check_interval)
            temp = get_gpu_temperature()
            if temp:
                print(f"    GPU temperature: {temp}C")
        print("  GPU cooled down, resuming...")


def update_progress_file(
    output_dir: Path,
    current_test: str,
    current_agent: str,
    tests_completed: int,
    total_tests: int,
    agents_completed: int,
    total_agents: int,
    start_time: float,
    completed_results: dict[str, dict[str, TestResult]],
) -> None:
    """Update progress file with current status for external monitoring."""
    progress_path = output_dir / "progress.json"

    elapsed = time.time() - start_time
    total_agent_tests = total_tests * total_agents
    completed_agent_tests = sum(len(agents) for agents in completed_results.values())

    # Estimate remaining time based on completed work
    if completed_agent_tests > 0:
        avg_time_per_test = elapsed / completed_agent_tests
        remaining_tests = total_agent_tests - completed_agent_tests
        estimated_remaining = avg_time_per_test * remaining_tests
    else:
        estimated_remaining = None

    progress_data = {
        "status": "running",
        "current_test": current_test,
        "current_agent": current_agent,
        "progress": {
            "tests_completed": tests_completed,
            "total_tests": total_tests,
            "agents_in_current_test": agents_completed,
            "total_agents": total_agents,
            "total_agent_tests_completed": completed_agent_tests,
            "total_agent_tests": total_agent_tests,
            "percent_complete": round(100 * completed_agent_tests / total_agent_tests, 1) if total_agent_tests > 0 else 0,
        },
        "timing": {
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_human": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s",
            "estimated_remaining_seconds": round(estimated_remaining, 1) if estimated_remaining else None,
            "estimated_remaining_human": f"{int(estimated_remaining // 3600)}h {int((estimated_remaining % 3600) // 60)}m" if estimated_remaining else "calculating...",
        },
        "gpu": {
            "temperature_c": get_gpu_temperature(),
        },
        "last_updated": datetime.now().isoformat(),
    }

    with open(progress_path, "w") as f:
        json.dump(progress_data, f, indent=2)


def finalize_progress_file(output_dir: Path, start_time: float) -> None:
    """Mark progress as complete."""
    progress_path = output_dir / "progress.json"
    elapsed = time.time() - start_time

    progress_data = {
        "status": "completed",
        "timing": {
            "total_seconds": round(elapsed, 1),
            "total_human": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s",
        },
        "completed_at": datetime.now().isoformat(),
    }

    with open(progress_path, "w") as f:
        json.dump(progress_data, f, indent=2)


def run_all_tests(
    agents: dict[str, BaseAgent],
    config: Config,
    output_dir: Path,
    test_filter: list[str] | None = None,
    verbose: bool = False,
    resume_checkpoint: dict[str, dict[str, dict]] | None = None,
    gpu_cooldown: bool = False,
    gpu_max_temp: float = 80.0,
    start_time: float | None = None,
) -> dict[str, dict[str, TestResult]]:
    """Run predicate tests on agents with checkpoint support."""
    all_results: dict[str, dict[str, TestResult]] = {}

    if start_time is None:
        start_time = time.time()

    all_tests = ["frustration", "means_end", "extinction", "anticipation", "opportunity_cost", "state_dependent"]
    tests_to_run = test_filter if test_filter else all_tests
    total_tests = len(tests_to_run)
    total_agents = len(agents)

    # Test name mappings (short name -> result key)
    test_name_map = {
        "frustration": "frustration_intensification",
        "means_end": "means_end_decoupling",
        "extinction": "extinction_resistance",
        "anticipation": "anticipatory_activation",
        "opportunity_cost": "opportunity_cost_sensitivity",
        "state_dependent": "state_dependent_valuation",
    }

    tests_completed = 0

    # Test 1: Frustration Intensification
    if "frustration" in tests_to_run:
        test_key = "frustration_intensification"
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
        # Restore from checkpoint if available
        if resume_checkpoint and test_key in resume_checkpoint:
            for agent_name, result_dict in resume_checkpoint[test_key].items():
                fi_results[agent_name] = result_from_dict(result_dict)
                print(f"  Restored {agent_name} from checkpoint (score: {result_dict['score']:.3f})")

        agents_completed = len(fi_results)
        for agent_name, agent in agents.items():
            if agent_name in fi_results:
                continue  # Skip already completed

            if gpu_cooldown:
                wait_for_gpu_cooldown(gpu_max_temp)

            # Update progress file
            update_progress_file(
                output_dir, test_key, agent_name,
                tests_completed, total_tests, agents_completed, total_agents,
                start_time, all_results,
            )

            print(f"\n  Testing {agent_name}...")
            sys.stdout.flush()

            start = time.time()
            result = fi_test.run(agent)
            elapsed = time.time() - start

            fi_results[agent_name] = result
            agents_completed += 1
            print(f"    FI Score: {result.score:.3f} ({elapsed:.1f}s)")

            # Save checkpoint after each agent
            all_results[test_key] = fi_results
            save_test_checkpoint(output_dir, all_results)
            print(f"    Checkpoint saved")

        all_results[test_key] = fi_results
        tests_completed += 1

    # Test 2: Means-End Decoupling
    if "means_end" in tests_to_run:
        test_key = "means_end_decoupling"
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
        # Restore from checkpoint if available
        if resume_checkpoint and test_key in resume_checkpoint:
            for agent_name, result_dict in resume_checkpoint[test_key].items():
                med_results[agent_name] = result_from_dict(result_dict)
                print(f"  Restored {agent_name} from checkpoint (score: {result_dict['score']:.3f})")

        agents_completed = len(med_results)
        for agent_name, agent in agents.items():
            if agent_name in med_results:
                continue  # Skip already completed

            if gpu_cooldown:
                wait_for_gpu_cooldown(gpu_max_temp)

            # Update progress file
            update_progress_file(
                output_dir, test_key, agent_name,
                tests_completed, total_tests, agents_completed, total_agents,
                start_time, all_results,
            )

            print(f"\n  Testing {agent_name}...")
            sys.stdout.flush()

            start = time.time()
            result = med_test.run(agent)
            elapsed = time.time() - start

            med_results[agent_name] = result
            agents_completed += 1
            print(f"    MED Score: {result.score:.3f} ({elapsed:.1f}s)")

            # Save checkpoint after each agent
            all_results[test_key] = med_results
            save_test_checkpoint(output_dir, all_results)
            print(f"    Checkpoint saved")

        all_results[test_key] = med_results
        tests_completed += 1

    # Test 3: Extinction Resistance
    if "extinction" in tests_to_run:
        test_key = "extinction_resistance"
        print("\n" + "="*50)
        print("Running Extinction Resistance Test")
        print("="*50)

        er_test = ExtinctionResistanceTest(
            n_trials=config.test.n_trials,
            baseline_episodes=config.test.baseline_episodes,
            test_episodes=config.test.test_episodes,
            grid_size=config.environment.grid_size,
            seed=config.training.seed,
            output_dir=str(output_dir / "extinction"),
        )

        er_results = {}
        # Restore from checkpoint if available
        if resume_checkpoint and test_key in resume_checkpoint:
            for agent_name, result_dict in resume_checkpoint[test_key].items():
                er_results[agent_name] = result_from_dict(result_dict)
                print(f"  Restored {agent_name} from checkpoint (score: {result_dict['score']:.3f})")

        agents_completed = len(er_results)
        for agent_name, agent in agents.items():
            if agent_name in er_results:
                continue  # Skip already completed

            if gpu_cooldown:
                wait_for_gpu_cooldown(gpu_max_temp)

            # Update progress file
            update_progress_file(
                output_dir, test_key, agent_name,
                tests_completed, total_tests, agents_completed, total_agents,
                start_time, all_results,
            )

            print(f"\n  Testing {agent_name}...")
            sys.stdout.flush()

            start = time.time()
            result = er_test.run(agent)
            elapsed = time.time() - start

            er_results[agent_name] = result
            agents_completed += 1
            print(f"    ER Score: {result.score:.3f} ({elapsed:.1f}s)")

            # Save checkpoint after each agent
            all_results[test_key] = er_results
            save_test_checkpoint(output_dir, all_results)
            print(f"    Checkpoint saved")

        all_results[test_key] = er_results
        tests_completed += 1

    # Test 4: Anticipatory Activation
    if "anticipation" in tests_to_run:
        test_key = "anticipatory_activation"
        print("\n" + "="*50)
        print("Running Anticipatory Activation Test")
        print("="*50)

        aa_test = AnticipatoryActivationTest(
            n_trials=config.test.n_trials,
            baseline_episodes=config.test.baseline_episodes,
            test_episodes=config.test.test_episodes,
            grid_size=config.environment.grid_size,
            seed=config.training.seed,
            output_dir=str(output_dir / "anticipation"),
        )

        aa_results = {}
        # Restore from checkpoint if available
        if resume_checkpoint and test_key in resume_checkpoint:
            for agent_name, result_dict in resume_checkpoint[test_key].items():
                aa_results[agent_name] = result_from_dict(result_dict)
                print(f"  Restored {agent_name} from checkpoint (score: {result_dict['score']:.3f})")

        agents_completed = len(aa_results)
        for agent_name, agent in agents.items():
            if agent_name in aa_results:
                continue  # Skip already completed

            if gpu_cooldown:
                wait_for_gpu_cooldown(gpu_max_temp)

            # Update progress file
            update_progress_file(
                output_dir, test_key, agent_name,
                tests_completed, total_tests, agents_completed, total_agents,
                start_time, all_results,
            )

            print(f"\n  Testing {agent_name}...")
            sys.stdout.flush()

            start = time.time()
            result = aa_test.run(agent)
            elapsed = time.time() - start

            aa_results[agent_name] = result
            agents_completed += 1
            print(f"    AA Score: {result.score:.3f} ({elapsed:.1f}s)")

            # Save checkpoint after each agent
            all_results[test_key] = aa_results
            save_test_checkpoint(output_dir, all_results)
            print(f"    Checkpoint saved")

        all_results[test_key] = aa_results
        tests_completed += 1

    # Test 5: Opportunity Cost Sensitivity
    if "opportunity_cost" in tests_to_run:
        test_key = "opportunity_cost_sensitivity"
        print("\n" + "="*50)
        print("Running Opportunity Cost Sensitivity Test")
        print("="*50)

        ocs_test = OpportunityCostSensitivityTest(
            n_trials=config.test.n_trials,
            baseline_episodes=config.test.baseline_episodes,
            test_episodes=config.test.test_episodes,
            grid_size=config.environment.grid_size,
            seed=config.training.seed,
            output_dir=str(output_dir / "opportunity_cost"),
        )

        ocs_results = {}
        # Restore from checkpoint if available
        if resume_checkpoint and test_key in resume_checkpoint:
            for agent_name, result_dict in resume_checkpoint[test_key].items():
                ocs_results[agent_name] = result_from_dict(result_dict)
                print(f"  Restored {agent_name} from checkpoint (score: {result_dict['score']:.3f})")

        agents_completed = len(ocs_results)
        for agent_name, agent in agents.items():
            if agent_name in ocs_results:
                continue  # Skip already completed

            if gpu_cooldown:
                wait_for_gpu_cooldown(gpu_max_temp)

            # Update progress file
            update_progress_file(
                output_dir, test_key, agent_name,
                tests_completed, total_tests, agents_completed, total_agents,
                start_time, all_results,
            )

            print(f"\n  Testing {agent_name}...")
            sys.stdout.flush()

            start = time.time()
            result = ocs_test.run(agent)
            elapsed = time.time() - start

            ocs_results[agent_name] = result
            agents_completed += 1
            print(f"    OCS Score: {result.score:.3f} ({elapsed:.1f}s)")

            # Save checkpoint after each agent
            all_results[test_key] = ocs_results
            save_test_checkpoint(output_dir, all_results)
            print(f"    Checkpoint saved")

        all_results[test_key] = ocs_results
        tests_completed += 1

    # Test 6: State-Dependent Valuation
    if "state_dependent" in tests_to_run:
        test_key = "state_dependent_valuation"
        print("\n" + "="*50)
        print("Running State-Dependent Valuation Test")
        print("="*50)

        sdv_test = StateDependentValuationTest(
            n_trials=config.test.n_trials,
            baseline_episodes=config.test.baseline_episodes,
            test_episodes=config.test.test_episodes,
            grid_size=config.environment.grid_size,
            seed=config.training.seed,
            output_dir=str(output_dir / "state_dependent"),
        )

        sdv_results = {}
        # Restore from checkpoint if available
        if resume_checkpoint and test_key in resume_checkpoint:
            for agent_name, result_dict in resume_checkpoint[test_key].items():
                sdv_results[agent_name] = result_from_dict(result_dict)
                print(f"  Restored {agent_name} from checkpoint (score: {result_dict['score']:.3f})")

        agents_completed = len(sdv_results)
        for agent_name, agent in agents.items():
            if agent_name in sdv_results:
                continue  # Skip already completed

            if gpu_cooldown:
                wait_for_gpu_cooldown(gpu_max_temp)

            # Update progress file
            update_progress_file(
                output_dir, test_key, agent_name,
                tests_completed, total_tests, agents_completed, total_agents,
                start_time, all_results,
            )

            print(f"\n  Testing {agent_name}...")
            sys.stdout.flush()

            start = time.time()
            result = sdv_test.run(agent)
            elapsed = time.time() - start

            sdv_results[agent_name] = result
            agents_completed += 1
            print(f"    SDV Score: {result.score:.3f} ({elapsed:.1f}s)")

            # Save checkpoint after each agent
            all_results[test_key] = sdv_results
            save_test_checkpoint(output_dir, all_results)
            print(f"    Checkpoint saved")

        all_results[test_key] = sdv_results
        tests_completed += 1

    # Finalize progress file
    finalize_progress_file(output_dir, start_time)

    return all_results


def analyze_results(
    results: dict[str, dict[str, TestResult]],
) -> dict[str, dict]:
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
        help="Specific agent(s) to test: heuristic, rl_ppo, rl_dqn, rl_a2c, active_inference",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists",
    )
    parser.add_argument(
        "--gpu-cooldown",
        action="store_true",
        help="Enable GPU temperature monitoring and cooldown (requires pynvml)",
    )
    parser.add_argument(
        "--gpu-max-temp",
        type=float,
        default=80.0,
        help="Maximum GPU temperature before cooldown (default: 80C)",
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
        print("Verbose mode: ON")
    if args.debug:
        print("Debug mode: ON (minimal episodes)")
    if args.resume:
        print("Resume mode: ON (will restore from checkpoint if exists)")
    if args.gpu_cooldown:
        temp = get_gpu_temperature()
        if temp is not None:
            print(f"GPU cooldown: ON (max {args.gpu_max_temp}C, current: {temp}C)")
        else:
            print("GPU cooldown: ON (pynvml not available, monitoring disabled)")

    # Load checkpoint if resuming
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = load_test_checkpoint(output_dir)
        if resume_checkpoint:
            completed_count = sum(len(agents) for agents in resume_checkpoint.values())
            print(f"Loaded checkpoint: {len(resume_checkpoint)} tests, {completed_count} agent results")
        else:
            print("No checkpoint found, starting fresh")

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
    test_start_time = time.time()
    results = run_all_tests(
        agents, config, output_dir,
        test_filter=args.test,
        verbose=args.verbose,
        resume_checkpoint=resume_checkpoint,
        gpu_cooldown=args.gpu_cooldown,
        gpu_max_temp=args.gpu_max_temp,
        start_time=test_start_time,
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
