"""
Discriminant-validity check of the six v1 predicates (after bug fixes).

Runs each predicate on control agents that have no desire by construction. A predicate
that such an agent "passes" cannot be evidence of desire in any other agent.

Controls:
- random:       uniform random actions, constant internal state
- heuristic:    the v1 rule-based baseline (one-hot rule as internal state)
- sensor_echo:  heuristic policy whose "internal state" is just the raw resource channel
                of its current observation
- thermostat:   random walk whose movement probability is 1 - energy; consumes when on food

Usage:
    python experiments/legacy_validity_check.py --n-trials 5 --episodes 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.base_agent import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.tests.anticipation import AnticipatoryActivationTest
from src.tests.extinction import ExtinctionResistanceTest
from src.tests.frustration import FrustrationIntensificationTest
from src.tests.means_end import MeansEndDecouplingTest
from src.tests.opportunity_cost import OpportunityCostSensitivityTest
from src.tests.state_dependent import StateDependentValuationTest

CONSUME = 5
STAY = 4


class RandomAgent(BaseAgent):
    def act(self, observation):
        return int(self._rng.integers(0, self.action_space.n))

    def get_internal_state(self):
        return np.ones(1, dtype=np.float32)

    def get_internal_response(self, observation):
        return np.ones(1, dtype=np.float32)

    def reset(self):
        self.step_count = 0


class SensorEchoAgent(HeuristicAgent):
    """Heuristic policy; 'internal state' is a copy of the resource channel it currently sees."""

    def act(self, observation):
        self._last_view = observation["local_view"][:, :, 1].flatten().copy()
        return super().act(observation)

    def get_internal_state(self):
        return getattr(self, "_last_view", np.zeros(1, dtype=np.float32))


class ThermostatAgent(BaseAgent):
    """Moves randomly with probability (1 - energy); otherwise stays. Eats when standing on food."""

    def act(self, observation):
        view = observation["local_view"]
        center = view.shape[0] // 2
        if view[center, center, 1] > 0.5:
            return CONSUME
        energy = float(observation["energy"][0])
        if self._rng.random() < 1.0 - energy:
            return int(self._rng.integers(0, 4))
        return STAY

    def get_internal_state(self):
        return np.ones(1, dtype=np.float32)

    def get_internal_response(self, observation):
        return np.ones(1, dtype=np.float32)

    def reset(self):
        self.step_count = 0


def make_agents(seed: int) -> dict[str, BaseAgent]:
    env = ForagingGridEnv(grid_size=20)
    spaces = (env.observation_space, env.action_space)
    return {
        "random": RandomAgent(*spaces, name="random", seed=seed),
        "heuristic": HeuristicAgent(*spaces, name="heuristic", seed=seed),
        "sensor_echo": SensorEchoAgent(*spaces, name="sensor_echo", seed=seed),
        "thermostat": ThermostatAgent(*spaces, name="thermostat", seed=seed),
    }


# How v1 interpreted each predicate as "desire-like" (see docs/RESEARCH_DESIGN.md and the paper).
def desire_like(test_key: str, result) -> tuple[bool, float, str]:
    trial_data = result.raw_data["trial_data"]
    if test_key == "anticipation":
        r = float(np.mean([d.get("distance_activation_correlation", 0.0) for d in trial_data]))
        return r < -0.1, r, "distance-activation r < -0.1 ('true anticipation')"
    if test_key == "frustration":
        return result.score > 1.0, result.score, "FI > 1"
    if test_key == "means_end":
        return result.score > 0.5, result.score, "MED > 0.5 (efficient re-routing)"
    if test_key == "extinction":
        return result.score > 0.5, result.score, "ER > 0.5 (persistence)"
    if test_key == "opportunity_cost":
        return result.score > 0.5, result.score, "OCS > 0.5"
    if test_key == "state_dependent":
        return result.score > 0.2, result.score, "SDV > 0.2"
    raise ValueError(test_key)


def make_tests(n_trials: int, episodes: int, seed: int) -> dict:
    common = {"n_trials": n_trials, "seed": seed}
    return {
        "frustration": FrustrationIntensificationTest(
            baseline_episodes=episodes, test_episodes=episodes, **common),
        "means_end": MeansEndDecouplingTest(
            training_episodes=episodes, transfer_episodes=episodes, **common),
        "extinction": ExtinctionResistanceTest(
            baseline_episodes=episodes, test_episodes=episodes, **common),
        "anticipation": AnticipatoryActivationTest(
            baseline_episodes=episodes, test_episodes=episodes, **common),
        "opportunity_cost": OpportunityCostSensitivityTest(
            baseline_episodes=episodes, test_episodes=episodes, **common),
        "state_dependent": StateDependentValuationTest(
            baseline_episodes=episodes, test_episodes=episodes, **common),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/legacy_validity/results.json"))
    args = parser.parse_args()

    tests = make_tests(args.n_trials, args.episodes, args.seed)
    agents = make_agents(args.seed)
    results = {}

    for test_key, test in tests.items():
        results[test_key] = {}
        for agent_name, agent in agents.items():
            start = time.time()
            result = test.run(agent)
            passed, value, criterion = desire_like(test_key, result)
            results[test_key][agent_name] = {
                "score": result.score,
                "trial_scores": result.raw_data["trial_scores"],
                "statistic": value,
                "criterion": criterion,
                "desire_like": bool(passed),
            }
            print(f"{test_key:17s} {agent_name:12s} stat={value:+.3f} "
                  f"desire_like={passed!s:5s} ({time.time() - start:.0f}s)", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"config": vars(args) | {"output": str(args.output)}, "results": results}, indent=2))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
