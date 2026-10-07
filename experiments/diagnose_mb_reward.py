"""
Exploratory: how does mb_reward's learned reward model extrapolate to need states it never
experienced? Retrains each seed exactly as run_planners.py does and compares the model's
predicted reward for eating A (whose need varied in training) and C (whose need never fell
below setpoint) against the true drive-reduction reward.

    python experiments/diagnose_mb_reward.py --seeds 0-9
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_planners import parse_seeds, train  # noqa: E402
from src.agents.base import Option  # noqa: E402
from src.agents.planners import RewardModelPlanner  # noqa: E402
from src.env import BITE, REWARD_SCALE, SETPOINT, A, C, body_drift, drive  # noqa: E402

LEVELS = np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.9])


def true_reward(h: np.ndarray, k: int) -> np.ndarray:
    after = h.copy()
    after[:, k] = np.minimum(1.0, after[:, k] + BITE)
    return REWARD_SCALE * (drive(h) - drive(body_drift(after)))


def plan_values(agent: RewardModelPlanner, distance: int = 4) -> dict:
    """Decompose the planner's predicted return for each option in a deprivedC probe into the
    eating step and all other (no-eat) steps, next to the true summed reward."""
    h = np.array([0.3, 0.3, 0.05])
    options = [Option(C, distance, 0), Option(A, distance, 1), Option(None, 0, 4)]
    before, events, after = agent.simulate(options, h, agent.mean_effects())
    x = agent.features(before, events).reshape(-1, before.shape[-1] + 4)
    predicted = agent.model.predict(x).reshape(len(options), agent.horizon)
    true = REWARD_SCALE * (drive(before) - drive(after))
    out = {}
    for o, name in zip(range(len(options)), ("eat_C", "eat_A", "stay"), strict=True):
        eat = events[o] > 0
        out[name] = {"predicted_eat_step": float(predicted[o, eat].sum()),
                     "predicted_other_steps": float(predicted[o, ~eat].sum()),
                     "predicted_total": float(predicted[o].sum()),
                     "true_total": float(true[o].sum())}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--out", type=Path, default=Path("outputs/v2/mb_reward_extrapolation.json"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    results = {"levels": LEVELS.tolist(), "seeds": {}}
    for seed in parse_seeds(args.seeds):
        agent = RewardModelPlanner(seed=seed)
        train(agent, rounds=20, n_envs=10, seed=seed)
        agent.freeze()
        entry = {}
        for k, name in ((A, "eat_A"), (C, "eat_C")):
            h = np.tile([0.3, 0.3, SETPOINT], (len(LEVELS), 1))
            h[:, k] = LEVELS
            predicted = agent.model.predict(agent.features(h, np.full(len(LEVELS), k + 1)))
            entry[name] = {"predicted": predicted.tolist(), "true": true_reward(h, k).tolist()}
        entry["plan_values"] = plan_values(agent)
        results["seeds"][seed] = entry
        pv = entry["plan_values"]
        print(f"seed {seed}: eat C at h_C=0.05 predicted {entry['eat_C']['predicted'][0]:+.2f} "
              f"(true {entry['eat_C']['true'][0]:+.2f}) | plan totals pred/true: " + ", ".join(
                  f"{k} {v['predicted_total']:+.2f}/{v['true_total']:+.2f} "
                  f"(other steps {v['predicted_other_steps']:+.2f})" for k, v in pv.items()),
              flush=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
