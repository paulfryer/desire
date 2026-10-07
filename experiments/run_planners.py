"""
Train the option planners and run the assays on them and on the ground-truth controls.

One JSON per (agent, seed) in outputs/v2/planners/. Planners learn for `--rounds` batches of
`--n-envs` episodes in the free-need, interoception-on environment, are frozen, then probed.
For online learners an exploratory incentive-learning control (T2b, Amendment 1) follows.

    python experiments/run_planners.py --seeds 0-9
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.controls import (  # noqa: E402
    FixedHabitAgent,
    NearestAgent,
    NeedNoveltyAgent,
    RandomAgent,
)
from src.agents.planners import (  # noqa: E402
    ActiveInferencePlanner,
    HomeostaticPlanner,
    OptionPlanner,
    RewardModelPlanner,
    make_oracle,
)
from src.assays import T2_CONDITIONS, assay_t1, assay_t2, competence  # noqa: E402
from src.env import C, EnvConfig, NutrientGrid, ProbeSpec  # noqa: E402

CONTROLS = {
    "random": RandomAgent,
    "nearest": NearestAgent,
    "fixed_habit": FixedHabitAgent,
    "need_novelty": NeedNoveltyAgent,
    "oracle": make_oracle,
}
PLANNERS = {
    "homeo": lambda seed: HomeostaticPlanner(seed=seed),
    "mb_reward": lambda seed: RewardModelPlanner(seed=seed),
    "aif": lambda seed: ActiveInferencePlanner(seed=seed),
    "aif_no_novelty": lambda seed: ActiveInferencePlanner(seed=seed, novelty=False),
}


def train(agent: OptionPlanner, rounds: int, n_envs: int, seed: int) -> list[dict]:
    env = NutrientGrid(n_envs, EnvConfig(), seed=seed)
    history = []
    for r in range(rounds):
        obs = env.reset()
        agent.reset(n_envs)
        done = np.zeros(n_envs, dtype=bool)
        drives = []
        while not done.all():
            h_before = obs["intero"].copy()
            obs, reward, done, eaten = env.step(agent.act(obs))
            agent.learn(h_before, eaten, obs["intero"], reward)
            drives.append(env.mean_drive())
        agent.end_of_round()
        history.append({"round": r, "mean_drive": float(np.mean(drives)),
                        "consumed": agent.n_consumed.tolist()})
    return history


def incentive_learning(agent: OptionPlanner, n_exposures: int, seed: int) -> None:
    """T2b: let the agent eat C a few times while deprived of C, with learning on."""
    condition = next(c for c in T2_CONDITIONS if c.name == "deprivedC")
    rng = np.random.default_rng(seed)
    agent.learning = True
    for _ in range(n_exposures):
        r, c = rng.integers(1, 8, size=2)
        spec = ProbeSpec(np.array([[r, c]]), np.array([[[r, c + 1], [0, 0]]]),
                         np.array([[C, C]]), np.array([condition.h]))
        env = NutrientGrid(1, EnvConfig(), seed=seed)
        obs = env.reset_probe(spec)
        h_before = obs["intero"].copy()
        obs, reward, _, eaten = env.step(np.array([3]))   # step right onto C
        assert eaten[0] == C
        agent.learn(h_before, eaten, obs["intero"], reward)
    agent.end_of_round()
    agent.freeze()


def run_one(name: str, seed: int, args) -> dict:
    start = time.time()
    result = {"agent": name, "seed": seed}
    if name in CONTROLS:
        agent = CONTROLS[name](seed)
    else:
        agent = PLANNERS[name](seed)
        result["training"] = train(agent, args.rounds, args.n_envs, seed)
        result["consumed_in_training"] = agent.n_consumed.tolist()
        result["learned_effects"] = agent.mean_effects().tolist()
        agent.freeze()
    result["T0"] = competence(agent, n_episodes=32, seed=10_000 + seed)
    result["T1"] = assay_t1(agent, args.probes, seed=20_000 + seed)
    result["T2"] = assay_t2(agent, args.probes, seed=30_000 + seed)
    if name in PLANNERS:
        incentive_learning(agent, n_exposures=3, seed=40_000 + seed)
        result["T2b"] = assay_t2(agent, args.probes, seed=50_000 + seed)
    result["seconds"] = round(time.time() - start, 1)
    return result


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", default=list(CONTROLS) + list(PLANNERS))
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--probes", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("outputs/v2/planners"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.agents:
        for seed in parse_seeds(args.seeds):
            path = args.out / f"{name}_seed{seed}.json"
            if path.exists():
                continue
            result = run_one(name, seed, args)
            path.write_text(json.dumps(result, indent=1))
            t1, t2 = result["T1"], result["T2"]
            print(f"{name:15s} seed {seed}: DI={t1['DI']:+.2f} RI={t2['RI']:+.2f} "
                  f"RI_spec={t2['RI_spec']:+.2f} drive={result['T0']['mean_drive']:.3f} "
                  f"({result['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
