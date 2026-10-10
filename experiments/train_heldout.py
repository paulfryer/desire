"""
Train one arm of the held-out-need experiment (docs/HELDOUT_DESIGN.md §4) for one seed.

    python experiments/train_heldout.py --arm innate-level-K6 --seed 0 --steps 30000000

PPO settings are the v2 Amendment 3 settings and are not tuned here.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.multi import EXTRACTORS, MultiVecEnv  # noqa: E402
from src.env import drive  # noqa: E402
from src.env_n import make_world  # noqa: E402

ARMS = {
    # name: (architecture, binding, read-out, depleting needs K, H constant in training)
    "innate-level-K6": ("attn", "innate", "level", 6, False),       # primary (best shot), 20 seeds
    "learned-level-K6": ("attn", "learned", "level", 6, False),     # the match must be learned
    "innate-code-K6": ("attn", "innate", "code", 6, False),         # value may be nutrient-specific
    "mlp-K6": ("mlp", None, None, 6, False),                        # generic learner
    "innate-level-K2": ("attn", "innate", "level", 2, False),       # fewer trained needs
    "innate-level-K6-constH": ("attn", "innate", "level", 6, True),  # H's full side never varies
}
PRIMARY = "innate-level-K6"
SEEDS = {arm: range(20) if arm == PRIMARY else range(10) for arm in ARMS}
PPO_SETTINGS = {"learning_rate": 3e-4, "n_steps": 256, "batch_size": 4096, "n_epochs": 4,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01}
N_ENVS = 64


def world_of(arm: str, seed: int):
    _, _, _, k, constant = ARMS[arm]
    return make_world(seed, n_depleting=k, held_out_constant=constant)


def checkpoint_schedule(total: int, n: int = 6) -> list[int]:
    return [int(p) for p in np.unique(np.geomspace(total / 2 ** (n - 1), total, n).astype(int))]


class Progress(BaseCallback):
    """Appends episode statistics to training_log.csv and saves scheduled checkpoints."""

    def __init__(self, out_dir: Path, checkpoints: list[int], held_out: int, n_depleting: int):
        super().__init__()
        self.out_dir, self.pending, self.held_out = out_dir, sorted(checkpoints), held_out
        self.n_depleting = n_depleting
        self._returns, self._eaten, self._drives, self._header = [], [], [], True
        self._start = time.time()

    def _on_step(self) -> bool:
        env = self.training_env
        while not hasattr(env, "grid_env"):
            env = env.venv
        self._drives.append(float(drive(env.grid_env.h).mean()))
        for info in self.locals["infos"]:
            if "episode" in info:
                self._returns.append(info["episode"]["r"])
                self._eaten.append(info["episode"]["eaten"])
        if len(self._returns) >= 64:
            eaten = np.mean(self._eaten, axis=0)
            trained = eaten[: self.n_depleting]
            row = {"timesteps": self.num_timesteps, "mean_drive": float(np.mean(self._drives)),
                   "mean_return": float(np.mean(self._returns)),
                   "eaten_per_trained": float(trained.mean()), "eaten_held_out": float(eaten[self.held_out]),
                   "seconds": round(time.time() - self._start, 1)}
            with open(self.out_dir / "training_log.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if self._header:
                    writer.writeheader()
                    self._header = False
                writer.writerow(row)
            self._returns, self._eaten, self._drives = [], [], []
        while self.pending and self.num_timesteps >= self.pending[0]:
            self.model.save(self.out_dir / "checkpoints" / f"step_{self.pending.pop(0):09d}")
        return True


def build_model(arm: str, env, seed: int, learning_rate: float | None = None) -> PPO:
    arch, binding, readout = ARMS[arm][:3]
    policy_kwargs = {"features_extractor_class": EXTRACTORS[arch],
                     "net_arch": {"pi": [256, 256], "vf": [256, 256]}}
    if arch == "attn":
        policy_kwargs["features_extractor_kwargs"] = {"readout": readout, "binding": binding}
    settings = {**PPO_SETTINGS, **({"learning_rate": learning_rate} if learning_rate else {})}
    return PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, seed=seed, device="cpu",
               verbose=0, **settings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, help="only via a recorded amendment (§9)")
    parser.add_argument("--out", type=Path, default=Path("outputs/v3/runs"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    out_dir = args.out / args.arm / f"seed_{args.seed}"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "training_log.csv").unlink(missing_ok=True)
    world = world_of(args.arm, args.seed)
    env = VecNormalize(MultiVecEnv(N_ENVS, world, seed=args.seed), norm_obs=False,
                       norm_reward=True, gamma=PPO_SETTINGS["gamma"])
    model = build_model(args.arm, env, args.seed, args.learning_rate)
    checkpoints = checkpoint_schedule(args.steps)
    start = time.time()
    model.learn(total_timesteps=args.steps,
                callback=Progress(out_dir, checkpoints, world.held_out, world.n_depleting))
    model.save(out_dir / "final")
    (out_dir / "run.json").write_text(json.dumps({
        "arm": args.arm, "seed": args.seed, "steps": args.steps, "architecture": ARMS[args.arm][0],
        "binding": ARMS[args.arm][1], "readout": ARMS[args.arm][2], "n_nutrients": world.n,
        "n_depleting": world.n_depleting,
        "held_out_constant": world.held_out_constant, "held_out": world.held_out,
        "ppo": {**PPO_SETTINGS, **({"learning_rate": args.learning_rate} if args.learning_rate else {})},
        "n_envs": N_ENVS, "checkpoints": checkpoints, "seconds": round(time.time() - start, 1),
    }, indent=2))
    print(f"{args.arm} seed {args.seed}: {args.steps} steps in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
