"""
Train one model-free agent (one seed) on NutrientGrid, saving checkpoints for the T3
training-amount analysis.

    python experiments/train_model_free.py --agent ppo_free_intero --seed 0 --steps 3000000
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.model_free import GridFeatures, NutrientVecEnv  # noqa: E402
from src.env import EnvConfig, drive  # noqa: E402

AGENTS = {
    # name: (algorithm, need_dynamics, interoception)
    "ppo_free_intero": ("ppo", "free", True),
    "ppo_free_blind": ("ppo", "free", False),
    "ppo_clamped": ("ppo", "clamped", True),
    "rppo_free_intero": ("rppo", "free", True),
}


def agent_config(name: str, reward: str = "reduction") -> EnvConfig:
    _, dynamics, intero = AGENTS[name]
    return EnvConfig(need_dynamics=dynamics, interoception=intero, reward=reward)


def checkpoint_schedule(total: int, n: int = 8) -> list[int]:
    """Roughly log-spaced checkpoint steps ending at total."""
    points = np.unique(np.geomspace(total / 2 ** (n - 1), total, n).astype(int))
    return [int(p) for p in points]


class ProgressCallback(BaseCallback):
    """Logs episode statistics and saves checkpoints at scheduled timesteps."""

    def __init__(self, out_dir: Path, checkpoints: list[int]):
        super().__init__()
        self.out_dir = out_dir
        self.pending = sorted(checkpoints)
        self.log_rows: list[dict] = []
        self._returns, self._eaten, self._drives = [], [], []
        self._start = time.time()

    def _on_step(self) -> bool:
        # Episode return telescopes to the change in drive, so mean drive is the progress metric.
        self._drives.append(float(drive(self.training_env.grid_env.h).mean()))
        for info in self.locals["infos"]:
            if "episode" in info:
                self._returns.append(info["episode"]["r"])
                self._eaten.append(info["episode"]["eaten"])
        if len(self._returns) >= 64:
            eaten = np.mean(self._eaten, axis=0)
            self._append_row({
                "timesteps": self.num_timesteps,
                "mean_drive": float(np.mean(self._drives)),
                "mean_return": float(np.mean(self._returns)),
                "eaten_A": float(eaten[0]), "eaten_B": float(eaten[1]), "eaten_C": float(eaten[2]),
                "seconds": round(time.time() - self._start, 1),
            })
            self._returns, self._eaten, self._drives = [], [], []
        while self.pending and self.num_timesteps >= self.pending[0]:
            step = self.pending.pop(0)
            self.model.save(self.out_dir / "checkpoints" / f"step_{step:09d}")
        return True

    def _append_row(self, row: dict) -> None:
        """Append to training_log.csv as training runs, so progress can be watched."""
        path = self.out_dir / "training_log.csv"
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not self.log_rows:
                writer.writeheader()
            writer.writerow(row)
        self.log_rows.append(row)


def build_model(name: str, env, seed: int):
    algo = AGENTS[name][0]
    policy_kwargs = {"features_extractor_class": GridFeatures,
                     "net_arch": {"pi": [256, 256], "vf": [256, 256]}}
    common = dict(learning_rate=3e-4, n_steps=128, n_epochs=4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.01, seed=seed, device="cpu", verbose=0)
    if algo == "ppo":
        return PPO("MultiInputPolicy", env, batch_size=1024, policy_kwargs=policy_kwargs, **common)
    policy_kwargs.update(lstm_hidden_size=128, n_lstm_layers=1, shared_lstm=False,
                         enable_critic_lstm=True)
    return RecurrentPPO("MultiInputLstmPolicy", env, batch_size=1024,
                        policy_kwargs=policy_kwargs, **common)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reward", choices=["reduction", "penalty"], default="reduction")
    parser.add_argument("--out", type=Path, default=Path("outputs/v2/model_free"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = args.out / args.agent / f"seed_{args.seed}"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "training_log.csv").unlink(missing_ok=True)

    env = NutrientVecEnv(args.n_envs, agent_config(args.agent, args.reward), seed=args.seed)
    model = build_model(args.agent, env, args.seed)
    callback = ProgressCallback(out_dir, checkpoint_schedule(args.steps))
    start = time.time()
    model.learn(total_timesteps=args.steps, callback=callback)
    model.save(out_dir / "final")
    (out_dir / "run.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "algorithm": AGENTS[args.agent][0],
        "env_config": agent_config(args.agent, args.reward).__dict__,
        "checkpoints": checkpoint_schedule(args.steps),
        "seconds": round(time.time() - start, 1),
    }, indent=2))
    print(f"{args.agent} seed {args.seed}: {args.steps} steps in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
