"""
Main evaluation for the held-out-need experiment (docs/HELDOUT_DESIGN.md §5, §7, §9).

    python experiments/run_heldout.py validity     # scripted controls, seeds 0-9
    python experiments/run_heldout.py learned      # main runs; refuses until ALL have finished

Writes one JSON per (agent, seed) to outputs/v3/validity/ or outputs/v3/assays/: the final
model's assays and, for learned agents, the assays at each checkpoint (exploratory curves).
Every invocation is appended to outputs/v3/COMMANDS.log.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_heldout import ARMS, SEEDS, world_of  # noqa: E402
from src.agents.model_free import SB3Agent  # noqa: E402
from src.agents.multi import CONTROLS  # noqa: E402
from src.assays_n import assay_heldout, assay_trained, competence  # noqa: E402
from src.env_n import make_world  # noqa: E402

PROBES = 200
ROOT = Path("outputs/v3")


def evaluate(agent, world, seed: int, with_competence: bool = True) -> dict:
    torch.manual_seed(seed)
    out = {"trained": assay_trained(agent, world, PROBES, seed=20_000 + seed),
           "heldout": assay_heldout(agent, world, PROBES, seed=30_000 + seed)}
    if with_competence:
        out["competence"] = competence(agent, world, 32, seed=10_000 + seed)
    return out


def validity(seeds: range):
    out = ROOT / "validity"
    out.mkdir(parents=True, exist_ok=True)
    for name, cls in CONTROLS.items():
        for seed in seeds:
            path = out / f"{name}_seed{seed}.json"
            if path.exists():
                continue
            world = make_world(seed)
            result = {"agent": name, "seed": seed, **evaluate(cls(world, seed), world, seed)}
            path.write_text(json.dumps(result, indent=1))
            h = result["heldout"]
            print(f"{name:14s} seed {seed}: RI_T={result['trained']['RI_T']:+.2f} RI_H={h['RI_H']:+.2f} "
                  f"RI_spec_H={h['RI_spec_H']:+.2f} RI_dir_H={h['RI_dir_H']:+.2f} SI_H={h['SI_H']:+.2f}",
                  flush=True)


def load(path: Path, arm: str) -> SB3Agent:
    return SB3Agent(PPO.load(path, device="cpu"), arm, interoception=True)


def learned(runs: Path):
    missing = [f"{arm}/seed_{s}" for arm in ARMS for s in SEEDS[arm]
               if not (runs / arm / f"seed_{s}" / "final.zip").exists()]
    if missing:
        sys.exit(f"refusing: held-out assays run only after all main training has finished; "
                 f"{len(missing)} runs missing (e.g. {missing[0]})")
    torch.set_num_threads(1)
    out = ROOT / "assays"
    out.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for seed in SEEDS[arm]:
            run_dir = runs / arm / f"seed_{seed}"
            path = out / f"{arm}_seed{seed}.json"
            if path.exists():
                continue
            world = world_of(arm, seed)
            result = {"arm": arm, "seed": seed, **evaluate(load(run_dir / "final.zip", arm), world, seed),
                      "checkpoints": {}}
            for ckpt in sorted((run_dir / "checkpoints").glob("step_*.zip")):
                result["checkpoints"][int(ckpt.stem.split("_")[1])] = evaluate(
                    load(ckpt, arm), world, seed, with_competence=False)
            path.write_text(json.dumps(result, indent=1))
            h = result["heldout"]
            print(f"{arm:21s} seed {seed}: RI_T={result['trained']['RI_T']:+.2f} RI_H={h['RI_H']:+.2f} "
                  f"RI_spec_H={h['RI_spec_H']:+.2f} RI_dir_H={h['RI_dir_H']:+.2f} SI_H={h['SI_H']:+.2f}",
                  flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["validity", "learned"])
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "COMMANDS.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} run_heldout.py {' '.join(sys.argv[1:])}\n")
    if args.what == "validity":
        validity(range(10))
    else:
        learned(args.runs)


if __name__ == "__main__":
    main()
