"""
Pilot evaluation for the held-out-need experiment (docs/HELDOUT_DESIGN.md §9).

Reports competence and revaluation on TRAINED nutrients only (RI_T, SI_T) for pilot runs
(seeds >= 100). It deliberately does not import or compute any held-out-nutrient probe, so
pilots cannot reveal the primary outcome.

    python experiments/pilot_heldout.py --runs outputs/v3/pilot
"""

import argparse
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_heldout import world_of  # noqa: E402
from src.agents.model_free import SB3Agent  # noqa: E402
from src.assays_n import assay_trained, competence  # noqa: E402  (assay_heldout NOT imported)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("outputs/v3/pilot"))
    parser.add_argument("--probes", type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for run_dir in sorted(args.runs.glob("*/seed_*")):
        arm, seed = run_dir.parent.name, int(run_dir.name.split("_")[1])
        assert seed >= 100, "pilots use seeds >= 100 only"
        world = world_of(arm, seed)
        for ckpt in sorted((run_dir / "checkpoints").glob("step_*.zip")) + [run_dir / "final.zip"]:
            if not ckpt.exists():
                continue
            agent = SB3Agent(PPO.load(ckpt, device="cpu"), arm, interoception=True)
            tr = assay_trained(agent, world, args.probes, seed=20_000 + seed)
            comp = competence(agent, world, 32, seed=10_000 + seed)
            choice = min(c["choice_rate"] for c in tr["conditions"].values())
            print(f"{arm:17s} seed {seed} {ckpt.stem:>18s}: RI_T={tr['RI_T']:+.2f} SI_T={tr['SI_T']:+.2f} "
                  f"choice={choice:.2f} drive={comp['mean_drive']:.3f}", flush=True)


if __name__ == "__main__":
    main()
