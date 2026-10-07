"""
Run the assays on trained model-free agents: the final model (T0, T1, T2) and every saved
checkpoint (T3, training amount).

    python experiments/run_model_free_assays.py --runs outputs/v2/model_free
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_model_free import AGENTS, agent_config  # noqa: E402
from src.agents.model_free import SB3Agent  # noqa: E402
from src.assays import assay_t1, assay_t2, competence  # noqa: E402


def load_agent(path: Path, name: str) -> SB3Agent:
    algo = AGENTS[name][0]
    cls = PPO if algo == "ppo" else RecurrentPPO
    model = cls.load(path, device="cpu")
    return SB3Agent(model, name, interoception=agent_config(name).interoception,
                    recurrent=algo == "rppo")


def evaluate(agent: SB3Agent, seed: int, probes: int, with_competence: bool) -> dict:
    intero = agent.interoception
    out = {
        "T1": assay_t1(agent, probes, seed=20_000 + seed, interoception=intero),
        "T2": assay_t2(agent, probes, seed=30_000 + seed, interoception=intero),
    }
    if with_competence:
        out["T0"] = competence(agent, n_episodes=32, seed=10_000 + seed,
                               config=agent_config(agent.name))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("outputs/v2/model_free"))
    parser.add_argument("--out", type=Path, default=Path("outputs/v2/model_free_assays"))
    parser.add_argument("--agents", nargs="+", default=sorted(AGENTS))
    parser.add_argument("--probes", type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)

    for name in args.agents:
        for run_dir in sorted((args.runs / name).glob("seed_*")):
            final = run_dir / "final.zip"
            path = args.out / f"{name}_{run_dir.name}.json"
            if not final.exists() or path.exists():
                continue
            start = time.time()
            seed = int(run_dir.name.split("_")[1])
            result = {"agent": name, "seed": seed,
                      **evaluate(load_agent(final, name), seed, args.probes, True),
                      "checkpoints": {}}
            for ckpt in sorted((run_dir / "checkpoints").glob("step_*.zip")):
                step = int(ckpt.stem.split("_")[1])
                result["checkpoints"][step] = evaluate(load_agent(ckpt, name), seed,
                                                       args.probes, False)
            result["seconds"] = round(time.time() - start, 1)
            path.write_text(json.dumps(result, indent=1))
            print(f"{name:17s} {run_dir.name}: DI={result['T1']['DI']:+.2f} "
                  f"RI={result['T2']['RI']:+.2f} RI_spec={result['T2']['RI_spec']:+.2f} "
                  f"choice={result['T1']['choice_rate']:.2f}/{result['T2']['choice_rate']:.2f} "
                  f"drive={result['T0']['mean_drive']:.3f} ({result['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
