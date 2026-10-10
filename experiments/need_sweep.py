"""
Exploratory (not pre-registered): need-sweep curves.

For a target nutrient k, lower h_k from sated (0.5) to deprived (0.05) while the alternative
item's nutrient sits at a fixed moderate need (0.3), and record P(choose k | a choice was made).
Sweep A pairs A with B; A's need varied throughout training, so this traces in-distribution
need sensitivity. Sweep C pairs C with A or B, as in T2; C's need never fell below setpoint in
training, so this traces zero-shot revaluation. An ideal agent's curves both cross 0.5 where the
two needs are equal (h_k = 0.3).

    python experiments/need_sweep.py --seeds 0-9            # results per (agent, seed)
    python experiments/need_sweep.py --plot                 # figure from saved results
    python experiments/need_sweep.py --agents ppo_free_intero rppo_free_intero \
        --runs outputs/v2/model_free_30M --out outputs/v2/need_sweep_30M     # Amendment 5 models
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.analyze_v2 import INK, INK_2, INK_3, SERIES, style_axis  # noqa: E402
from experiments.run_model_free_assays import load_agent  # noqa: E402
from experiments.run_planners import CONTROLS, PLANNERS, parse_seeds, train  # noqa: E402
from experiments.train_model_free import AGENTS as MODEL_FREE  # noqa: E402
from src.assays import Condition, run_conditions  # noqa: E402
from src.env import SETPOINT, A, B, C  # noqa: E402

LEVELS = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
ALT_NEED = 0.3
SWEEP_AGENTS = ["oracle", "need_novelty", "homeo", "aif", "mb_reward",
                "ppo_free_intero", "ppo_free_blind", "ppo_clamped", "rppo_free_intero"]


def sweep_conditions(target: str) -> list[Condition]:
    if target == "A":
        return [Condition(f"A{lv}", (lv, ALT_NEED, SETPOINT), ((A, B),)) for lv in LEVELS]
    return [Condition(f"C{lv}", (ALT_NEED, ALT_NEED, lv), ((C, A), (C, B))) for lv in LEVELS]


def build_agent(name: str, seed: int, runs: Path):
    if name in CONTROLS:
        return CONTROLS[name](seed), True
    if name in PLANNERS:
        agent = PLANNERS[name](seed)
        train(agent, rounds=20, n_envs=10, seed=seed)
        agent.freeze()
        return agent, True
    final = runs / name / f"seed_{seed}" / "final.zip"
    if not final.exists():
        return None, None
    agent = load_agent(final, name)
    return agent, agent.interoception


def sweep(agent, seed: int, interoception: bool, probes: int) -> dict:
    out = {}
    for target in ("A", "C"):
        k = "ABC".index(target)
        cond = run_conditions(agent, sweep_conditions(target), probes, seed=60_000 + seed,
                              interoception=interoception)
        out[target] = {"p_target": [c["p_choose"]["ABC"[k]] for c in cond.values()],
                       "choice_rate": [c["choice_rate"] for c in cond.values()]}
    return out


def plot(root: Path, path: Path):
    agents = [a for a in SWEEP_AGENTS if list(root.glob(f"{a}_seed*.json"))]
    cols = 3
    rows = int(np.ceil(len(agents) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.6 * rows + 0.6), sharex=True,
                             sharey=True, facecolor="#fcfcfb")
    labels = {"A": "A: need varied in training", "C": "C: never deprived in training"}
    x = np.array(LEVELS)
    for ax in axes.flat[len(agents):]:
        ax.set_visible(False)
    for ax, agent in zip(axes.flat, agents, strict=False):
        style_axis(ax)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.axhline(0.5, color=INK_3, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.axvline(ALT_NEED, color=INK_3, linewidth=0.8)
        results = [json.loads(p.read_text()) for p in sorted(root.glob(f"{agent}_seed*.json"))]
        for color, target in zip(SERIES, ("A", "C"), strict=False):
            curves = np.array([r[target]["p_target"] for r in results], dtype=float)
            for curve in curves:
                ax.plot(x, curve, color=color, linewidth=0.8, alpha=0.25)
            ax.plot(x, np.nanmean(curves, axis=0), color=color, linewidth=2, marker="o",
                    markersize=4, label=labels[target])
        ax.set_title(f"{agent} (n={len(results)})", fontsize=9, color=INK, loc="left")
        ax.set_xlim(0.52, 0.03)                   # deprivation increases to the right
        ax.set_ylim(-0.04, 1.04)
    for ax in axes[-1]:
        ax.set_xlabel("level of the target nutrient (lower = more deprived)", fontsize=8,
                      color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(choose target item)", fontsize=8, color=INK_2)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize=8,
               labelcolor=INK_2)
    fig.text(0.01, 0.995, "Alternative item's nutrient fixed at 0.3 (vertical line: equal need). "
             "Thick line = mean over seeds, thin lines = seeds. Exploratory, not pre-registered.",
             fontsize=7.5, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", default=SWEEP_AGENTS)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--probes", type=int, default=100)
    parser.add_argument("--runs", type=Path, default=Path("outputs/v2/model_free"))
    parser.add_argument("--out", type=Path, default=Path("outputs/v2/need_sweep"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--figure", type=Path, default=Path("outputs/v2/figures/need_sweep.png"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot(args.out, args.figure)
        return
    import torch
    torch.set_num_threads(1)
    for name in args.agents:
        for seed in parse_seeds(args.seeds):
            path = args.out / f"{name}_seed{seed}.json"
            if path.exists() or (name not in CONTROLS | PLANNERS | MODEL_FREE):
                continue
            agent, intero = build_agent(name, seed, args.runs)
            if agent is None:
                continue
            result = {"agent": name, "seed": seed, "levels": list(LEVELS), "alt_need": ALT_NEED,
                      **sweep(agent, seed, intero, args.probes)}
            path.write_text(json.dumps(result, indent=1))
            print(f"{name:17s} seed {seed}: P(A) {np.round(result['A']['p_target'], 2)} "
                  f"P(C) {np.round(result['C']['p_target'], 2)}", flush=True)


if __name__ == "__main__":
    main()
