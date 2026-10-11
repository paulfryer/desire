"""
Exploratory figures for the held-out-need experiment (not part of the frozen analysis; no verdicts).

    python experiments/plot_v3.py      # -> outputs/v3/fig_seeds.png, fig_training.png, fig_sweep.png
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.analyze_v2 import GRID, INK, INK_2, INK_3, SERIES, style_axis  # noqa: E402
from experiments.train_heldout import ARMS  # noqa: E402
from src.scoring_v3 import DELTA, t_interval  # noqa: E402

ROOT = Path("outputs/v3")
SURFACE = "#fcfcfb"


def load() -> tuple[dict[str, list[dict]], list[dict]]:
    runs = defaultdict(list)
    for path in sorted((ROOT / "assays").glob("*.json")):
        r = json.loads(path.read_text())
        runs[r["arm"]].append(r)
    oracle = [json.loads(p.read_text()) for p in sorted((ROOT / "validity").glob("oracle_seed*.json"))]
    return runs, oracle


def value(r: dict, key: str) -> float:
    return r["trained"][key] if key.endswith("_T") else r["heldout"][key]


def figure_seeds(runs, oracle, path: Path):
    rows = [("oracle (scripted)", oracle)] + [(arm, runs[arm]) for arm in ARMS]
    keys = [("RI_T", "RI_T: trained needs"), ("RI_H", "RI_H: held-out need"),
            ("SI_H", "SI_H: seeks H when deprived"), ("RI_novel", "RI_novel: unseen code (secondary)")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 0.46 * len(rows) + 1.5), sharey=True, facecolor=SURFACE)
    y = np.arange(len(rows))[::-1]
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes, keys, strict=True):
        style_axis(ax)
        ax.axvline(0, color=INK_3, linewidth=1)
        for x in (-DELTA, DELTA):
            ax.axvline(x, color=INK_3, linewidth=0.8, linestyle=(0, (3, 3)))
        for yi, (name, rs) in zip(y, rows, strict=True):
            color = SERIES[0] if name.startswith("oracle") else SERIES[2]
            seeds = np.array([value(r, key) for r in rs])
            mean, lo, hi = t_interval(seeds)
            ax.scatter(seeds, yi + rng.uniform(-0.17, 0.17, size=seeds.size), s=11, color=color,
                       alpha=0.4, linewidths=0)
            ax.plot([lo, hi], [yi, yi], color=color, linewidth=2, solid_capstyle="round")
            ax.scatter([mean], [yi], s=46, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.set_xlim(-1.08, 1.08)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_yticks(y, [f"{name} (n={len(rs)})" for name, rs in rows], fontsize=8, color=INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=SERIES[0], label="scripted control"),
               plt.Line2D([], [], marker="o", linestyle="", color=SERIES[2], label="learned (PPO)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8, labelcolor=INK_2)
    fig.text(0.01, 0.995, "Large dot = mean over seeds, bar = 95% t-interval, small dots = seeds. "
             f"Dashed lines = ±δ ({DELTA}).", fontsize=7.5, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def figure_training(runs, path: Path):
    lines = [("RI_T", "RI_T (trained needs)"), ("RI_H", "RI_H (held-out need)"),
             ("RI_novel", "RI_novel (unseen code)")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, arm in zip(axes.flat, ARMS, strict=True):
        style_axis(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.axhline(0, color=INK_3, linewidth=1)
        rs = runs[arm]
        steps = sorted({int(s) for r in rs for s in r["checkpoints"]})
        for color, (key, _) in zip(SERIES, lines, strict=False):
            stats = [t_interval([value(r["checkpoints"][str(s)], key) for r in rs]) for s in steps]
            mean, lo, hi = (np.array(v) for v in zip(*stats, strict=True))
            ax.fill_between(steps, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(steps, mean, color=color, linewidth=2, marker="o", markersize=3.5)
        ax.set_xscale("log")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{arm} (n={len(rs)})", fontsize=9, color=INK, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("training steps", fontsize=8, color=INK_2)
    handles = [plt.Line2D([], [], color=c, linewidth=2, label=label)
               for c, (_, label) in zip(SERIES, lines, strict=False)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK_2)
    fig.text(0.01, 0.995, "Mean over seeds at each checkpoint, band = 95% t-interval. Exploratory.",
             fontsize=7.5, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def figure_sweep(runs, oracle, path: Path):
    levels = sorted(oracle[0]["heldout"]["sweep"], key=float, reverse=True)
    x = [float(v) for v in levels]
    ideal = [np.mean([r["heldout"]["sweep"][v] for r in oracle]) for v in levels]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, arm in zip(axes.flat, ARMS, strict=True):
        style_axis(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.axhline(0.5, color=INK_3, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.plot(x, ideal, color=SERIES[0], linewidth=2, linestyle=(0, (4, 2)))
        for r in runs[arm]:
            ax.plot(x, [r["heldout"]["sweep"][v] for v in levels], color=SERIES[2], linewidth=0.8, alpha=0.3)
        ax.plot(x, [np.mean([r["heldout"]["sweep"][v] for r in runs[arm]]) for v in levels],
                color=SERIES[2], linewidth=2.2, marker="o", markersize=4)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{arm} (n={len(runs[arm])})", fontsize=9, color=INK, loc="left")
    axes[0, 0].invert_xaxis()   # shared: flips every panel
    for ax in axes[1]:
        ax.set_xlabel("H level (lower = more deprived); alternative j fixed at 0.3", fontsize=8, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(choose H)", fontsize=8, color=INK_2)
    handles = [plt.Line2D([], [], color=SERIES[0], linewidth=2, linestyle=(0, (4, 2)), label="oracle (scripted)"),
               plt.Line2D([], [], color=SERIES[2], linewidth=2.2, marker="o", markersize=4, label="arm mean"),
               plt.Line2D([], [], color=SERIES[2], linewidth=0.8, alpha=0.4, label="one seed")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK_2)
    fig.text(0.01, 0.995, "Need sweep (secondary, descriptive): P(choose H over j) as H is depleted.",
             fontsize=7.5, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    runs, oracle = load()
    figure_seeds(runs, oracle, ROOT / "fig_seeds.png")
    figure_training(runs, ROOT / "fig_training.png")
    figure_sweep(runs, oracle, ROOT / "fig_sweep.png")


if __name__ == "__main__":
    main()
