"""
Aggregate v2 assay results over seeds, score the pre-registered predictions, and draw figures.

Reads outputs/v2/planners/*.json and outputs/v2/model_free_assays/*.json; writes
outputs/v2/summary.json, outputs/v2/results_tables.md and outputs/v2/figures/*.png.

    python experiments/analyze_v2.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stats import bootstrap_ci, holm, permutation_test, verdict  # noqa: E402

CONTROLS = ["random", "nearest", "fixed_habit", "need_novelty", "oracle"]
PLANNERS = ["homeo", "mb_reward", "aif", "aif_no_novelty"]
MODEL_FREE = ["ppo_free_intero", "ppo_free_blind", "ppo_clamped", "rppo_free_intero"]
FAMILY = {**dict.fromkeys(CONTROLS, "control"), **dict.fromkeys(PLANNERS, "planner"),
          **dict.fromkeys(MODEL_FREE, "model-free")}
# Reference palette (dataviz skill), categorical slots in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
FAMILY_COLOR = {"control": SERIES[0], "planner": SERIES[1], "model-free": SERIES[2]}
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COMPETENT = 0.9


def load(out_root: Path) -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for path in sorted((out_root / "planners").glob("*.json")) + \
            sorted((out_root / "model_free_assays").glob("*.json")):
        result = json.loads(path.read_text())
        runs[result["agent"]].append(result)
    return runs


def seed_scores(results: list[dict]) -> dict[str, np.ndarray]:
    get = lambda f: np.array([f(r) for r in results], dtype=float)  # noqa: E731
    scores = {
        "DI": get(lambda r: r["T1"]["DI"]),
        "RI": get(lambda r: r["T2"]["RI"]),
        "RI_spec": get(lambda r: r["T2"]["RI_spec"]),
        "choice_T1": get(lambda r: r["T1"]["choice_rate"]),
        "choice_T2": get(lambda r: r["T2"]["choice_rate"]),
        "drive": get(lambda r: r["T0"]["mean_drive"]),
    }
    if "consumed_in_training" in results[0]:
        scores["C_exposures"] = get(lambda r: r["consumed_in_training"][2])
    if "T2b" in results[0]:
        scores["RI_after_incentive"] = get(lambda r: r["T2b"]["RI"])
        scores["RI_spec_after_incentive"] = get(lambda r: r["T2b"]["RI_spec"])
    return scores


def summarize(runs) -> dict:
    summary = {}
    for agent, results in runs.items():
        scores = seed_scores(results)
        entry = {"n_seeds": len(results), "family": FAMILY.get(agent, "?")}
        for key, values in scores.items():
            mean, lo, hi = bootstrap_ci(values)
            entry[key] = {"mean": mean, "ci": [lo, hi], "seeds": values.tolist()}
        competent = min(entry["choice_T1"]["mean"], entry["choice_T2"]["mean"]) >= COMPETENT
        t2_pass = (verdict(scores["RI"]) == "pass" and entry["RI_spec"]["mean"] >= 0.5
                   and entry["RI_spec"]["ci"][0] > 0)
        entry["competent"] = bool(competent)
        ok_seeds = np.minimum(scores["choice_T1"], scores["choice_T2"]) >= COMPETENT
        entry["n_competent_seeds"] = int(ok_seeds.sum())
        if FAMILY.get(agent) == "model-free" and ok_seeds.sum() >= 5:
            # Exploratory sensitivity analysis (Amendment 4): competent seeds only.
            entry["competent_seeds_only"] = {
                key: dict(zip(("mean", "lo", "hi"), bootstrap_ci(scores[key][ok_seeds]), strict=True))
                for key in ("DI", "RI", "RI_spec")}
        entry["verdict_T1"] = verdict(scores["DI"])
        entry["verdict_RI"] = verdict(scores["RI"])
        if t2_pass:
            entry["verdict_T2"] = "pass"
        elif verdict(scores["RI"]) == "pass":
            entry["verdict_T2"] = "fail (guard)"   # RI passes but RI_spec does not (Amendment 2)
        else:
            entry["verdict_T2"] = verdict(scores["RI"])
        summary[agent] = entry
    return summary


def comparisons(runs) -> dict:
    """Permutation tests for the comparisons named in §6, Holm-corrected."""
    scores = {a: seed_scores(r) for a, r in runs.items()}
    plan = [
        ("aif vs homeo: DI", "aif", "homeo", "DI"),
        ("aif vs homeo: RI", "aif", "homeo", "RI"),
        ("aif vs aif_no_novelty: C exposures", "aif", "aif_no_novelty", "C_exposures"),
        ("aif vs aif_no_novelty: RI", "aif", "aif_no_novelty", "RI"),
        ("ppo_free_intero vs ppo_clamped: DI", "ppo_free_intero", "ppo_clamped", "DI"),
        ("ppo_free_intero vs ppo_free_blind: DI", "ppo_free_intero", "ppo_free_blind", "DI"),
        ("ppo_free_intero vs homeo: RI", "ppo_free_intero", "homeo", "RI"),
        ("rppo_free_intero vs ppo_free_intero: RI", "rppo_free_intero", "ppo_free_intero", "RI"),
    ]
    raw, diffs = {}, {}
    for label, a, b, key in plan:
        if a in scores and b in scores and key in scores[a] and key in scores[b]:
            raw[label] = permutation_test(scores[a][key], scores[b][key])
            diffs[label] = float(np.nanmean(scores[a][key]) - np.nanmean(scores[b][key]))
    adjusted = holm(raw) if raw else {}
    return {label: {"diff": diffs[label], "p": raw[label], "p_holm": adjusted[label]}
            for label in raw}


def scorecard(s: dict, comps: dict) -> list[tuple[str, str, str]]:
    """Evaluate P1-P9 exactly as written in the design doc (with Amendment 2)."""
    def has(*agents):
        return all(a in s for a in agents)

    def m(agent, key):
        return s[agent][key]["mean"]

    rows = []
    if has("oracle", "random", "nearest", "fixed_habit", "need_novelty"):
        ok = (s["oracle"]["verdict_T1"] == "pass" and s["oracle"]["verdict_T2"] == "pass"
              and all(s[a]["verdict_T1"] != "pass" and s[a]["verdict_T2"] != "pass"
                      for a in ("random", "nearest", "fixed_habit", "need_novelty"))
              and s["need_novelty"]["verdict_RI"] == "pass")
        rows.append(("P1", "oracle passes T1/T2; no-desire controls fail; need_novelty passes RI "
                     "but fails the guard", "supported" if ok else "not supported"))
    def outcome(ok, *agents):
        # Assays are interpreted only for competent agents (§4 T0, Amendment 4).
        weak = [a for a in agents if not s[a]["competent"]]
        if weak:
            return f"not interpretable ({', '.join(weak)} below competence criterion)"
        return "supported" if ok else "not supported"

    if has("ppo_free_blind", "ppo_clamped"):
        ok = all(s[a]["verdict_T1"] == "fail" for a in ("ppo_free_blind", "ppo_clamped"))
        rows.append(("P2", "ppo_free_blind and ppo_clamped fail T1",
                     f"{outcome(ok, 'ppo_free_blind', 'ppo_clamped')} "
                     f"(T1 {s['ppo_free_blind']['verdict_T1']}, "
                     f"{s['ppo_clamped']['verdict_T1']})"))
    for pid, agent in (("P3", "ppo_free_intero"), ("P4", "rppo_free_intero")):
        if has(agent):
            ok = s[agent]["verdict_T1"] == "pass" and s[agent]["verdict_T2"].startswith("fail")
            rows.append((pid, f"{agent} passes T1 but fails T2",
                         f"{outcome(ok, agent)} "
                         f"(T1 {s[agent]['verdict_T1']}, T2 {s[agent]['verdict_T2']})"))
    if has("homeo", "aif", "aif_no_novelty"):
        ok = all(s[a]["verdict_T1"] == "pass" and s[a]["verdict_T2"] == "pass"
                 for a in ("homeo", "aif", "aif_no_novelty"))
        rows.append(("P5", "homeo, aif, aif_no_novelty pass T1 and T2",
                     "supported" if ok else "not supported"))
    if has("mb_reward"):
        ok = s["mb_reward"]["verdict_T1"] == "pass" and s["mb_reward"]["verdict_T2"] != "pass"  # partial or fail
        rows.append(("P6", "mb_reward passes T1; T2 partial or fail",
                     f"{'supported' if ok else 'not supported'} "
                     f"(T2 {s['mb_reward']['verdict_T2']}, RI {m('mb_reward', 'RI'):+.2f})"))
    if has("aif", "homeo"):
        d_ri = abs(m("aif", "RI") - m("homeo", "RI"))
        d_di = abs(m("aif", "DI") - m("homeo", "DI"))
        rows.append(("P7", "|RI_aif - RI_homeo| < 0.15 and |DI_aif - DI_homeo| < 0.15",
                     f"{'supported' if d_ri < 0.15 and d_di < 0.15 else 'not supported'} "
                     f"(ΔRI {d_ri:.2f}, ΔDI {d_di:.2f})"))
    if has("aif", "aif_no_novelty"):
        # "Increases" is tested with the planned permutation test (Holm-corrected), per §7.
        test = comps.get("aif vs aif_no_novelty: C exposures", {"diff": 0.0, "p_holm": 1.0})
        more_c = test["diff"] > 0 and test["p_holm"] < 0.05
        same_t2 = abs(m("aif", "RI") - m("aif_no_novelty", "RI")) < 0.15
        rows.append(("P8", "novelty increases C exposures in training but not T2",
                     f"{'supported' if more_c and same_t2 else 'not supported'} "
                     f"(C exposures {m('aif', 'C_exposures'):.0f} vs "
                     f"{m('aif_no_novelty', 'C_exposures'):.0f}, Holm p = {test['p_holm']:.2f})"))
    rows.append(("P9", "T3 exploratory (no directional prediction)", "see T3 figure"))
    return rows


def fmt(entry, key):
    if key not in entry:
        return "—"
    mean, (lo, hi) = entry[key]["mean"], entry[key]["ci"]
    return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def tables_markdown(s: dict, comps: dict, card) -> str:
    order = [a for a in CONTROLS + PLANNERS + MODEL_FREE if a in s]
    lines = ["## Assay results (mean over seeds, 95% bootstrap CI)", "",
             "| Agent | Family | Seeds | Choice rate T1/T2 | Mean drive | DI (T1) | RI (T2) | RI_spec | "
             "T1 | T2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a in order:
        e = s[a]
        choice = f"{e['choice_T1']['mean']:.2f} / {e['choice_T2']['mean']:.2f}"
        flag = "" if e["competent"] else " ⚠"
        lines.append(f"| `{a}` | {e['family']} | {e['n_seeds']} | {choice}{flag} | "
                     f"{e['drive']['mean']:.3f} | {fmt(e, 'DI')} | {fmt(e, 'RI')} | "
                     f"{fmt(e, 'RI_spec')} | {e['verdict_T1']} | {e['verdict_T2']} |")
    lines += ["", "⚠ = below the competence criterion (choice rate < 0.9); assay verdicts not "
              "interpretable."]
    sens = [a for a in MODEL_FREE if a in s]
    if sens:
        lines += ["", "## Model-free competence by seed (exploratory sensitivity, Amendment 4)", "",
                  "| Agent | Competent seeds | DI | RI | RI_spec (competent seeds only) |",
                  "|---|---|---|---|---|"]
        for a in sens:
            sub = s[a].get("competent_seeds_only")
            cells = (["—"] * 3 if sub is None else
                     [f"{sub[k]['mean']:+.2f} [{sub[k]['lo']:+.2f}, {sub[k]['hi']:+.2f}]"
                      for k in ("DI", "RI", "RI_spec")])
            lines.append(f"| `{a}` | {s[a]['n_competent_seeds']}/{s[a]['n_seeds']} | "
                         + " | ".join(cells) + " |")
    lines += ["", "## Pre-registered predictions", "",
              "| ID | Prediction | Outcome |", "|---|---|---|"]
    lines += [f"| {pid} | {text} | {outcome} |" for pid, text, outcome in card]
    if comps:
        lines += ["", "## Planned comparisons (permutation test on seed scores, Holm-corrected)",
                  "", "| Comparison | Difference | p | p (Holm) |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['diff']:+.3f} | {v['p']:.4f} | {v['p_holm']:.4f} |"
                  for k, v in comps.items()]
    t2b = [a for a in PLANNERS if a in s and "RI_after_incentive" in s[a]]
    if t2b:
        lines += ["", "## T2b incentive-learning control (exploratory)", "",
                  "| Agent | RI before | RI after 3 deprived C exposures | RI_spec after |",
                  "|---|---|---|---|"]
        lines += [f"| `{a}` | {fmt(s[a], 'RI')} | {fmt(s[a], 'RI_after_incentive')} | "
                  f"{fmt(s[a], 'RI_spec_after_incentive')} |" for a in t2b]
    return "\n".join(lines) + "\n"


def style_axis(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_indices(s: dict, path: Path):
    order = [a for a in CONTROLS + PLANNERS + MODEL_FREE if a in s]
    keys = [("DI", "T1 devaluation index (DI)"), ("RI", "T2 revaluation index (RI)"),
            ("RI_spec", "T2 specificity guard (RI_spec)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.42 * len(order) + 1.4), sharey=True,
                             facecolor="#fcfcfb")
    y = np.arange(len(order))[::-1]
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes, keys, strict=True):
        style_axis(ax)
        ax.axvline(0, color=INK_3, linewidth=1)
        ax.axvline(0.5, color=INK_3, linewidth=0.8, linestyle=(0, (3, 3)))
        for yi, agent in zip(y, order, strict=True):
            e = s[agent]
            color = FAMILY_COLOR[e["family"]]
            seeds = np.array(e[key]["seeds"], dtype=float)
            ax.scatter(seeds, yi + rng.uniform(-0.15, 0.15, size=seeds.size), s=10,
                       color=color, alpha=0.35, linewidths=0)
            lo, hi = e[key]["ci"]
            ax.plot([lo, hi], [yi, yi], color=color, linewidth=2, solid_capstyle="round")
            ax.scatter([e[key]["mean"]], [yi], s=46, color=color, edgecolors="#fcfcfb",
                       linewidths=1.5, zorder=3)
        ax.set_xlim(-1.08, 1.08)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_yticks(y, [a + ("" if s[a]["competent"] else " ⚠") for a in order],
                       fontsize=8, color=INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=f, markersize=6)
               for f, c in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               labelcolor=INK_2)
    fig.text(0.01, 0.995, "Large dot = mean over seeds, bar = 95% bootstrap CI, small dots = "
             "seeds. Dashed line = pass threshold 0.5. ⚠ = below competence criterion.",
             fontsize=7.5, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def figure_training_amount(runs, path: Path):
    agents = [a for a in MODEL_FREE if a in runs and runs[a][0].get("checkpoints")]
    if not agents:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor="#fcfcfb")
    for ax, (key, test, title) in zip(axes, [("DI", "T1", "T1 devaluation index (DI)"),
                                             ("RI", "T2", "T2 revaluation index (RI)")],
                                      strict=True):
        style_axis(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.axhline(0, color=INK_3, linewidth=1)
        for color, agent in zip(SERIES, agents, strict=False):
            steps = sorted({int(k) for r in runs[agent] for k in r["checkpoints"]})
            means, los, his = [], [], []
            for step in steps:
                vals = [r["checkpoints"][str(step)][test][key] for r in runs[agent]
                        if str(step) in r["checkpoints"]]
                mean, lo, hi = bootstrap_ci(vals)
                means.append(mean), los.append(lo), his.append(hi)
            ax.fill_between(steps, los, his, color=color, alpha=0.12, linewidth=0)
            ax.plot(steps, means, color=color, linewidth=2, marker="o", markersize=4)
            ax.annotate(agent, (steps[-1], means[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7.5, color=INK_2, va="center")
        ax.set_xscale("log")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("training steps", fontsize=8, color=INK_2)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    handles = [plt.Line2D([], [], color=c, linewidth=2, label=a)
               for c, a in zip(SERIES, agents, strict=False)]
    fig.legend(handles=handles, loc="lower center", ncol=len(agents), frameon=False, fontsize=8,
               labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.08, 0.93, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/v2"))
    args = parser.parse_args()
    runs = load(args.root)
    s = summarize(runs)
    comps = comparisons(runs)
    card = scorecard(s, comps)
    (args.root / "figures").mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(
        {"agents": s, "comparisons": comps, "predictions": card}, indent=1))
    md = tables_markdown(s, comps, card)
    (args.root / "results_tables.md").write_text(md)
    figure_indices(s, args.root / "figures" / "assay_indices.png")
    figure_training_amount(runs, args.root / "figures" / "training_amount.png")
    print(md)


if __name__ == "__main__":
    main()
