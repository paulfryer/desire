"""Build the self-contained v2 HTML report (docs/report/v2_wanting_vs_habit.html) from outputs/v2.

No network, no external assets. Run after analyze_v2.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.stats import bootstrap_ci  # noqa: E402

OUT = REPO / "docs/report/v2_wanting_vs_habit.html"
V2 = REPO / "outputs/v2"

summary = json.loads((V2 / "summary.json").read_text())
S, EXT = summary["agents"], summary["extension_30M"]["agents"]

DESC = {
    "oracle": "ideal wanting agent (true item effects)",
    "need_novelty": "tries the unfamiliar item when needy",
    "nearest": "habit: nearest A or B item",
    "fixed_habit": "habit: always A, else B",
    "random": "random actions",
    "homeo": "learned item model × innate need",
    "aif": "active inference",
    "aif_no_novelty": "active inference, no novelty term",
    "mb_reward": "learned item model × learned reward",
    "ppo_free_intero": "model-free, feedforward, senses needs",
    "rppo_free_intero": "model-free, recurrent, senses needs",
    "ppo_clamped": "model-free, needs fixed in training",
    "ppo_free_blind": "model-free, cannot sense its needs",
}


def metric(e, key):
    return {"mean": e[key]["mean"], "lo": e[key]["ci"][0], "hi": e[key]["ci"][1],
            "seeds": [None if v != v else v for v in e[key]["seeds"]]}


def row(agent, e, family, budget=None):
    return {
        "id": agent, "family": family, "budget": budget,
        "desc": DESC[agent] + (f" · {budget}" if budget else ""),
        "DI": metric(e, "DI"), "RI": metric(e, "RI"), "RI_spec": metric(e, "RI_spec"),
        "choice": [e["choice_T1"]["mean"], e["choice_T2"]["mean"]],
        "drive": e["drive"]["mean"], "competent": e["competent"],
        "vT1": e["verdict_T1"], "vT2": e["verdict_T2"], "n": e["n_seeds"],
    }


groups = [
    ("Known-answer controls: the tests must sort these correctly", "control",
     [(a, S[a], None) for a in ("oracle", "need_novelty", "nearest", "fixed_habit", "random")]),
    ("Planners: value computed at decision time", "planner",
     [(a, S[a], None) for a in ("homeo", "aif", "aif_no_novelty", "mb_reward")]),
    ("Model-free after the 30M extension (Amendment 5)", "model-free",
     [(a, EXT[a], "30M steps") for a in ("rppo_free_intero", "ppo_free_intero")]),
    ("Model-free at the pre-registered 10M budget", "model-free",
     [(a, S[a], "10M steps") for a in ("rppo_free_intero", "ppo_free_intero", "ppo_clamped",
                                        "ppo_free_blind")]),
]
dot_rows = [{"group": title, "rows": [row(a, e, fam, b) for a, e, b in members]}
            for title, fam, members in groups]

# Need sweep (exploratory)
levels = None
sweep = []
for agent, folder, budget in [("oracle", "need_sweep", None), ("homeo", "need_sweep", None),
                              ("aif", "need_sweep", None), ("mb_reward", "need_sweep", None),
                              ("need_novelty", "need_sweep", None),
                              ("ppo_clamped", "need_sweep", "10M"),
                              ("ppo_free_intero", "need_sweep_30M", "30M"),
                              ("rppo_free_intero", "need_sweep_30M", "30M")]:
    files = sorted((V2 / folder).glob(f"{agent}_seed*.json"))
    rs = [json.loads(p.read_text()) for p in files]
    levels = rs[0]["levels"]
    entry = {"id": agent, "budget": budget, "desc": DESC[agent], "n": len(rs)}
    for t in ("A", "C"):
        seeds = np.array([r[t]["p_target"] for r in rs], dtype=float)
        entry[t] = [None if np.isnan(v) else float(v) for v in np.nanmean(seeds, axis=0)]
        entry[t + "_seeds"] = [[None if np.isnan(v) else float(v) for v in s] for s in seeds]
    sweep.append(entry)

# Training amount (T3) from checkpoints, with the 30M extension appended
t3 = []
for agent in ("ppo_free_intero", "rppo_free_intero", "ppo_clamped", "ppo_free_blind"):
    pts = []
    for folder, ext in (("model_free_assays", False), ("model_free_assays_30M", True)):
        rs = [json.loads(p.read_text()) for p in sorted((V2 / folder).glob(f"{agent}_seed_*.json"))]
        if not rs:
            continue
        for step in sorted({int(k) for r in rs for k in r["checkpoints"]}):
            vals = {k: [r["checkpoints"][str(step)][t][f] for r in rs if str(step) in r["checkpoints"]]
                    for k, t, f in (("DI", "T1", "DI"), ("RI", "T2", "RI"),
                                    ("choice", "T2", "choice_rate"))}
            p = {"step": step, "ext": ext}
            for k, v in vals.items():
                m, lo, hi = bootstrap_ci(v)
                p[k], p[k + "_lo"], p[k + "_hi"] = m, lo, hi
            pts.append(p)
    t3.append({"id": agent, "points": pts})

preds = {pid: (text, outcome) for pid, text, outcome in summary["predictions"]}
ext_preds = {pid: outcome for pid, _, outcome in summary["extension_30M"]["predictions"]}

data = {"dot": dot_rows, "levels": levels, "sweep": sweep, "t3": t3,
        "preds": preds, "ext_preds": ext_preds,
        "oracle_drive": S["oracle"]["drive"]["mean"],
        "hero": {"rppo": row("rppo_free_intero", EXT["rppo_free_intero"], "model-free", "30M"),
                 "oracle": row("oracle", S["oracle"], "control"),
                 "homeo": row("homeo", S["homeo"], "planner")}}

html = (REPO / "experiments/report_templates/v2.html").read_text()
html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
OUT.write_text(html)
print(OUT, f"{OUT.stat().st_size / 1024:.0f} KB")
