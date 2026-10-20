"""Build the self-contained v3 HTML report (docs/report/v3_heldout_need.html) from outputs/v3.

No network, no external assets. Run after analyze_v3.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from experiments.analyze_v3 import PRIMARY_PRIOR, summarize_arm  # noqa: E402
from experiments.train_heldout import ARMS  # noqa: E402
from src.scoring_v3 import seed_transfers, t_interval  # noqa: E402

OUT = REPO / "docs/report/v3_heldout_need.html"
V3 = REPO / "outputs/v3"

summary = json.loads((V3 / "summary.json").read_text())

LABEL = {
    "oracle": ("Ideal agent", "scripted: true item effects + need function"),
    "innate-level-K6": ("Best shot", "built-in item↔need match · value from need level only · 6 trained needs"),
    "learned-level-K6": ("Must learn the match", "item↔need match learned from scratch"),
    "innate-code-K6": ("Value can depend on item", "built-in match, but value may also use item identity"),
    "mlp-K6": ("Generic network", "plain MLP, no built-in structure"),
    "innate-level-K2": ("Only 2 trained needs", "best shot, but only 2 nutrients ever ran low"),
    "innate-level-K6-constH": ("H never varied", "best shot, but H's level was constant in training"),
}
VALID_DESC = {
    "oracle": "true model + need function",
    "nearest": "habit: goes to the nearest item",
    "random": "random actions",
    "need_novelty": "picks H whenever any need is high",
    "veto": "only avoids items it is full of",
    "breakdown": "ideal, but random when H looks unfamiliar",
    "unusual_seeker": "picks H whenever H's level is unusual",
}


def m(metric):
    return {"mean": metric["mean"], "lo": metric["lo"], "hi": metric["hi"],
            "seeds": [float(v) for v in metric["seeds"]]}


def arm_row(arm_id, s, kind):
    mm = s["metrics"]
    return {
        "id": arm_id, "kind": kind, "name": LABEL[arm_id][0], "desc": LABEL[arm_id][1],
        "n": s["n_seeds"], "verdict": s["heldout"]["verdict"], "failed": s["heldout"]["failed_checks"],
        "trained": s["trained"], "gap": s["gap"], "count": s["seed_transfer_count"],
        **{k: m(mm[k]) for k in ("RI_T", "SI_T", "RI_H", "SI_H", "RI_novel", "SI_novel", "gap")},
        "drive": mm["drive"]["mean"], "choice": [mm["choice_dep_H"]["mean"], mm["choice_dep_T"]["mean"]],
    }


oracle_runs = [json.loads(p.read_text()) for p in sorted((V3 / "validity").glob("oracle_seed*.json"))]
oracle = arm_row("oracle", summarize_arm(oracle_runs), "control")
arms = [oracle] + [arm_row(a, summary["arms"][a], "learned") for a in ARMS]

primary_runs = sorted((json.loads(p.read_text()) for p in (V3 / "assays").glob("innate-level-K6_seed*.json")),
                      key=lambda r: r["seed"])
seeds = []
for r in primary_runs:
    h = r["heldout"]
    seeds.append({"seed": r["seed"], "RI_H": h["RI_H"], "SI_H": h["SI_H"], "RI_T": r["trained"]["RI_T"],
                  "RI_novel": h["RI_novel"], "pdep": h["conditions"]["dep_H"]["p_target"],
                  "transfers": bool(seed_transfers(h["RI_H"], h["RI_spec_H"], h["RI_dir_H"], h["SI_H"]))})

levels = sorted(primary_runs[0]["heldout"]["sweep"], key=float, reverse=True)
sweep = {"levels": [float(v) for v in levels],
         "primary": [float(np.mean([r["heldout"]["sweep"][v] for r in primary_runs])) for v in levels],
         "primary_seeds": [[r["heldout"]["sweep"][v] for v in levels] for r in primary_runs],
         "oracle": [float(np.mean([r["heldout"]["sweep"][v] for r in oracle_runs])) for v in levels]}

novel_gap = t_interval([s["RI_novel"] - s["RI_H"] for s in seeds])

validity = [{"id": k, "desc": VALID_DESC[k], "verdict": v["verdict"], "failed": v["failed_checks"],
             "passes": v["passes"]} for k, v in summary["validity"].items()]

data = {
    "arms": arms, "seeds": seeds, "sweep": sweep, "validity": validity,
    "comparisons": summary["comparisons"], "prior": PRIMARY_PRIOR, "brier": summary["prior_brier"],
    "novel_gap": list(novel_gap),
}

html = (REPO / "experiments/report_templates/v3.html").read_text()
html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
OUT.write_text(html)
print(OUT, f"{OUT.stat().st_size / 1024:.0f} KB")
