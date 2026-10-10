"""
Score the held-out-need experiment exactly as pre-registered (docs/HELDOUT_DESIGN.md §6, §8).

Reads outputs/v3/validity/*.json and outputs/v3/assays/*.json; writes outputs/v3/summary.json
and outputs/v3/results_tables.md. Uses only src/scoring_v3.py for verdicts, CIs and tests.

    python experiments/analyze_v3.py [--root outputs/v3]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_heldout import ARMS, PRIMARY  # noqa: E402
from src.scoring_v3 import (  # noqa: E402
    brier,
    clopper_pearson,
    gap_verdict,
    heldout_verdict,
    holm,
    seed_transfers,
    sign_flip_test,
    t_interval,
    trained_verdict,
)

HELDOUT_KEYS = ("RI_H", "RI_spec_H", "RI_dir_H", "SI_H", "RI_novel", "SI_novel")
EXPECTED_CONTROLS = {
    "oracle": ("transfers", []), "nearest": ("negligible", []), "random": ("negligible", []),
    "need_novelty": ("revaluation without wanting signature", ["specific"]),
    "veto": ("revaluation without wanting signature", ["seeking"]),
    "breakdown": ("revaluation without wanting signature", ["seeking"]),
    "unusual_seeker": ("revaluation without wanting signature", ["directional"]),
}
# Prior over the primary verdict, collapsed to four categories (§2); scored with Brier.
PRIMARY_PRIOR = {"transfers": 0.45, "without signature": 0.15, "negligible": 0.25,
                 "inconclusive": 0.15}
COMPARISONS = [
    # (label, metric, arm a, arm b); paired over the seeds the arms share
    ("C1 innate vs learned binding (RI_H)", "RI_H", PRIMARY, "learned-level-K6"),
    ("C2 level-only vs code read-out (RI_H)", "RI_H", PRIMARY, "innate-code-K6"),
    ("C3 relational vs MLP (gap)", "gap", PRIMARY, "mlp-K6"),
    ("C4 six vs two trained needs (RI_H)", "RI_H", PRIMARY, "innate-level-K2"),
    ("C5 H full side varies vs constant (RI_H)", "RI_H", PRIMARY, "innate-level-K6-constH"),
]


def category(verdict: str) -> str:
    if verdict == "transfers":
        return "transfers"
    if verdict == "negligible":
        return "negligible"
    if verdict == "revaluation without wanting signature":
        return "without signature"
    return "inconclusive"


def per_seed(results: list[dict]) -> dict[str, dict[int, float]]:
    out = defaultdict(dict)
    for r in results:
        s = r["seed"]
        h = r["heldout"]
        for k in HELDOUT_KEYS:
            out[k][s] = h[k]
        out["RI_T"][s] = r["trained"]["RI_T"]
        out["SI_T"][s] = r["trained"]["SI_T"]
        out["gap"][s] = r["trained"]["RI_T"] - h["RI_H"]
        if "competence" in r:
            out["drive"][s] = r["competence"]["mean_drive"]
        out["choice_dep_H"][s] = h["conditions"]["dep_H"]["choice_rate"]
        out["choice_dep_T"][s] = r["trained"]["conditions"]["dep_T"]["choice_rate"]
        for level, p in h["sweep"].items():
            out[f"sweep_{level}"][s] = p
    return out


def ci(values) -> dict:
    mean, lo, hi = t_interval(list(values))
    return {"mean": mean, "lo": lo, "hi": hi}


def summarize_arm(results: list[dict]) -> dict:
    v = per_seed(results)
    arr = {k: np.array([d[s] for s in sorted(d)]) for k, d in v.items()}
    verdict = heldout_verdict(arr["RI_H"], arr["RI_spec_H"], arr["RI_dir_H"], arr["SI_H"])
    count = sum(seed_transfers(a, b, c, d) for a, b, c, d in
                zip(arr["RI_H"], arr["RI_spec_H"], arr["RI_dir_H"], arr["SI_H"], strict=True))
    n = len(arr["RI_H"])
    return {
        "n_seeds": n,
        "heldout": verdict,
        "trained": trained_verdict(arr["RI_T"]),
        "gap": gap_verdict(arr["gap"]),
        "seed_transfer_count": {"k": int(count), "n": n, "ci": clopper_pearson(int(count), n)},
        "metrics": {k: {**ci(a), "seeds": a.tolist()} for k, a in arr.items()},
    }


def comparisons(by_arm: dict[str, dict]) -> dict:
    raw = {}
    for label, metric, a, b in COMPARISONS:
        if a in by_arm and b in by_arm:
            da, db = by_arm[a][metric], by_arm[b][metric]
            shared = sorted(set(da) & set(db))
            diffs = [da[s] - db[s] for s in shared]
            raw[label] = {"diff": float(np.mean(diffs)), "p": sign_flip_test(diffs), "n": len(shared)}
    for arm, d in by_arm.items():
        gaps = list(d["gap"].values())
        raw[f"gap != 0 within {arm}"] = {"diff": float(np.mean(gaps)), "p": sign_flip_test(gaps),
                                         "n": len(gaps)}
    adjusted = holm({k: v["p"] for k, v in raw.items()})
    return {k: {**v, "p_holm": adjusted[k]} for k, v in raw.items()}


def validity(root: Path) -> dict:
    out = {}
    for name, (want, failed) in EXPECTED_CONTROLS.items():
        rs = [json.loads(p.read_text()) for p in sorted((root / "validity").glob(f"{name}_seed*.json"))]
        if not rs:
            continue
        a = {k: np.array(list(d.values())) for k, d in per_seed(rs).items()}
        v = heldout_verdict(a["RI_H"], a["RI_spec_H"], a["RI_dir_H"], a["SI_H"])
        ok = v["verdict"] == want and set(failed) <= set(v["failed_checks"])
        out[name] = {**v, "expected": want, "passes": bool(ok), "RI_H": ci(a["RI_H"])}
    return out


def fmt(m: dict) -> str:
    return f"{m['mean']:+.2f} [{m['lo']:+.2f}, {m['hi']:+.2f}]"


def tables(summary: dict) -> str:
    lines = ["## Validity: the scoring rule applied to scripted controls", "",
             "| Control | Verdict | Failed checks | Expected | OK |", "|---|---|---|---|---|"]
    for name, v in summary["validity"].items():
        lines.append(f"| `{name}` | {v['verdict']} | {', '.join(v['failed_checks']) or '—'} | "
                     f"{v['expected']} | {'yes' if v['passes'] else '**NO**'} |")
    lines += ["", "## Arms (mean over seeds, 95% t-interval)", "",
              "| Arm | Seeds | Zero-shot verdict | RI_H | RI_spec_H | RI_dir_H | SI_H | RI_T | "
              "In-distribution | Gap | Seeds transferring |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, s in summary["arms"].items():
        m = s["metrics"]
        failed = f" (not {', '.join(s['heldout']['failed_checks'])})" if s["heldout"]["failed_checks"] else ""
        c = s["seed_transfer_count"]
        lines.append(f"| `{arm}` | {s['n_seeds']} | **{s['heldout']['verdict']}**{failed} | {fmt(m['RI_H'])} | "
                     f"{fmt(m['RI_spec_H'])} | {fmt(m['RI_dir_H'])} | {fmt(m['SI_H'])} | {fmt(m['RI_T'])} | "
                     f"{s['trained']} | {s['gap']} | {c['k']}/{c['n']} |")
    lines += ["", "## Secondary and covariates", "",
              "| Arm | RI_novel | SI_novel | Mean drive | Choice rate dep_H / dep_T |", "|---|---|---|---|---|"]
    for arm, s in summary["arms"].items():
        m = s["metrics"]
        drive = f"{m['drive']['mean']:.3f}" if "drive" in m else "—"
        lines.append(f"| `{arm}` | {fmt(m['RI_novel'])} | {fmt(m['SI_novel'])} | {drive} | "
                     f"{m['choice_dep_H']['mean']:.2f} / {m['choice_dep_T']['mean']:.2f} |")
    lines += ["", "## Planned comparisons (paired sign-flip tests, Holm-corrected)", "",
              "| Comparison | Mean difference | n | p | p (Holm) |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['diff']:+.3f} | {v['n']} | {v['p']:.4f} | {v['p_holm']:.4f} |"
              for k, v in summary["comparisons"].items()]
    if "prior_brier" in summary:
        lines += ["", f"Brier score of the pre-registered prior on the primary verdict: "
                  f"{summary['prior_brier']:.3f} (0 = certain and right; 0.5 = uniform over 4)."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/v3"))
    args = parser.parse_args()
    runs = defaultdict(list)
    for path in sorted((args.root / "assays").glob("*.json")):
        r = json.loads(path.read_text())
        runs[r["arm"]].append(r)
    arms = {arm: summarize_arm(runs[arm]) for arm in ARMS if runs.get(arm)}
    summary = {"validity": validity(args.root), "arms": arms,
               "comparisons": comparisons({a: per_seed(runs[a]) for a in arms})}
    if PRIMARY in arms:
        summary["primary"] = {"arm": PRIMARY, **arms[PRIMARY]["heldout"]}
        summary["prior_brier"] = brier(PRIMARY_PRIOR, category(arms[PRIMARY]["heldout"]["verdict"]))
    (args.root / "summary.json").write_text(json.dumps(summary, indent=1))
    md = tables(summary)
    (args.root / "results_tables.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
