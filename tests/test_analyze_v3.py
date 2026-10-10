"""The frozen analysis, run end to end on synthetic results with known answers."""

import json
import sys

import numpy as np

from experiments import analyze_v3
from experiments.train_heldout import ARMS, SEEDS

PATTERNS = {   # arm: (RI_H, RI_spec_H, RI_dir_H, SI_H, RI_T) seed means; spread 0.05
    "innate-level-K6": (0.9, 0.9, 0.9, 0.8, 0.9),        # transfers
    "learned-level-K6": (0.0, 0.0, 0.0, -0.9, 0.9),      # negligible
    "innate-code-K6": (0.5, 0.5, 0.5, 0.0, 0.9),         # without signature (not seeking)
    "mlp-K6": (0.3, 0.3, 0.3, 0.2, 0.3),                  # spread below makes this inconclusive
    "innate-level-K2": (0.9, 0.0, 0.9, 0.8, 0.9),         # without signature (not specific)
    "innate-level-K6-constH": (0.9, 0.9, 0.0, 0.8, 0.9),  # without signature (not directional)
}
EXPECTED = {
    "innate-level-K6": ("transfers", []), "learned-level-K6": ("negligible", []),
    "innate-code-K6": ("revaluation without wanting signature", ["seeking"]),
    "mlp-K6": ("inconclusive", []),
    "innate-level-K2": ("revaluation without wanting signature", ["specific"]),
    "innate-level-K6-constH": ("revaluation without wanting signature", ["directional"]),
}


def fake(arm, seed, rng):
    ri_h, spec, dir_, si, ri_t = PATTERNS[arm]
    spread = 0.6 if arm == "mlp-K6" else 0.05
    j = lambda m: float(m + rng.normal(0, spread))  # noqa: E731
    cond = {"choice_rate": 0.95}
    return {"arm": arm, "seed": seed, "competence": {"mean_drive": 0.3},
            "trained": {"RI_T": j(ri_t), "SI_T": 0.5, "conditions": {"dep_T": cond}},
            "heldout": {"RI_H": j(ri_h), "RI_spec_H": j(spec), "RI_dir_H": j(dir_), "SI_H": j(si),
                        "RI_novel": 0.0, "SI_novel": 0.0, "sweep": {"0.05": 0.9},
                        "conditions": {"dep_H": cond}}}


def test_analysis_end_to_end_on_synthetic_results(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    (tmp_path / "assays").mkdir()
    (tmp_path / "validity").mkdir()
    for arm in ARMS:
        for s in SEEDS[arm]:
            (tmp_path / "assays" / f"{arm}_seed{s}.json").write_text(json.dumps(fake(arm, s, rng)))
    monkeypatch.setattr(sys, "argv", ["analyze_v3.py", "--root", str(tmp_path)])
    analyze_v3.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    for arm, (verdict, failed) in EXPECTED.items():
        got = summary["arms"][arm]["heldout"]
        assert got["verdict"] == verdict, arm
        assert set(failed) <= set(got["failed_checks"]), arm
    assert summary["arms"]["innate-level-K6"]["n_seeds"] == 20
    assert summary["primary"]["verdict"] == "transfers"
    assert 0 <= summary["prior_brier"] <= 2
    holm = [v["p_holm"] for v in summary["comparisons"].values()]
    assert len(holm) == 5 + len(ARMS) and all(0 <= p <= 1 for p in holm)
