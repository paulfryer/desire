import numpy as np
import pytest

from src.agents.multi import CONTROLS, item_tokens, need_levels
from src.assays_n import assay_heldout, assay_trained, ctrl_h, dep_h, dep_t, guard_h, over_h
from src.env import SETPOINT
from src.env_n import CODE_DIM, MultiNutrientGrid, make_world
from src.scoring_v3 import gap_verdict, heldout_verdict, sign_flip_test, t_interval, trained_verdict


def test_world_geometry_is_orthonormal_and_shared_across_arms():
    w = make_world(3)
    assert w.n == 7 and w.held_out == 6 and w.trained == [0, 1, 2, 3, 4, 5]
    gram = np.vstack([w.codes, w.spare_code]) @ np.vstack([w.codes, w.spare_code]).T
    assert np.allclose(gram, np.eye(8), atol=1e-9)
    assert np.array_equal(make_world(3, n_depleting=2).codes, w.codes)
    assert np.isclose(w.depletion.sum(), make_world(3, n_depleting=2).depletion.sum())


def test_held_out_never_falls_but_varies_on_its_full_side():
    w = make_world(1)
    env = MultiNutrientGrid(32, w, seed=0)
    env.reset()
    start = env.h[:, w.held_out].copy()
    assert start.min() >= 0.5 and start.max() <= 0.8 and start.std() > 0.05
    rng = np.random.default_rng(0)
    for _ in range(200):
        env.step(rng.integers(0, 5, size=32))
        assert env.h[:, w.held_out].min() >= SETPOINT - 1e-9
    const = MultiNutrientGrid(8, make_world(1, held_out_constant=True), seed=0)
    const.reset()
    assert np.allclose(const.h[:, w.held_out], SETPOINT)


def test_observations_decode_and_needs_are_sorted():
    w = make_world(2)
    env = MultiNutrientGrid(4, w, seed=0)
    obs = env.reset()
    assert obs["items"].shape == (4, 14, CODE_DIM + 3) and obs["needs"].shape == (4, 7, CODE_DIM + 1)
    assert np.allclose(need_levels(obs, w), env.h, atol=1e-6)
    assert (np.diff(obs["needs"][..., CODE_DIM], axis=1) >= 0).all()
    for b, items in enumerate(item_tokens(obs, w)):
        assert len(items) == env.items[b].sum()
        for k, dr, dc in items:
            assert env.items[b, k, env.pos[b, 0] + dr, env.pos[b, 1] + dc]


def test_sparse_two_item_episodes_occur():
    env = MultiNutrientGrid(400, make_world(0), seed=0)
    env.reset()
    share = (env.items.sum(axis=(1, 2, 3)) == 2).mean()
    assert 0.15 < share < 0.4


def test_probe_conditions_set_the_intended_levels():
    w = make_world(0)
    lv, (target, other) = dep_h(w, 5)
    assert target == w.held_out and lv[target] == 0.05 and lv[other] == 0.3 and other in w.trained
    assert ctrl_h(w, 5)[0][w.held_out] == SETPOINT and over_h(w, 5)[0][w.held_out] == 1.0
    lv, (target, other) = guard_h(w, 5)
    deprived = [k for k, v in lv.items() if v == 0.05]
    assert lv[w.held_out] == SETPOINT and len(deprived) == 1 and deprived[0] not in (target, other)
    lv, (k, j) = dep_t(w, 5)
    assert k != j and {k, j} <= set(w.trained) and lv[k] == 0.05


EXPECTED = {
    "oracle": "transfers",
    "nearest": "negligible",
    "random": "negligible",
    "need_novelty": ("revaluation without wanting signature", ["specific"]),
    "veto": ("revaluation without wanting signature", ["seeking"]),
    "breakdown": ("revaluation without wanting signature", ["seeking"]),
    "unusual_seeker": ("revaluation without wanting signature", ["directional"]),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_scoring_rule_sorts_every_control(name):
    seeds = range(6)
    rows = []
    for s in seeds:
        w = make_world(s)
        probes = 200 if name in ("random", "nearest") else 80   # random rarely chooses: needs more
        r = assay_heldout(CONTROLS[name](w, s), w, probes, seed=s)
        rows.append([r["RI_H"], r["RI_spec_H"], r["RI_dir_H"], r["SI_H"]])
    v = heldout_verdict(*np.array(rows).T)
    want = EXPECTED[name]
    if isinstance(want, tuple):
        assert v["verdict"] == want[0] and set(want[1]) <= set(v["failed_checks"])
    else:
        assert v["verdict"] == want


def test_oracle_learns_trained_needs():
    w = make_world(0)
    assert assay_trained(CONTROLS["oracle"](w, 0), w, 60, seed=1)["RI_T"] > 0.9


def test_scoring_rule_on_synthetic_patterns():
    ones, zeros = np.full(10, 0.95), np.zeros(10)
    assert heldout_verdict(ones, ones, ones, ones)["verdict"] == "transfers"
    assert heldout_verdict(zeros + 0.02, zeros, zeros, zeros)["verdict"] == "negligible"
    noisy = np.array([0.0, 0.9] * 5)
    assert heldout_verdict(noisy, noisy, noisy, noisy)["verdict"] == "inconclusive"
    assert trained_verdict(ones) == "learned" and gap_verdict(ones - 0.9) == "complete"
    assert gap_verdict(ones) == "incomplete"


def test_t_interval_coverage_on_bimodal_seeds():
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(4000):
        x = np.where(rng.random(10) < 0.5, rng.normal(0.05, 0.05, 10), rng.normal(0.85, 0.05, 10))
        _, lo, hi = t_interval(x)
        hits += lo <= 0.45 <= hi
    assert hits / 4000 > 0.92


def test_sign_flip_test():
    assert sign_flip_test(np.full(10, 0.5)) < 0.01
    assert sign_flip_test(np.array([0.1, -0.1] * 5)) > 0.9
