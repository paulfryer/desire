# Wanting vs. Habit in Artificial Agents

Which computational ingredients make an agent *want* an outcome — value it according to what it currently needs — rather than repeat a habit? And can we tell from behavior alone, using tests checked against agents whose answer we already know?

This repository borrows two tests from animal-learning research:

- **Outcome devaluation** (Dickinson & Balleine): satiate the agent on one outcome, then see whether it switches, without any chance to relearn.
- **Zero-shot revaluation** (Robinson & Berridge 2013): put the agent in a need state it has never experienced, and see whether it immediately goes for the one item that fixes it.

The tests are run on a set of agents built from controlled ingredients:
- model-free RL, with and without the ability to sense its needs (interoception), and with need levels varied or fixed during training
- recurrent model-free RL
- model-based RL with a learned reward model
- a homeostatic planner
- active inference
- ground-truth control agents that each test must correctly sort before it is trusted

> **History.** This project began as "do active-inference agents show desire-like signatures that RL agents don't?" (v1, the paper draft in `paper/`). That question was retired in October 2026: its six behavioral predicates turned out to be passed by agents with no desire at all, or to depend on scoring bugs (`docs/LEGACY_VALIDITY.md`). The v1 code remains on `main`; this branch replaces it.

## Documents

| File | Contents |
|---|---|
| `docs/RESEARCH_DESIGN_v2.md` | The question, environment, assays, agents and the **pre-registered predictions** (with dated amendments made before the relevant results) |
| `docs/RESULTS_v2.md` | Results against those predictions |
| `docs/LITERATURE.md` | Literature review and novelty assessment |
| `docs/LEGACY_VALIDITY.md` | Why the v1 predicates were retired |

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy scipy pandas matplotlib gymnasium torch \
    stable-baselines3 sb3-contrib scikit-learn pytest ruff
.venv/bin/python -m pytest tests
```

## Reproducing

```bash
# Planners and ground-truth controls: train (planners), then run every assay. ~15 min.
python experiments/run_planners.py --seeds 0-9

# Model-free agents: 4 agents x 10 seeds, then assays on final models and checkpoints.
python experiments/launch_model_free.py --seeds 0-9 --steps <steps> --parallel 8 [training flags]

# Tables, prediction scorecard and figures -> outputs/v2/
python experiments/analyze_v2.py
```

The exact training flags and step budgets used for the reported runs are recorded in each run's `outputs/v2/model_free/<agent>/seed_<n>/run.json` and in `docs/RESULTS_v2.md`.

## Layout

```
src/
  env.py              NutrientGrid: batched homeostatic gridworld (3 nutrients, drive-reduction reward)
  assays.py           T0 competence, T1 devaluation, T2 zero-shot revaluation (+ specificity guard)
  stats.py            seed-level bootstrap CIs, permutation tests, Holm, pre-registered verdicts
  agents/
    controls.py       random, nearest, fixed_habit, need_novelty
    planners.py       homeostatic planner / oracle, learned-reward model-based, active inference
    model_free.py     stable-baselines3 PPO / recurrent PPO adapters
experiments/          runners, analysis, diagnostics
tests/                unit tests
outputs/v2/           per-seed assay results (JSON), summary, tables, figures
```
