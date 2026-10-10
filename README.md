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

## Main result

Agents that compute an outcome's value at decision time, from a learned model of what each item does and an innate need function, pass both tests. That holds whether the agent is a homeostatic planner or active inference; the two are indistinguishable.

Agents whose valuations are learned fail zero-shot revaluation:
- **Model-based agent with a learned reward model:** RI +0.15.
- **Recurrent model-free agent, trained to 30M steps:** forages as well as the oracle, passes devaluation perfectly (DI +1.00) and shows graded sensitivity to familiar needs. It still never once seeks the remedy for an unfamiliar need (RI 0.00).

In-distribution need sensitivity is not wanting. Details: `docs/RESULTS_v2.md`.

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

# Model-free agents: 4 agents x 10 seeds x 10M steps, then assays on final models and
# checkpoints (Amendment 3 settings). About 5 hours on a 10-core laptop.
python experiments/launch_model_free.py --seeds 0-9 --parallel 8 --steps 10000000 \
    --features split --norm-reward --n-envs 64 --n-steps 256 --batch-size 4096

# Amendment 5 extension: the two interoceptive agents continued to 30M steps.
python experiments/launch_model_free.py --agents ppo_free_intero rppo_free_intero \
    --seeds 0-9 --parallel 10 --steps 30000000 --features split --norm-reward --n-envs 64 \
    --n-steps 256 --batch-size 4096 --resume-from outputs/v2/model_free \
    --out outputs/v2/model_free_30M --assay-out outputs/v2/model_free_assays_30M

# Tables, prediction scorecard and figures -> outputs/v2/
python experiments/analyze_v2.py

# Exploratory need-sweep curves -> outputs/v2/need_sweep*/ and figures/need_sweep*.png
python experiments/need_sweep.py --seeds 0-9 && python experiments/need_sweep.py --plot
```

Set `OMP_NUM_THREADS=1` when running training jobs in parallel. Each run records its exact settings in `outputs/v2/model_free*/<agent>/seed_<n>/run.json`.

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
