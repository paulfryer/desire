# Claude Code Project Context

Research codebase asking which computational ingredients make an artificial agent *want* an outcome (need-sensitive valuation of a represented outcome) rather than act from habit, measured with outcome-devaluation and zero-shot-revaluation assays validated on ground-truth control agents.

**Start here:** `docs/RESEARCH_DESIGN_v2.md` (design and pre-registered predictions, with dated amendments), `docs/RESULTS_v2.md` (results), `docs/LEGACY_VALIDITY.md` (why the v1 "AIF vs RL desire predicates" project was retired). The v1 code lives on `main`; this line of work is on branch `wanting-vs-habit`.

## Environment

```bash
.venv/bin/python -m pytest tests          # Python 3.12 venv created with uv
```

Always set `OMP_NUM_THREADS=1` when running several training processes in parallel.

## Key operations

```bash
# Planners + ground-truth controls (train, then assays). ~15 min for 10 seeds.
python experiments/run_planners.py --seeds 0-9

# Model-free agents (settings fixed in Amendment 3). Skips finished runs, so it is resumable.
python experiments/launch_model_free.py --seeds 0-9 --parallel 8 --steps 10000000 \
    --features split --norm-reward --n-envs 64 --n-steps 256 --batch-size 4096
tail outputs/v2/logs/launcher.log                       # progress
tail -n1 outputs/v2/model_free/<agent>/seed_<n>/training_log.csv

# Assays on finished model-free runs (the launcher also runs this at the end)
python experiments/run_model_free_assays.py

# Amendment 5: continue the interoceptive agents to 30M (see README for the full command)
python experiments/launch_model_free.py --agents ppo_free_intero rppo_free_intero ... \
    --resume-from outputs/v2/model_free --out outputs/v2/model_free_30M \
    --assay-out outputs/v2/model_free_assays_30M

# Exploratory need-sweep curves (P(choose target) vs need level)
python experiments/need_sweep.py --seeds 0-9 && python experiments/need_sweep.py --plot

# Tables, prediction scorecard, figures -> outputs/v2/
python experiments/analyze_v2.py
```

## Rules for this project

- **Pre-registration:** predictions in the design doc are fixed. Any design change goes in a new dated amendment, made *before* the results it could affect, with the reason. Never tune anything by looking at assay indices (DI, RI, RI_spec); choose settings by competence only (mean drive, choice rate).
- **Validity first:** a new assay must be passed by `oracle` and failed by the no-desire controls (`random`, `nearest`, `fixed_habit`, `need_novelty`) before any learned agent is scored on it.
- **Replication unit is the training seed.** Report seed-level bootstrap CIs; never treat probes or episodes from one seed as independent replicates.
- Pilot runs use seeds ≥ 100 and are never reported.

## Layout

- `src/env.py`: NutrientGrid (batched; egocentric observations; free/clamped needs; interoception flag)
- `src/assays.py`: T0 competence, T1 devaluation (DI), T2 revaluation (RI) + specificity guard (RI_spec)
- `src/agents/`: `controls.py`, `planners.py` (homeo/oracle, mb_reward, aif), `model_free.py` (SB3 adapters)
- `src/stats.py`: bootstrap CIs, permutation tests, Holm, verdict rule
- `experiments/`: runners, launcher, analysis, diagnostics
- `outputs/v2/`: per-seed JSON results (committed); models, checkpoints and logs are git-ignored
