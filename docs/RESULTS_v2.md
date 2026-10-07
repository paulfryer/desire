# Wanting vs. Habit — v2 Results

*Status: planners and controls complete (10 seeds each). Model-free agents: **pending** (training in progress); this section is filled in when they finish.*

All numbers are means over 10 independently trained seeds, with 95% percentile-bootstrap CIs over seeds. Verdicts follow the pre-registered rules in `docs/RESEARCH_DESIGN_v2.md` §6 and Amendment 2. Generated tables: `outputs/v2/results_tables.md`. Per-seed data: `outputs/v2/planners/*.json`, `outputs/v2/model_free_assays/*.json`.

## 1. The assays are valid (P1 — supported)

| Agent | Ground truth | DI (T1) | RI (T2) | RI_spec | Choice rate |
|---|---|---|---|---|---|
| `oracle` | ideal wanting | +1.00 | +1.00 | +1.00 | 1.00 |
| `nearest` | outcome-general habit | +0.02 [−0.02, +0.04] | 0.00 | 0.00 | 1.00 |
| `fixed_habit` | outcome-specific S–R habit | 0.00 | 0.00 | 0.00 | 1.00 |
| `need_novelty` | seeks the unfamiliar item when needy | +0.01 | **+1.00** | **0.00** | 1.00 |
| `random` | nothing | +0.07 | −0.05 | −0.02 | 0.34 (fails competence) |

Every no-desire control fails both tests, and the oracle passes both. The `need_novelty` row is the reason for Amendment 2. Without the specificity guard, an agent that simply tries the unfamiliar item whenever it is in need scores a perfect zero-shot revaluation index. Our own pilot active-inference agent did exactly this while knowing nothing about C (§5). Any Berridge-style revaluation test of an artificial agent needs this guard.

## 2. Planners

| Agent | Value computed from | DI (T1) | RI (T2) | RI_spec | Mean drive | C eaten in training |
|---|---|---|---|---|---|---|
| `homeo` | learned outcome model × innate drive | **+1.00** | **+1.00** | **+1.00** | 0.100 | 53 |
| `aif` | learned outcome model × preferences over interoception (expected free energy) | **+1.00** | **+1.00** | **+1.00** | 0.101 | 109 |
| `aif_no_novelty` | same, no novelty term | **+1.00** | **+1.00** | **+1.00** | 0.101 | 107 |
| `mb_reward` | learned outcome model × **learned scalar reward model** | **+1.00** | +0.15 [0.00, +0.30] | +0.15 [0.00, +0.30] | 0.139 | 358 |

- **P5 supported.** Agents that compute an outcome's value at decision time, from a learned model of what each item does and an innate need function, pass both devaluation and zero-shot revaluation in every seed.
- **P6 supported.** The learned-reward model-based agent passes devaluation (an in-distribution test) and **fails zero-shot revaluation**. It ate C more than any other planner during training, so the failure is not lack of experience with C.
- **P7 supported.** Active inference and the homeostatic planner are indistinguishable (ΔDI = ΔRI = 0.00; permutation p = 1.0). With matched ingredients, the framework label makes no difference.
- **P8 not supported.** The novelty term did not increase sampling of C (109 vs 107 exposures, Holm p = 1.0). ε-exploration already gave every planner ample experience of C, and with Dirichlet counts in the hundreds, parameter information gain is negligible next to risk.

### Why `mb_reward` fails zero-shot revaluation (exploratory)

`experiments/diagnose_mb_reward.py` retrains each seed and queries the learned reward model (`outputs/v2/mb_reward_extrapolation.json`).

- **The immediate reward for eating C when deprived usually extrapolates correctly.** Predicted reward at h_C = 0.05 (never experienced) is median +1.39 against a true +1.77, and is clearly positive in 9 of 10 seeds. The network partly transfers "eating the deficient nutrient is good" from A and B.
- **The plan value does not.** A plan also sums predicted rewards for every non-eating step, and in a deprived-of-C state all of those steps sit at C-levels never seen in training. Those extrapolations are large and erratic. For example, seed 8 predicts +12.9 for staying put (true −0.8) and +3.1 for eating C (true +1.0). The resulting choices are effectively arbitrary: per item pair, seeds either pick C, pick the other item, or stay.

The lesson is about where valuation lives, not about having a model. An agent with an accurate model of what C does still cannot value it in a novel need state if the value of states is itself a learned scalar function. Revaluation came for free only when the value of a predicted internal state was computed by an innate function of need.

### Incentive-learning control (T2b, exploratory)

After three forced consumptions of C while deprived, with learning on, `homeo` and `aif` were unchanged (already at ceiling). `mb_reward` did not improve (RI +0.15 → +0.05). Three new samples among a buffer of thousands of consumption events barely move a refit MLP. In this implementation the learned-reward agent is a slow incentive learner, unlike rats (Balleine 1992), which typically need only a few such experiences.

## 3. Model-free agents

*Pending — training 4 agents × 10 seeds × 10M steps.*

## 4. Deviations from the original design

All of these were made before the results they could affect, and are recorded with dates in the design doc's amendments.

1. Setpoint 0.7 → 0.5, because C's effect was censored at the ceiling and unlearnable (Amendment 2).
2. Specificity guard `RI_spec` and the `need_novelty` control added (Amendment 2).
3. Novelty zero at test; indices conditional on a choice; exploratory T2b (Amendment 1).
4. Model-free architecture and training settings fixed on pilot seeds by competence only (Amendment 3).

## 5. Things that went wrong and were caught

- **v1 battery** (`docs/LEGACY_VALIDITY.md`): three predicates were passed by no-desire controls, two were artifacts of scoring bugs, and one was never measured.
- **An active-inference agent "passed" zero-shot revaluation out of ignorance.** In the first pilot, a flat Dirichlet prior over C's (unobservable) effect plus the KL-risk term made the uncertain item look attractive under deprivation (RI 0.77). This is a general validity threat: under deprivation, uncertainty-seeking mimics wanting.
- **The scorecard initially marked P8 "supported"** on a raw difference of 109 vs 107. It now uses the planned permutation test, as §7 requires.
