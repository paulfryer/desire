## Assay results (mean over seeds, 95% bootstrap CI)

| Agent | Family | Seeds | Choice rate T1/T2 | Mean drive | DI (T1) | RI (T2) | RI_spec | T1 | T2 |
|---|---|---|---|---|---|---|---|---|---|
| `random` | control | 10 | 0.34 / 0.33 ⚠ | 0.259 | +0.07 [+0.00, +0.14] | -0.05 [-0.08, -0.02] | -0.02 [-0.07, +0.03] | fail | fail |
| `nearest` | control | 10 | 1.00 / 1.00 | 0.398 | +0.01 [-0.02, +0.04] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | fail | fail |
| `fixed_habit` | control | 10 | 1.00 / 1.00 | 0.384 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | fail | fail |
| `need_novelty` | control | 10 | 1.00 / 1.00 | 0.441 | +0.01 [-0.03, +0.05] | +1.00 [+1.00, +1.00] | +0.00 [+0.00, +0.00] | fail | fail (guard) |
| `oracle` | control | 10 | 1.00 / 1.00 | 0.100 | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | pass | pass |
| `homeo` | planner | 10 | 1.00 / 1.00 | 0.100 | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | pass | pass |
| `mb_reward` | planner | 10 | 1.00 / 0.93 | 0.139 | +1.00 [+1.00, +1.00] | +0.15 [+0.00, +0.30] | +0.15 [+0.00, +0.30] | pass | fail |
| `aif` | planner | 10 | 1.00 / 1.00 | 0.101 | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | pass | pass |
| `aif_no_novelty` | planner | 10 | 1.00 / 1.00 | 0.101 | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | pass | pass |
| `ppo_free_intero` | model-free | 10 | 0.91 / 0.80 ⚠ | 0.140 | +0.63 [+0.51, +0.71] | +0.02 [+0.00, +0.04] | +0.01 [-0.01, +0.03] | pass | fail |
| `ppo_free_blind` | model-free | 10 | 0.45 / 0.34 ⚠ | 0.240 | +0.02 [-0.02, +0.07] | +0.00 [-0.01, +0.01] | -0.00 [-0.01, +0.01] | fail | fail |
| `ppo_clamped` | model-free | 10 | 1.00 / 1.00 | 0.180 | -0.01 [-0.03, +0.01] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | fail | fail |
| `rppo_free_intero` | model-free | 10 | 0.91 / 0.87 ⚠ | 0.128 | +0.54 [+0.47, +0.63] | +0.00 [-0.01, +0.01] | -0.02 [-0.04, -0.01] | pass | fail |

⚠ = below the competence criterion (choice rate < 0.9); assay verdicts not interpretable.

## Model-free competence by seed (exploratory sensitivity, Amendment 4)

| Agent | Competent seeds | DI | RI | RI_spec (competent seeds only) |
|---|---|---|---|---|
| `ppo_free_intero` | 0/10 | — | — | — |
| `ppo_free_blind` | 0/10 | — | — | — |
| `ppo_clamped` | 10/10 | -0.01 [-0.03, +0.01] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| `rppo_free_intero` | 3/10 | — | — | — |

## Pre-registered predictions

| ID | Prediction | Outcome |
|---|---|---|
| P1 | oracle passes T1/T2; no-desire controls fail; need_novelty passes RI but fails the guard | supported |
| P2 | ppo_free_blind and ppo_clamped fail T1 | not interpretable (ppo_free_blind below competence criterion) (T1 fail, fail) |
| P3 | ppo_free_intero passes T1 but fails T2 | not interpretable (ppo_free_intero below competence criterion) (T1 pass, T2 fail) |
| P4 | rppo_free_intero passes T1 but fails T2 | not interpretable (rppo_free_intero below competence criterion) (T1 pass, T2 fail) |
| P5 | homeo, aif, aif_no_novelty pass T1 and T2 | supported |
| P6 | mb_reward passes T1; T2 partial or fail | supported (T2 fail, RI +0.15) |
| P7 | |RI_aif - RI_homeo| < 0.15 and |DI_aif - DI_homeo| < 0.15 | supported (ΔRI 0.00, ΔDI 0.00) |
| P8 | novelty increases C exposures in training but not T2 | not supported (C exposures 109 vs 107, Holm p = 1.00) |
| P9 | T3 exploratory (no directional prediction) | see T3 figure |

## Planned comparisons (permutation test on seed scores, Holm-corrected)

| Comparison | Difference | p | p (Holm) |
|---|---|---|---|
| aif vs homeo: DI | +0.000 | 1.0000 | 1.0000 |
| aif vs homeo: RI | +0.000 | 1.0000 | 1.0000 |
| aif vs aif_no_novelty: C exposures | +1.700 | 0.7985 | 1.0000 |
| aif vs aif_no_novelty: RI | +0.000 | 1.0000 | 1.0000 |
| ppo_free_intero vs ppo_clamped: DI | +0.647 | 0.0001 | 0.0012 |
| ppo_free_intero vs ppo_free_blind: DI | +0.612 | 0.0001 | 0.0012 |
| ppo_free_intero vs homeo: RI | -0.979 | 0.0001 | 0.0012 |
| rppo_free_intero vs ppo_free_intero: RI | -0.020 | 0.1068 | 0.5340 |

## T2b incentive-learning control (exploratory)

| Agent | RI before | RI after 3 deprived C exposures | RI_spec after |
|---|---|---|---|
| `homeo` | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] |
| `mb_reward` | +0.15 [+0.00, +0.30] | +0.05 [+0.00, +0.15] | +0.05 [+0.00, +0.15] |
| `aif` | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] |
| `aif_no_novelty` | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] |

## Amendment 5 extension: interoceptive model-free agents continued to 30M steps

| Agent | Seeds | Choice rate T1/T2 | Mean drive | DI (T1) | RI (T2) | RI_spec | T1 | T2 |
|---|---|---|---|---|---|---|---|---|
| `ppo_free_intero` | 10 | 0.97 / 0.99 | 0.108 | +0.87 [+0.83, +0.91] | -0.00 [-0.00, +0.00] | -0.00 [-0.01, +0.00] | pass | fail |
| `rppo_free_intero` | 10 | 1.00 / 0.99 | 0.103 | +1.00 [+0.99, +1.00] | +0.00 [+0.00, +0.00] | -0.00 [-0.00, +0.00] | pass | fail |

| ID | Prediction (30M) | Outcome |
|---|---|---|
| P3 | ppo_free_intero passes T1 but fails T2 | supported (T1 pass, T2 fail) |
| P4 | rppo_free_intero passes T1 but fails T2 | supported (T1 pass, T2 fail) |
