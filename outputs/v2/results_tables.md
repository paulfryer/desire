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

⚠ = below the competence criterion (choice rate < 0.9); assay verdicts not interpretable.

## Pre-registered predictions

| ID | Prediction | Outcome |
|---|---|---|
| P1 | oracle passes T1/T2; no-desire controls fail; need_novelty passes RI but fails the guard | supported |
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

## T2b incentive-learning control (exploratory)

| Agent | RI before | RI after 3 deprived C exposures | RI_spec after |
|---|---|---|---|
| `homeo` | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] |
| `mb_reward` | +0.15 [+0.00, +0.30] | +0.05 [+0.00, +0.15] | +0.05 [+0.00, +0.15] |
| `aif` | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] |
| `aif_no_novelty` | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] | +1.00 [+1.00, +1.00] |
