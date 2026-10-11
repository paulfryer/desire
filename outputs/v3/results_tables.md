## Validity: the scoring rule applied to scripted controls

| Control | Verdict | Failed checks | Expected | OK |
|---|---|---|---|---|
| `oracle` | transfers | — | transfers | yes |
| `nearest` | negligible | — | negligible | yes |
| `random` | negligible | — | negligible | yes |
| `need_novelty` | revaluation without wanting signature | specific | revaluation without wanting signature | yes |
| `veto` | revaluation without wanting signature | seeking | revaluation without wanting signature | yes |
| `breakdown` | revaluation without wanting signature | seeking | revaluation without wanting signature | yes |
| `unusual_seeker` | revaluation without wanting signature | directional | revaluation without wanting signature | yes |

## Arms (mean over seeds, 95% t-interval)

| Arm | Seeds | Zero-shot verdict | RI_H | RI_spec_H | RI_dir_H | SI_H | RI_T | In-distribution | Gap | Seeds transferring |
|---|---|---|---|---|---|---|---|---|---|---|
| `innate-level-K6` | 20 | **inconclusive** | +0.28 [+0.13, +0.44] | +0.28 [+0.13, +0.44] | +0.28 [+0.13, +0.44] | -0.43 [-0.74, -0.12] | +0.96 [+0.95, +0.97] | learned | incomplete | 4/20 |
| `learned-level-K6` | 10 | **negligible** | -0.00 [-0.02, +0.01] | +0.00 [-0.00, +0.01] | -0.01 [-0.05, +0.02] | -0.94 [-1.01, -0.87] | +0.63 [+0.48, +0.78] | learned | incomplete | 0/10 |
| `innate-code-K6` | 10 | **inconclusive** | +0.10 [-0.05, +0.25] | +0.10 [-0.05, +0.25] | +0.10 [-0.05, +0.25] | -0.79 [-1.09, -0.49] | +0.98 [+0.97, +1.00] | learned | incomplete | 0/10 |
| `mlp-K6` | 10 | **negligible** | +0.02 [-0.08, +0.11] | +0.14 [+0.04, +0.25] | +0.25 [+0.18, +0.32] | +0.00 [-0.16, +0.17] | +0.00 [-0.06, +0.07] | negligible | complete | 0/10 |
| `innate-level-K2` | 10 | **negligible** | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -1.00 [-1.00, -1.00] | +0.75 [+0.69, +0.82] | learned | incomplete | 0/10 |
| `innate-level-K6-constH` | 10 | **inconclusive** | +0.18 [-0.06, +0.41] | +0.18 [-0.06, +0.41] | +0.18 [-0.06, +0.41] | -0.64 [-1.11, -0.17] | +0.97 [+0.96, +0.98] | learned | incomplete | 1/10 |

## Secondary and covariates

| Arm | RI_novel | SI_novel | Mean drive | Choice rate dep_H / dep_T |
|---|---|---|---|---|
| `innate-level-K6` | +0.68 [+0.57, +0.80] | +0.37 [+0.14, +0.61] | 0.493 | 0.94 / 1.00 |
| `learned-level-K6` | -0.00 [-0.02, +0.01] | -0.84 [-1.01, -0.68] | 0.665 | 0.72 / 0.83 |
| `innate-code-K6` | +0.44 [+0.18, +0.69] | -0.12 [-0.64, +0.39] | 0.489 | 0.96 / 1.00 |
| `mlp-K6` | -0.04 [-0.10, +0.03] | -0.02 [-0.11, +0.08] | 0.728 | 0.09 / 0.34 |
| `innate-level-K2` | +0.01 [-0.01, +0.04] | -0.97 [-1.02, -0.92] | 0.238 | 1.00 / 1.00 |
| `innate-level-K6-constH` | +0.64 [+0.39, +0.89] | +0.29 [-0.21, +0.78] | 0.494 | 0.87 / 1.00 |

## Planned comparisons (paired sign-flip tests, Holm-corrected)

| Comparison | Mean difference | n | p | p (Holm) |
|---|---|---|---|---|
| C1 innate vs learned binding (RI_H) | +0.218 | 10 | 0.0020 | 0.0195 |
| C2 level-only vs code read-out (RI_H) | +0.114 | 10 | 0.3555 | 1.0000 |
| C3 relational vs MLP (gap) | +0.759 | 10 | 0.0020 | 0.0195 |
| C4 six vs two trained needs (RI_H) | +0.216 | 10 | 0.0039 | 0.0195 |
| C5 H full side varies vs constant (RI_H) | +0.040 | 10 | 0.7305 | 1.0000 |
| gap != 0 within innate-level-K6 | +0.680 | 20 | 0.0000 | 0.0000 |
| gap != 0 within learned-level-K6 | +0.631 | 10 | 0.0020 | 0.0195 |
| gap != 0 within innate-code-K6 | +0.881 | 10 | 0.0020 | 0.0195 |
| gap != 0 within mlp-K6 | -0.015 | 10 | 0.7109 | 1.0000 |
| gap != 0 within innate-level-K2 | +0.754 | 10 | 0.0020 | 0.0195 |
| gap != 0 within innate-level-K6-constH | +0.792 | 10 | 0.0039 | 0.0195 |

Brier score of the pre-registered prior on the primary verdict: 1.010 (0 = certain and right; a uniform prior over the four categories scores 0.75; 2 = certain and wrong).
