# Retracted: "Desire-Likeness is Multi-Dimensional" (v1 paper draft)

*Retracted by the author on 2026-10-06; archived here on 2026-10-20. Kept so the record of how this project's thinking evolved stays intact. Nothing in this folder should be cited as a finding.*

## What this paper claimed

That five agent types (heuristic, PPO, DQN, A2C, active inference) could be compared on six behavioral "desire predicates" (anticipatory activation, state-dependent valuation, extinction resistance, frustration intensification, opportunity-cost sensitivity, means–end decoupling), and that "desire-likeness is multi-dimensional": DQN and active inference show anticipatory activation, A2C shows state-dependent valuation and frustration, PPO shows inverse anticipation.

## Why it is retracted

The predicates were never validated against agents known to have no desire. When that check was run (`docs/LEGACY_VALIDITY.md`, `experiments/legacy_validity_check.py`):

- **Three predicates are passed by agents with no desire by construction.** A sensor readout shows "true anticipation" as strongly as the paper's DQN. A one-line thermostat beats the paper's best agent on state-dependent valuation. Every control lands in the same extinction-resistance range as every agent in the paper.
- **Two predicates separated agents only because of scoring bugs.** Under the original code, every control agent was "frustrated" and a random-walking thermostat earned a perfect opportunity-cost score. The heuristic's reported OCS of 1.00 was entirely a classification bug.
- **One predicate was never measured.** Means–end decoupling always returned 0 because of a bug in goal detection.
- **The active-inference agent did not do inference.** Its belief update never applied the transition model, and action selection was driven by a small preference for central grid cells. It camped at the centre.
- **Pseudo-replication.** Fixed layouts and deterministic policies made the 10 "trials" identical for several agent–predicate pairs.

So the paper's central contrast (which frameworks show which signatures) has no evidential basis. The question it asked, whether active inference shows desire signatures that RL does not, was also set aside: the two frameworks are inter-translatable, and the behavioral phenomena that distinguish wanting from habit (devaluation, revaluation) require assays validated on known-answer agents first.

## What replaced it

- `docs/RESEARCH_DESIGN_v2.md`, `docs/RESULTS_v2.md`: outcome-devaluation and zero-shot-revaluation assays, validated on ground-truth controls, run on planners and model-free agents.
- `docs/HELDOUT_DESIGN.md`, `docs/RESULTS_HELDOUT.md`: the pre-registered held-out-need experiment.
- `docs/CLAIMS.md`: what the project is now prepared to claim, with confidence levels.

## Files here

- `main.tex`: the original source, with a retraction preface added at the top of the document body. Nothing else was changed.
- `main.pdf`: the original compiled PDF with a one-page retraction notice prepended (no LaTeX toolchain was available to rebuild it). The unmodified PDF is in git history before commit "Archive the retracted v1 paper" and at `main` on this repository.
- `generate_figures.py`, `figures/`: the original figure code and outputs. The script's paths assume it is run from the old `paper/` location.
