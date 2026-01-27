# Desire-Likeness is Multi-Dimensional: Comparing Behavioral Signatures Across Learning Frameworks

This repository contains the code and data for reproducing the experiments in our paper investigating whether Active Inference agents exhibit "desire-like" behavioral signatures that distinguish them from standard Reinforcement Learning agents.

## Key Findings

**Desire-likeness is multi-dimensional**—no single agent type dominates across all behavioral predicates:

| Agent | Anticipation (r) | SDV | FI | OCS |
|-------|------------------|-----|-----|-----|
| **DQN** | -0.65 (true) | 0.00 | 0.00 | 0.00 |
| **Active Inference** | -0.59 (true) | 0.40 | 0.05 | 0.91 |
| **A2C** | -0.20 (weak) | **0.73** | **0.88** | 0.50 |
| **PPO** | +0.63 (inverse) | 0.00 | 0.69 | 0.00 |
| **Heuristic** | 0.00 (none) | 0.00 | 0.02 | **1.00** |

Key insights:
- **DQN and Active Inference** both show true anticipatory activation (negative correlation)
- **PPO** shows *inverse* anticipation in shared features, but **true anticipation in policy head** (r=-0.39)
- **A2C** shows the highest state-dependent valuation and frustration response
- **Architecture matters more than framework** (RL vs Active Inference)
- **Robustness verified**: DQN's Q-values confirm hidden activation patterns (r=-0.85)

## Installation

Requires Python 3.10+

```bash
# Clone the repository
git clone https://github.com/paulfryer/desire.git
cd desire

# Install dependencies
pip install -e .

# For GPU support (recommended), install PyTorch with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Quick Start

### Run the full experiment pipeline:

```bash
# 1. Train all agents (heuristic, rl_ppo, rl_dqn, rl_a2c, active_inference)
python experiments/train_agents.py --output-dir outputs/training

# 2. Run behavioral predicate tests with checkpointing
python experiments/run_tests.py \
    --model-dir outputs/training \
    --output-dir outputs/tests \
    --resume \
    --gpu-cooldown

# 3. Analyze results and generate figures
python experiments/desire_profile_analysis.py
```

### Run a quick test (faster, fewer trials):

```bash
python experiments/train_agents.py --config experiments/configs/quick_test.yaml
python experiments/run_tests.py --config experiments/configs/quick_test.yaml
```

## Checkpointing and Resuming

Both training and testing support checkpointing to prevent losing progress.

### Training Checkpoints

```bash
# Resume training from latest checkpoint
python experiments/train_agents.py \
    --resume latest \
    --resume-dir outputs/training_XXXXXX
```

### Test Checkpoints

```bash
# Resume tests after crash (automatically detects checkpoint)
python experiments/run_tests.py \
    --model-dir outputs/training_XXXXXX \
    --output-dir outputs/tests_XXXXXX \
    --resume
```

### GPU Temperature Monitoring

For long test runs, enable GPU cooldown to prevent thermal throttling:

```bash
python experiments/run_tests.py \
    --model-dir outputs/training \
    --gpu-cooldown \
    --gpu-max-temp 80
```

## Project Structure

```
desire/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── heuristic.py          # Rule-based baseline
│   │   ├── rl_ppo.py             # PPO agent (stable-baselines3)
│   │   ├── rl_dqn.py             # DQN agent (stable-baselines3)
│   │   ├── rl_a2c.py             # A2C agent (stable-baselines3)
│   │   └── active_inference.py   # Active Inference agent (pymdp)
│   ├── environments/     # Grid-world foraging environment
│   ├── tests/            # Behavioral predicate tests
│   └── analysis/         # Metrics and visualization
├── experiments/          # Training and test scripts
│   ├── train_agents.py           # Train all agent types
│   ├── run_tests.py              # Run predicate tests with checkpointing
│   ├── desire_profile_analysis.py # Multi-metric analysis
│   ├── aa_robustness_analysis.py  # Q-value/value-head robustness check
│   └── configs/                  # YAML configuration files
├── paper/                # LaTeX source and figures
│   ├── main.tex                  # Paper source
│   ├── main.pdf                  # Compiled paper
│   └── figures/                  # Publication figures
├── outputs/              # Experiment results (generated)
└── docs/                 # Research design documentation
```

## Behavioral Predicates

We test six behavioral predicates inspired by the psychology of desire:

| Predicate | Description | Key Finding |
|-----------|-------------|-------------|
| **AA** (Anticipatory Activation) | Correlation between internal state and goal proximity | DQN & AI show true anticipation; PPO shows inverse |
| **SDV** (State-Dependent Valuation) | Behavior varies with energy level | A2C highest (0.73) |
| **FI** (Frustration Intensification) | Activity spike when paths blocked | A2C (0.88), PPO (0.69) highest |
| **OCS** (Opportunity Cost Sensitivity) | Rational tradeoff decisions | Heuristic & AI highest |
| **ER** (Extinction Resistance) | Persistence when rewards cease | All agents similar (~0.93) |
| **MED** (Means-End Decoupling) | Flexible re-routing to goals | Not measured in current run |

## Generating Paper Figures

```bash
# Generate all publication figures
cd paper
python generate_figures.py

# Compile the paper
pdflatex main.tex && pdflatex main.tex
```

## Running Unit Tests

```bash
pytest tests/
```

## Citation

If you use this code, please cite:

```bibtex
@article{fryer2026desire,
  title={Desire-Likeness is Multi-Dimensional: Comparing Behavioral Signatures Across Learning Frameworks},
  author={Fryer, Paul},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License

## Acknowledgments

This project uses:
- [pymdp](https://github.com/infer-actively/pymdp) for Active Inference
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) for RL agents (PPO, DQN, A2C)
- [Gymnasium](https://gymnasium.farama.org/) for the environment API
