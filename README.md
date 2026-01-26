# Anticipatory Activation Distinguishes Active Inference from Reinforcement Learning Agents

This repository contains the code and data for reproducing the experiments in our paper investigating whether Active Inference agents exhibit "desire-like" behavioral signatures that distinguish them from standard Reinforcement Learning agents.

## Key Finding

**Anticipatory Activation (AA)** shows a clean dissociation between agent types:
- **Heuristic**: AA = 1.000
- **Active Inference**: AA = 0.974
- **RL (PPO)**: AA = 0.000

This pattern was statistically significant (p = 0.002) and 100% consistent across random seeds.

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
# 1. Train all agents
python experiments/train_agents.py --config experiments/configs/default.yaml

# 2. Run behavioral predicate tests
python experiments/run_tests.py --config experiments/configs/default.yaml

# 3. Analyze results
python experiments/analyze_results.py
```

### Run a quick test (faster, fewer trials):

```bash
python experiments/train_agents.py --config experiments/configs/quick_test.yaml
python experiments/run_tests.py --config experiments/configs/quick_test.yaml
```

## Checkpointing and Resuming Training

Training automatically saves periodic checkpoints to prevent losing progress if training is interrupted.

### Checkpoint Configuration

In your config YAML file:
```yaml
training:
  save_freq: 10000           # Save checkpoint every N timesteps
  checkpoint_keep_last: 5    # Keep only the last 5 checkpoints (older ones are deleted)
```

### Output Structure

```
outputs/training_XXXXXX/
  checkpoints/
    rl_ppo_10000_steps.zip
    rl_ppo_20000_steps.zip
    ...
  models/
    rl_ppo.zip              # Final model
```

### Resuming Training

**Resume from latest checkpoint:**
```bash
python experiments/train_agents.py \
    --config experiments/configs/default.yaml \
    --resume latest \
    --resume-dir outputs/training_20260124_105645
```

**Resume from specific checkpoint:**
```bash
python experiments/train_agents.py \
    --config experiments/configs/default.yaml \
    --resume outputs/training_XXXXX/checkpoints/rl_ppo_50000_steps.zip \
    --output-dir outputs/training_XXXXX
```

The training will automatically calculate remaining timesteps and continue from where it left off.

## Reproducing Paper Results

The paper results used `n=10` trials with robustness checks across 3 random seeds:

```bash
# Run publication-quality experiments
python experiments/train_agents.py --config experiments/configs/publication_focused.yaml
python experiments/run_tests.py --config experiments/configs/publication_focused.yaml

# Generate publication analysis
python experiments/generate_publication_analysis.py

# Run robustness check across multiple seeds
python experiments/robustness_check.py
```

Results will be saved to `outputs/`.

## Project Structure

```
desire/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── heuristic.py          # Rule-based baseline
│   │   ├── rl_agent.py           # PPO agent (stable-baselines3)
│   │   └── active_inference.py   # Active Inference agent (pymdp)
│   ├── environments/     # Grid-world foraging environment
│   ├── tests/            # Behavioral predicate tests (AA, FI, MED, etc.)
│   └── analysis/         # Metrics and visualization
├── experiments/          # Training and test scripts
│   └── configs/          # YAML configuration files
├── tests/                # Unit tests (pytest)
├── paper/                # LaTeX source for the paper
├── outputs/              # Experiment results (generated)
└── docs/                 # Research design documentation
```

## Behavioral Predicates

We test six behavioral predicates inspired by the psychology of desire:

| Predicate | Description | Key Result |
|-----------|-------------|------------|
| **AA** (Anticipatory Activation) | Internal state activation when approaching goals | **Significant** (p=0.002) |
| **FI** (Frustration Intensification) | Activity spike when paths are blocked | Not significant |
| **MED** (Means-End Decoupling) | Flexible re-routing to goals | Not significant |
| **ER** (Extinction Resistance) | Persistence when rewards cease | Not significant |
| **OCS** (Opportunity Cost Sensitivity) | Rational tradeoff decisions | Significant (p=0.011) |
| **SDV** (State-Dependent Valuation) | Valuation varies with internal state | Not significant |

## Running Tests

```bash
pytest tests/
```

## Citation

If you use this code, please cite:

```bibtex
@article{fryer2026anticipatory,
  title={Anticipatory Activation Distinguishes Active Inference from Reinforcement Learning Agents},
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
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) for PPO
- [Gymnasium](https://gymnasium.farama.org/) for the environment API
