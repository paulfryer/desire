# Claude Code Project Context

This is a research codebase for testing desire-like behavior in AI agents using active inference and reinforcement learning.

## Project Structure

- `src/` - Core library (agents, environments, tests, analysis)
- `experiments/` - Training and test scripts
- `outputs/` - Training outputs and test results

## Key Operations

### Running Predicate Tests

**See `experiments/RUNNING_TESTS.md` for detailed instructions.**

Quick start:
```bash
# Find latest training output
ls outputs/ | grep training

# Run tests with checkpoint and GPU protection
python experiments/run_tests.py \
    --model-dir outputs/training_YYYYMMDD_HHMMSS \
    --resume \
    --gpu-cooldown

# Check progress while running
cat outputs/tests_*/progress.json
```

### Resuming Interrupted Tests

```bash
# Find existing test output directory
ls outputs/ | grep tests

# Resume with same output directory
python experiments/run_tests.py \
    --model-dir outputs/training_YYYYMMDD_HHMMSS \
    --output-dir outputs/tests_YYYYMMDD_HHMMSS \
    --resume \
    --gpu-cooldown
```

### Training Agents

```bash
python experiments/train_agents.py --resume --checkpoint-interval 50000
```

## Important Notes

- **Always use `--resume`** when running tests - protects against crashes
- **Always use `--gpu-cooldown`** for long runs - prevents overheating
- Tests take several hours (6 tests x 5 agents x 10 trials each)
- Check `progress.json` to monitor long-running tests
- Checkpoint saves after each (test, agent) combination completes

## Agents

- `heuristic` - Rule-based baseline (no model file needed)
- `rl_ppo` - PPO agent (stable-baselines3)
- `rl_dqn` - DQN agent (stable-baselines3)
- `rl_a2c` - A2C agent (stable-baselines3)
- `active_inference` - Active inference agent (pymdp)

## Tests

- `frustration` - Frustration intensification
- `means_end` - Means-end decoupling
- `extinction` - Extinction resistance
- `anticipation` - Anticipatory activation
- `opportunity_cost` - Opportunity cost sensitivity
- `state_dependent` - State-dependent valuation
