# Running Predicate Tests - Operations Guide

This guide explains how to run long-duration predicate tests with checkpoint support, GPU monitoring, and progress tracking. Use this when resuming work or starting a fresh session.

## Quick Reference

### Start a New Test Run
```bash
python experiments/run_tests.py \
    --model-dir outputs/training_YYYYMMDD_HHMMSS \
    --resume \
    --gpu-cooldown \
    --gpu-max-temp 80
```

### Resume After Crash/Interruption
```bash
python experiments/run_tests.py \
    --model-dir outputs/training_YYYYMMDD_HHMMSS \
    --output-dir outputs/tests_YYYYMMDD_HHMMSS \
    --resume \
    --gpu-cooldown
```

### Check Progress (while running)
```bash
cat outputs/tests_YYYYMMDD_HHMMSS/progress.json
```

---

## Overview

The test suite runs 6 predicate tests on 5 agents (30 total combinations). Each test runs multiple trials (default 10). Full runs take several hours.

**Tests:** frustration, means_end, extinction, anticipation, opportunity_cost, state_dependent

**Agents:** heuristic, rl_ppo, rl_dqn, rl_a2c, active_inference

---

## Key Features

### 1. Checkpoint Support (`--resume`)
- Saves progress after each (test, agent) completes to `test_checkpoint.json`
- On resume, skips already-completed work
- **Critical for long runs** - protects against crashes, power failures, etc.

### 2. GPU Temperature Monitoring (`--gpu-cooldown`)
- Monitors GPU temperature using pynvml
- Pauses execution if GPU exceeds threshold (default 80C)
- Waits for cooldown to 75C before resuming
- Helps prevent thermal throttling and crashes during long runs

### 3. Progress Reporting (`progress.json`)
- Updated before each agent test starts
- Shows: current test, current agent, % complete, elapsed time, estimated remaining time, GPU temp
- Can be read externally while tests are running

---

## File Structure

```
outputs/tests_YYYYMMDD_HHMMSS/
├── progress.json          # Live progress (read this to check status)
├── test_checkpoint.json   # Checkpoint data (for resume)
├── test_results.json      # Final results (created at end)
├── frustration/           # Per-test output directories
├── means_end/
├── extinction/
├── anticipation/
├── opportunity_cost/
└── state_dependent/
```

---

## Step-by-Step: Running Tests

### Step 1: Find the Training Output Directory
```bash
ls -la outputs/ | grep training
```
Look for the most recent `training_YYYYMMDD_HHMMSS` directory containing trained models.

### Step 2: Verify Models Exist
```bash
ls outputs/training_YYYYMMDD_HHMMSS/models/
```
Should contain: `rl_ppo.zip`, `rl_dqn.zip`, `rl_a2c.zip`

### Step 3: Start the Test Run
```bash
python experiments/run_tests.py \
    --model-dir outputs/training_YYYYMMDD_HHMMSS \
    --resume \
    --gpu-cooldown \
    --gpu-max-temp 80
```

This will:
- Create a new `outputs/tests_YYYYMMDD_HHMMSS/` directory
- Start running tests with checkpoint and GPU protection
- Print the output directory path at startup

### Step 4: Monitor Progress
In another terminal or periodically:
```bash
cat outputs/tests_YYYYMMDD_HHMMSS/progress.json
```

Example output:
```json
{
  "status": "running",
  "current_test": "means_end_decoupling",
  "current_agent": "rl_ppo",
  "progress": {
    "tests_completed": 1,
    "total_tests": 6,
    "total_agent_tests_completed": 7,
    "total_agent_tests": 30,
    "percent_complete": 23.3
  },
  "timing": {
    "elapsed_human": "1h 15m 32s",
    "estimated_remaining_human": "4h 7m"
  },
  "gpu": {
    "temperature_c": 72.0
  }
}
```

---

## Step-by-Step: Resuming After Crash

### Step 1: Find the Test Output Directory
```bash
ls -la outputs/ | grep tests
```

### Step 2: Check What Was Completed
```bash
cat outputs/tests_YYYYMMDD_HHMMSS/test_checkpoint.json | python -c "
import json, sys
data = json.load(sys.stdin)
for test, agents in data.get('completed', {}).items():
    print(f'{test}: {list(agents.keys())}')"
```

### Step 3: Resume the Run
```bash
python experiments/run_tests.py \
    --model-dir outputs/training_YYYYMMDD_HHMMSS \
    --output-dir outputs/tests_YYYYMMDD_HHMMSS \
    --resume \
    --gpu-cooldown
```

**Important:** You must specify `--output-dir` to point to the existing test directory with the checkpoint.

---

## CLI Options Reference

| Option | Description |
|--------|-------------|
| `--model-dir` | Directory with trained models (required) |
| `--output-dir` | Output directory (auto-generated if not specified) |
| `--resume` | Enable checkpoint resume (always use this) |
| `--gpu-cooldown` | Enable GPU temperature monitoring |
| `--gpu-max-temp` | Max GPU temp before cooldown (default: 80C) |
| `--agent` | Specific agent(s) to test (space-separated) |
| `--test` | Specific test(s) to run (space-separated) |
| `--n_trials` | Override number of trials per test |
| `--verbose` | Enable verbose output |

---

## Troubleshooting

### "No checkpoint found, starting fresh"
This is normal when starting a new run. If you expected to resume, check that `--output-dir` points to a directory containing `test_checkpoint.json`.

### GPU monitoring not working
Install pynvml: `pip install pynvml`

### Tests are very slow
- Reduce trials: `--n_trials 5`
- Test specific agents: `--agent heuristic rl_ppo`
- Test specific tests: `--test frustration means_end`

### Out of memory
- The active_inference agent uses significant memory
- Try running it separately: `--agent active_inference`

---

## For Claude Code Sessions

When starting a new Claude Code session and asked to run or resume tests:

1. **Check for existing test runs:** `ls outputs/ | grep tests`
2. **Check for training outputs:** `ls outputs/ | grep training`
3. **If resuming:** Use the existing `--output-dir` with `--resume`
4. **If starting fresh:** Just use `--model-dir` and `--resume` (new output dir auto-created)
5. **Always use:** `--resume --gpu-cooldown` for safety
6. **Monitor with:** `cat outputs/tests_*/progress.json`
