"""
Run the main model-free training jobs (agents x seeds) through a small process pool, then the
assays on each finished run. Skips jobs whose final model already exists, so it can be re-run
after an interruption.

    python experiments/launch_model_free.py --seeds 0-9 --parallel 8 --steps 10000000 \
        --features split --norm-reward --n-envs 64 --n-steps 256 --batch-size 4096

Unrecognised flags are passed through to train_model_free.py.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_planners import parse_seeds  # noqa: E402
from experiments.train_model_free import AGENTS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", default=sorted(AGENTS))
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--steps", type=int, nargs="+", required=True,
                        help="training steps: one value, or one per agent in --agents order")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("outputs/v2/model_free"))
    args, train_flags = parser.parse_known_args()
    steps = dict(zip(args.agents, args.steps * len(args.agents) if len(args.steps) == 1
                     else args.steps, strict=True))

    jobs = [(a, s) for s in parse_seeds(args.seeds) for a in args.agents
            if not (args.out / a / f"seed_{s}" / "final.zip").exists()]
    # Recurrent jobs are slowest: start them first so they don't straggle at the end.
    jobs.sort(key=lambda job: AGENTS[job[0]][0] != "rppo")
    log_dir = ROOT / "outputs/v2/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore"}
    running: list[tuple[subprocess.Popen, str]] = []
    print(f"{len(jobs)} jobs", flush=True)
    while jobs or running:
        while jobs and len(running) < args.parallel:
            agent, seed = jobs.pop(0)
            cmd = [sys.executable, "experiments/train_model_free.py", "--agent", agent,
                   "--seed", str(seed), "--steps", str(steps[agent]), "--out", str(args.out),
                   *train_flags]
            log = open(log_dir / f"{agent}_seed{seed}.log", "w")
            running.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log, env=env),
                            f"{agent} seed {seed}"))
        time.sleep(10)
        for proc, label in list(running):
            if proc.poll() is not None:
                running.remove((proc, label))
                print(f"{time.strftime('%H:%M:%S')} finished {label} (exit {proc.returncode}); "
                      f"{len(jobs)} queued, {len(running)} running", flush=True)

    subprocess.run([sys.executable, "experiments/run_model_free_assays.py", "--runs",
                    str(args.out), "--agents", *args.agents], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
