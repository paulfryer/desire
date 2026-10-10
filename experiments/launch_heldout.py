"""
Run training jobs for the held-out-need experiment through a process pool (resumable: skips
runs whose final model exists). Every launch is appended to outputs/v3/COMMANDS.log.

    python experiments/launch_heldout.py main --steps 30000000 --parallel 9
    python experiments/launch_heldout.py pilot --steps 10000000 --parallel 6

`main` runs every arm with its pre-registered seeds (ARMS / SEEDS in train_heldout.py).
`pilot` runs seeds 100-101 of the three architectures into outputs/v3/pilot.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.train_heldout import ARMS, SEEDS  # noqa: E402

PILOT_ARMS = ["innate-level-K6", "learned-level-K6", "mlp-K6"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["main", "pilot"])
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--parallel", type=int, default=9)
    parser.add_argument("--learning-rate", nargs="*", default=[],
                        help="architecture=lr pairs, only as recorded in an amendment (§9)")
    args = parser.parse_args()
    lr = dict(pair.split("=") for pair in args.learning_rate)

    if args.what == "main":
        out = ROOT / "outputs/v3/runs"
        jobs = [(arm, s) for arm in ARMS for s in SEEDS[arm]]
    else:
        out = ROOT / "outputs/v3/pilot"
        jobs = [(arm, s) for arm in PILOT_ARMS for s in (100, 101)]
    jobs = [(a, s) for a, s in jobs if not (out / a / f"seed_{s}" / "final.zip").exists()]
    jobs.sort(key=lambda job: ARMS[job[0]][0] != "attn")       # slow (attention) jobs first
    logs = ROOT / "outputs/v3/logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "outputs/v3/COMMANDS.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} launch_heldout.py {' '.join(sys.argv[1:])} "
                f"({len(jobs)} jobs)\n")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore"}
    running = []
    print(f"{len(jobs)} jobs", flush=True)
    while jobs or running:
        while jobs and len(running) < args.parallel:
            arm, seed = jobs.pop(0)
            cmd = [sys.executable, "experiments/train_heldout.py", "--arm", arm, "--seed", str(seed),
                   "--steps", str(args.steps), "--out", str(out)]
            if ARMS[arm][0] in lr:
                cmd += ["--learning-rate", lr[ARMS[arm][0]]]
            log = open(logs / f"{args.what}_{arm}_seed{seed}.log", "w")
            running.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log, env=env),
                            f"{arm} seed {seed}"))
        time.sleep(10)
        for proc, label in list(running):
            if proc.poll() is not None:
                running.remove((proc, label))
                print(f"{time.strftime('%H:%M:%S')} finished {label} (exit {proc.returncode}); "
                      f"{len(jobs)} queued, {len(running)} running", flush=True)


if __name__ == "__main__":
    main()
