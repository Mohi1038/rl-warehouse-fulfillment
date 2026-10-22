"""Train every run of the study, publish checkpoints, then evaluate and plot.

    python experiments/run_all.py                    # full study (resumable: finished runs are skipped)
    python experiments/run_all.py --quick --jobs 4   # tiny budgets, for smoke-testing the pipeline
    python experiments/run_all.py --stage evaluate   # only re-run evaluation + figures

Runs land in ``runs/<name>/s<seed>``; the best seed per algorithm (by validation task
return, never by the held-out evaluation seeds) is copied to ``models/best_<algo>.pt``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN_ALGOS = ("ppo_attn", "ppo_mlp", "dqn")


def plan_runs(seeds: list[int], ablation_seeds: list[int], multi_seeds: list[int], quick: bool) -> list[dict]:
    runs = []
    for algo in MAIN_ALGOS:
        runs += [{"config": f"configs/{algo}.yaml", "name": algo, "seed": s, "set": []} for s in seeds]
    ablations = yaml.safe_load((ROOT / "experiments/ablations.yaml").read_text())
    for name, overrides in ablations.items():
        sets = [f"{k}={v}" for k, v in overrides.items()]
        runs += [{"config": "configs/ppo_attn.yaml", "name": f"abl_{name}", "seed": s, "set": sets} for s in ablation_seeds]
    for k in (1, 2, 3):
        runs += [{"config": f"configs/multi_k{k}.yaml", "name": f"multi_k{k}", "seed": s, "set": []} for s in multi_seeds]
    runs += [{"config": "configs/tiny_ppo.yaml", "name": "tiny_ppo", "seed": s, "set": []} for s in ablation_seeds]
    runs += [{"config": "configs/tiny_q.yaml", "name": "tiny_q", "seed": s, "set": []} for s in ablation_seeds]
    if quick:
        for run in runs:
            budget = "algo_config.episodes=3000" if run["name"] == "tiny_q" else "algo_config.total_decisions=16384"
            run["set"] = run["set"] + [budget, "eval_every=2", "eval_episodes=4"]
    for run in runs:
        run["set"] = run["set"] + [f"name={run['name']}"]
    return runs


def train_one(run: dict, out: Path, python: str) -> tuple[dict, float, int]:
    run_dir = out / run["name"] / f"s{run['seed']}"
    if (run_dir / "DONE").exists():
        return run, 0.0, 0
    log = ROOT / "logs" / f"{run['name']}_s{run['seed']}.log"
    log.parent.mkdir(exist_ok=True)
    cmd = [python, "train.py", "--config", run["config"], "--seed", str(run["seed"]), "--out", str(out), "--set", *run["set"]]
    start = time.time()
    with log.open("w") as handle:
        code = subprocess.call(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return run, time.time() - start, code


def best_validation(run_dir: Path) -> float:
    with (run_dir / "eval.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return max(float(r["task_return"]) for r in rows)


def publish(out: Path, seeds: list[int]) -> None:
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    for name in (*MAIN_ALGOS, "multi_k1", "multi_k2", "multi_k3", "tiny_ppo"):
        candidates = [out / name / f"s{s}" for s in seeds if (out / name / f"s{s}" / "best.pt").exists()]
        if not candidates:
            continue
        for run_dir in candidates:
            if name in MAIN_ALGOS:
                shutil.copy(run_dir / "best.pt", models / f"{name}_{run_dir.name}.pt")
        best = max(candidates, key=best_validation)
        shutil.copy(best / "best.pt", models / f"best_{name}.pt")
        print(f"published models/best_{name}.pt from {best} (validation task return {best_validation(best):.1f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ablation-seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--multi-seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--only", nargs="*", help="train only these run names")
    args = parser.parse_args()
    out = ROOT / args.out
    python = sys.executable

    if args.stage in ("all", "train"):
        runs = plan_runs(args.seeds, args.ablation_seeds, args.multi_seeds, args.quick)
        if args.only:
            runs = [r for r in runs if r["name"] in args.only]
        print(f"{len(runs)} training runs, {args.jobs} in parallel")
        failed = []
        with ThreadPoolExecutor(args.jobs) as pool:
            futures = [pool.submit(train_one, run, out, python) for run in runs]
            for future in as_completed(futures):
                run, seconds, code = future.result()
                status = "skipped (done)" if seconds == 0 and code == 0 else ("ok" if code == 0 else f"FAILED ({code})")
                print(f"  {run['name']:<18} seed {run['seed']}: {status} {seconds / 60:.1f} min", flush=True)
                if code:
                    failed.append(run)
        if failed:
            raise SystemExit(f"{len(failed)} runs failed; see logs/")
        publish(out, sorted(set(args.seeds) | set(args.multi_seeds) | set(args.ablation_seeds)))

    if args.stage in ("all", "evaluate"):
        episodes = ["--episodes", "10"] if args.quick else []
        subprocess.check_call([python, "experiments/evaluate_all.py", "--runs", str(out), *episodes], cwd=ROOT)
        subprocess.check_call([python, "plots/make_figures.py", "--runs", str(out)], cwd=ROOT)


if __name__ == "__main__":
    main()
