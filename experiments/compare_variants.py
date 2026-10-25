"""Compare PPO-attention training variants on validation and held-out seeds.

Each variant is a run name under ``runs/`` with one directory per training seed. Every
seed's ``best.pt`` is scored on the 100 validation seeds (nominal scenario; used to choose
between variants) and on the 100 held-out seeds of every scenario in ``results/scenarios.json``
(used for reporting only). Writes ``results/variants.json`` and ``results/variants.md``.

    python experiments/compare_variants.py ppo_attn_smdp abl_no_smdp ppo_attn_steady ppo_attn_big
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_all import NOMINAL, SCENARIOS, compact, eval_model, seed_dirs  # noqa: E402

from agents.common import EVAL_SEEDS, VAL_SEEDS  # noqa: E402

COLUMNS = (
    ("task_return", "Task return"),
    ("revenue_captured_pct", "Revenue %"),
    ("on_time_pct", "On-time %"),
    ("express_on_time_pct", "Express on-time %"),
    ("travel_ticks_per_order", "Travel / order"),
)


def ci_cell(values) -> str:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return f"{values.mean():.1f}"
    return f"{values.mean():.1f} ± {1.96 * values.std(ddof=1) / np.sqrt(len(values)):.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--runs", default="runs")
    args = parser.parse_args()
    out: dict = {}
    for name in args.variants:
        dirs = seed_dirs(ROOT / args.runs, name)
        validation = [compact(eval_model(d / "best.pt", NOMINAL, VAL_SEEDS))["summary"]["task_return"] for d in dirs]
        test = {
            s: [compact(eval_model(d / "best.pt", NOMINAL.replace(**c), EVAL_SEEDS)) for d in dirs] for s, c in SCENARIOS.items()
        }
        out[name] = {"seeds": [d.name for d in dirs], "validation_task_return": validation, "scenarios": test}
        print(f"  {name}: {len(dirs)} seeds, validation task return {np.mean(validation):.1f}", flush=True)
    results = ROOT / "results"
    (results / "variants.json").write_text(json.dumps(out, default=float))

    lines = ["# PPO-attention training variants", "",
             "Validation: mean task return of each seed's `best.pt` on the 100 validation seeds (nominal), "
             "used to pick the default. Scenario tables: the 100 held-out seeds. ± is a 95% CI over training seeds.", "",
             "| Variant | seeds | Validation task return |", "|---|---|---|"]  # fmt: skip
    lines += [f"| {n} | {len(v['seeds'])} | {ci_cell(v['validation_task_return'])} |" for n, v in out.items()]
    for scenario in SCENARIOS:
        lines += ["", f"## {scenario}", "", "| Variant | " + " | ".join(h for _, h in COLUMNS) + " |",
                  "|---|" + "---|" * len(COLUMNS)]  # fmt: skip
        for name, v in out.items():
            runs = v["scenarios"][scenario]
            lines.append(f"| {name} | " + " | ".join(ci_cell([r["summary"][k] for r in runs]) for k, _ in COLUMNS) + " |")
    (results / "variants.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
