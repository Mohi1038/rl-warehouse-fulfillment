#!/usr/bin/env bash
# Reproduce every result: train all algorithms x seeds, ablations, multi-robot and tiny-env runs,
# publish checkpoints to models/, evaluate on held-out seeds and regenerate figures + tables.
#   bash experiments/run_all.sh            # full study (resumable)
#   bash experiments/run_all.sh --quick    # tiny budgets, smoke test of the whole pipeline
#   JOBS=8 bash experiments/run_all.sh     # more runs in parallel
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
if [[ "${1:-}" == "--quick" ]]; then
  exec "$PY" experiments/run_all.py --quick --jobs "${JOBS:-4}" --seeds 0 1 --ablation-seeds 0 --multi-seeds 0 --out runs_quick
fi
exec "$PY" experiments/run_all.py --jobs "${JOBS:-5}" "$@"
