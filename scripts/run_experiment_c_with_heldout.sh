#!/usr/bin/env bash
# Same as fast run, plus stratified held-out evaluation on MAGE ``test`` (8k rows).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m experiments.objective_ablation.run_experiment_c \
  --fast \
  --out-dir runs/exp_c_fast_heldout \
  --report-split test \
  --report-max-samples 8000
