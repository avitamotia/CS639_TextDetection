#!/usr/bin/env bash
# Reproduce the "fast" Experiment C run (3k rows, 4 epochs, both objectives).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m experiments.objective_ablation.run_experiment_c --fast --out-dir runs/exp_c_fast
