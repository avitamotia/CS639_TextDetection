"""Sweep `n_gemini_train` and train an energy detector at each setting.

For each value of N, runs `train_classifier_energy.py` with:
  --dataset M4
  --n_gemini_train N
  --gemini_train_path .../gemini_train.jsonl
  --eval_full_test_path .../gemini-M4/full_test
  --name gemini-M4-energy-N{N}

Each run produces its own folder under runs/ with a config.yaml,
TensorBoard logs (val_auc, val_auc_full, ...), and a per-run
test_results_M4_energy.json containing both 'orig' and 'full' eval metrics.

Defaults match the project's M4 configuration; override anything from the CLI.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = REPO_ROOT / "ood-llm-detect-main"  # cwd for the trainer

DEFAULT_SWEEP = [0, 100, 500, 1000, 2500, 5000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", type=int, nargs="+", default=DEFAULT_SWEEP,
                        help="n_gemini_train values to sweep")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--m4_path", default="data/SemEval2024-M4/SubtaskA",
                        help="Path (relative to ood-llm-detect-main/) to the original M4 SubtaskA folder.")
    parser.add_argument("--gemini_train_path",
                        default="data/gemini-M4/_intermediate/gemini_train.jsonl",
                        help="Path (relative to ood-llm-detect-main/) to the gemini-only train jsonl.")
    parser.add_argument("--eval_full_test_path",
                        default="data/gemini-M4/full_test",
                        help="Path (relative to ood-llm-detect-main/) to the gemini-augmented test folder.")
    parser.add_argument("--pth_path", default="",
                        help="Optional pretrained encoder checkpoint to resume from.")
    parser.add_argument("--name_prefix", default="gemini-M4-energy")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the commands but don't execute.")
    args = parser.parse_args()

    if not PROJECT_DIR.exists():
        print(f"[error] expected project dir at {PROJECT_DIR}", file=sys.stderr)
        sys.exit(1)

    rc_total = 0
    for n in args.values:
        run_name = f"{args.name_prefix}-N{n}"
        cmd = [
            sys.executable, "train_classifier_energy.py",
            "--device_num", "1",
            "--per_gpu_batch_size", str(args.batch_size),
            "--per_gpu_eval_batch_size", str(args.eval_batch_size),
            "--max_length", str(args.max_length),
            "--total_epoch", str(args.epochs),
            "--lr", "2e-5",
            "--warmup_steps", "1000",
            "--method", "energy",
            # 10 classes after adding gemini-3.1-flash-lite to M4_model_set (indices 0..9).
            "--classifier_dim", "10",
            "--model_name", "princeton-nlp/unsup-simcse-roberta-base",
            "--dataset", "M4",
            "--path", args.m4_path,
            "--name", run_name,
            "--freeze_embedding_layer",
            "--database_name", "monolingual_train",
            "--test_dataset_name", "monolingual_test",
            "--num_workers", "0",
            "--precision", "16-mixed",
            "--n_gemini_train", str(n),
            "--gemini_train_path", args.gemini_train_path,
            "--eval_full_test_path", args.eval_full_test_path,
        ]
        if args.pth_path:
            cmd += ["--resum", "True", "--pth_path", args.pth_path]

        print("\n" + "=" * 60)
        print(f"[sweep] n_gemini_train = {n}   name = {run_name}")
        print("=" * 60)
        print(" ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
        if result.returncode != 0:
            print(f"[sweep] run N={n} exited with code {result.returncode}", file=sys.stderr)
            rc_total += 1
            # don't abort the sweep on one failure; continue with the next N
    sys.exit(0 if rc_total == 0 else 1)


if __name__ == "__main__":
    main()
