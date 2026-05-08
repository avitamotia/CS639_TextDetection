"""Build an augmented test set: original M4 test + gemini test samples appended.

Reads:
  ood-llm-detect-main/data/SemEval2024-M4/SubtaskA/subtaskA_test_monolingual.jsonl
  ood-llm-detect-main/data/gemini-M4/_intermediate/gemini_test.jsonl

Writes:
  ood-llm-detect-main/data/gemini-M4/full_test/subtaskA_test_monolingual.jsonl

Original records pass through unchanged. Gemini records are appended at
the end with `id` continued from max(original_id) + 1, so every record in
the output has a unique id and the original block is byte-equivalent to
the source file.

The output directory matches the layout expected by utils/M4_utils.py
load_M4(folder) — point it at .../gemini-M4/full_test to load this set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIG_TEST = REPO_ROOT / "ood-llm-detect-main" / "data" / "SemEval2024-M4" / "SubtaskA" / "subtaskA_test_monolingual.jsonl"
GEMINI_TEST = REPO_ROOT / "ood-llm-detect-main" / "data" / "gemini-M4" / "_intermediate" / "gemini_test.jsonl"
OUT_PATH = REPO_ROOT / "ood-llm-detect-main" / "data" / "gemini-M4" / "full_test" / "subtaskA_test_monolingual.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", default=str(ORIG_TEST))
    parser.add_argument("--gemini", default=str(GEMINI_TEST))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    orig_path = Path(args.orig)
    gemini_path = Path(args.gemini)
    out_path = Path(args.out)

    if not orig_path.exists():
        print(f"[error] missing original test file: {orig_path}", file=sys.stderr)
        sys.exit(1)
    if not gemini_path.exists():
        print(f"[error] missing gemini test file: {gemini_path} "
              f"(run generate_samples.py --split test first)", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_orig = 0
    max_orig_id = -1
    with orig_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            fout.write(stripped + "\n")
            n_orig += 1
            try:
                rec_id = json.loads(stripped).get("id")
                if isinstance(rec_id, int) and rec_id > max_orig_id:
                    max_orig_id = rec_id
            except json.JSONDecodeError:
                pass

        next_id = max_orig_id + 1
        n_gemini = 0
        for line in gemini_path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["id"] = next_id
            next_id += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_gemini += 1

    print(f"original: {n_orig} records (max id={max_orig_id})")
    print(f"appended: {n_gemini} gemini records (ids {max_orig_id + 1}..{next_id - 1})")
    print(f"total:    {n_orig + n_gemini} -> {out_path}")


if __name__ == "__main__":
    main()
