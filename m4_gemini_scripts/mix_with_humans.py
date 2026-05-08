"""Mix gemini-generated samples with original M4 human samples.

For each split, samples N humans from the corresponding original M4 file
(preserving the original machine:human ratio) and concatenates them with
the gemini samples already produced. Writes the final dataset to
ood-llm-detect-main/data/gemini-M4/SubtaskA/.

Output schema (raw M4 format; M4_utils.load_M4 will flip labels at load time):
- train/dev: text, label, model, source, id
- test:      text, label, model, domain, id
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIG_DIR = REPO_ROOT / "ood-llm-detect-main" / "data" / "SemEval2024-M4" / "SubtaskA"
INT_DIR = REPO_ROOT / "ood-llm-detect-main" / "data" / "gemini-M4" / "_intermediate"
OUT_DIR = REPO_ROOT / "ood-llm-detect-main" / "data" / "gemini-M4" / "SubtaskA"

SPLITS = ["train", "dev", "test"]
SEED = 42


def split_meta(split: str) -> tuple[str, str, Path, Path]:
    """(extra_field_name, extra_field_default, original_path, output_path)."""
    orig = ORIG_DIR / f"subtaskA_{split}_monolingual.jsonl"
    out = OUT_DIR / f"subtaskA_{split}_monolingual.jsonl"
    if split == "test":
        return "domain", "outfox", orig, out
    return "source", "wikihow", orig, out


def load_gemini(split: str) -> list[dict]:
    p = INT_DIR / f"gemini_{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}; run generate_samples.py --split {split}")
    out: list[dict] = []
    for line in p.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def stream_humans_and_count_machines(orig_path: Path, extra_key: str) -> tuple[list[dict], int]:
    humans: list[dict] = []
    n_machine = 0
    for line in orig_path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # raw labels: 0 = human, !=0 = machine
        if rec.get("label") == 0:
            human_rec = {
                "text": rec["text"],
                "label": 0,
                "model": rec.get("model", "human"),
                extra_key: rec.get(extra_key, "wikihow" if extra_key == "source" else "outfox"),
                "id": rec.get("id"),
            }
            humans.append(human_rec)
        else:
            n_machine += 1
    return humans, n_machine


def process_split(split: str, rng: random.Random) -> dict:
    extra_key, extra_default, orig_path, out_path = split_meta(split)
    if not orig_path.exists():
        raise FileNotFoundError(f"original split missing: {orig_path}")

    print(f"\n[{split}] reading {orig_path.name} ...")
    humans, n_machine_orig = stream_humans_and_count_machines(orig_path, extra_key)
    n_human_orig = len(humans)

    gemini = load_gemini(split)
    n_gemini = len(gemini)

    if n_machine_orig == 0:
        n_humans_keep = n_human_orig
    else:
        n_humans_keep = round(n_gemini * n_human_orig / n_machine_orig)
    n_humans_keep = min(n_humans_keep, n_human_orig)

    sampled_humans = rng.sample(humans, n_humans_keep) if n_humans_keep < n_human_orig else humans

    # ensure all gemini records carry the right extra key, then concat
    fixed_gemini: list[dict] = []
    for g in gemini:
        rec = {
            "text": g["text"],
            "label": 1,
            "model": g.get("model", "gemini-3.1-flash-lite"),
            extra_key: g.get(extra_key, extra_default),
            "id": g.get("id"),
        }
        fixed_gemini.append(rec)

    combined = fixed_gemini + sampled_humans
    rng.shuffle(combined)
    for new_id, rec in enumerate(combined):
        rec["id"] = new_id

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in combined:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  orig humans={n_human_orig}  orig machines={n_machine_orig}  "
          f"M:H={n_machine_orig / n_human_orig:.3f}" if n_human_orig else "")
    print(f"  gemini={n_gemini}  humans_kept={len(sampled_humans)}  "
          f"total={len(combined)}  -> {out_path}")
    return {
        "split": split,
        "n_gemini": n_gemini,
        "n_human": len(sampled_humans),
        "total": len(combined),
        "ratio": round(n_gemini / max(1, len(sampled_humans)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    args = parser.parse_args()

    rng = random.Random(SEED)
    summary = []
    for s in args.splits:
        try:
            summary.append(process_split(s, rng))
        except FileNotFoundError as exc:
            print(f"[skip {s}] {exc}", file=sys.stderr)

    if summary:
        print("\nSummary:")
        print(f"  {'split':<6} {'gemini':>7} {'human':>7} {'total':>7} {'M:H':>7}")
        for r in summary:
            print(f"  {r['split']:<6} {r['n_gemini']:>7} {r['n_human']:>7} "
                  f"{r['total']:>7} {r['ratio']:>7}")


if __name__ == "__main__":
    main()
