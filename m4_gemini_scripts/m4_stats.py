"""Stream-count the original SemEval2024-M4 SubtaskA splits.

Reports per-split totals, per-model counts, the schema fields seen
(`source` vs `domain`), and the projected number of human samples we'd
keep when adding 5000/500/1000 Gemini samples while preserving the
original machine:human ratio.

Outputs:
- ood-llm-detect-main/data/gemini-M4/_intermediate/stats.json
- printed summary table
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
M4_DIR = REPO_ROOT / "ood-llm-detect-main" / "data" / "SemEval2024-M4" / "SubtaskA"
OUT_DIR = REPO_ROOT / "ood-llm-detect-main" / "data" / "gemini-M4" / "_intermediate"

SPLITS = {
    "train": M4_DIR / "subtaskA_train_monolingual.jsonl",
    "dev": M4_DIR / "subtaskA_dev_monolingual.jsonl",
    "test": M4_DIR / "subtaskA_test_monolingual.jsonl",
}

GEMINI_TARGETS = {"train": 5000, "dev": 500, "test": 1000}


def stream_stats(path: Path) -> dict:
    by_model: Counter[str] = Counter()
    by_label: Counter[int] = Counter()
    fields_seen: set[str] = set()
    total = 0
    bad_lines = 0
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        total += 1
        by_model[rec.get("model", "<missing>")] += 1
        by_label[rec.get("label", -1)] += 1
        fields_seen.update(rec.keys())
    return {
        "total": total,
        "bad_lines": bad_lines,
        "by_model": dict(by_model),
        "by_label": {str(k): v for k, v in by_label.items()},
        "fields": sorted(fields_seen),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for split, path in SPLITS.items():
        if not path.exists():
            print(f"[WARN] missing: {path}", file=sys.stderr)
            continue
        print(f"Scanning {split}: {path.name} ...")
        s = stream_stats(path)
        # raw labels: 0 = human, !=0 = machine
        n_human = s["by_label"].get("0", 0)
        n_machine = s["total"] - n_human
        gemini_target = GEMINI_TARGETS[split]
        if n_machine > 0:
            projected_humans = round(gemini_target * n_human / n_machine)
        else:
            projected_humans = 0
        s["n_human"] = n_human
        s["n_machine"] = n_machine
        s["machine_to_human_ratio"] = round(n_machine / n_human, 3) if n_human else None
        s["gemini_target"] = gemini_target
        s["projected_humans_to_sample"] = projected_humans
        report[split] = s

    out = OUT_DIR / "stats.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"{'split':<6} {'total':>8} {'human':>7} {'machine':>9} {'M:H':>7} "
          f"{'gemini':>7} {'humans*':>8}  fields")
    for split, s in report.items():
        print(f"{split:<6} {s['total']:>8} {s['n_human']:>7} {s['n_machine']:>9} "
              f"{str(s['machine_to_human_ratio']):>7} {s['gemini_target']:>7} "
              f"{s['projected_humans_to_sample']:>8}  {s['fields']}")
    print(f"\n* humans = round(gemini_target * H_orig / M_orig); preserves M4 ratio")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
