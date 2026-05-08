"""Generate disjoint prompt pools for train/dev/test gemini-M4 splits.

- wikiHow-style "How to ..." titles for train (5000) + dev (500)
- outfox-style argumentative student-essay prompts for test (1000)

Outputs JSON arrays of strings to:
  ood-llm-detect-main/data/gemini-M4/_intermediate/{train,dev,test}_prompts.json

Idempotent: if an output file already has >= the required count, it's skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable

from gemini_common import (
    DEFAULT_MODEL,
    RateLimiter,
    call_with_retry,
    get_client,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "ood-llm-detect-main" / "data" / "gemini-M4" / "_intermediate"

TRAIN_TARGET = 5000
DEV_TARGET = 500
TEST_TARGET = 1000
WIKIHOW_OVER = 500  # generate this many extra to absorb dedup loss
TEST_OVER = 200

WIKIHOW_CATEGORIES = [
    "cooking & baking", "fitness & exercise", "video games", "gardening",
    "productivity & time management", "consumer electronics", "arts & crafts",
    "personal finance", "travel", "parenting & childcare", "software & coding",
    "pets & animal care", "home DIY & repair", "beauty & grooming",
    "automotive maintenance", "outdoor recreation", "music & instruments",
    "writing & communication", "studying & academics", "social & relationships",
    "health & wellness", "career & job search", "fashion & clothing",
    "photography & video", "languages & translation", "science experiments",
    "smart home", "cleaning & organization", "wedding & event planning",
    "religious & spiritual practices", "vehicle driving", "first aid & safety",
    "shopping & deals", "moving & housing", "interview preparation",
]

OUTFOX_TOPIC_AREAS = [
    "school policy and education reform", "use of phones and social media by teens",
    "environmental responsibility and climate action", "online learning vs. in-person",
    "standardized testing", "school dress codes", "sports and academic balance",
    "censorship and free speech in schools", "homework load",
    "gun control and school safety", "voting age and youth civic engagement",
    "AI in classrooms", "screen time and mental health",
    "mandatory community service", "public transportation in cities",
    "year-round school", "single-use plastics", "minimum wage policy",
    "video games and behavior", "junk food advertising to children",
]

PROMPT_BATCH_SIZE = 50


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".?!")


def parse_lines(text: str) -> list[str]:
    """Extract one item per line, stripping numbering / bullets / quotes."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip("\"'`")
        if not line:
            continue
        # strip leading "1.", "1)", "- ", "* "
        line = re.sub(r"^\s*(?:\d+[\.\):]|\-|\*)\s*", "", line).strip()
        line = line.strip("\"'`")
        if len(line) < 3:
            continue
        out.append(line)
    return out


def generate_wikihow_titles(client, rl: RateLimiter, target: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    rng = random.Random(42)
    rounds = 0
    max_rounds = 200
    while len(out) < target and rounds < max_rounds:
        rounds += 1
        category = rng.choice(WIKIHOW_CATEGORIES)
        prompt = (
            f"Generate {PROMPT_BATCH_SIZE} unique and specific wikiHow-style "
            f"article titles in the category: {category}. "
            "Each title must start with 'How to ' and describe a concrete, "
            "actionable task. Vary difficulty, sub-topic, and audience. "
            "Avoid duplicates and overly generic titles. "
            "Output: one title per line, no numbering, no extra commentary."
        )
        try:
            text = call_with_retry(client, prompt, rate_limiter=rl)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] wikihow batch failed ({category}): {exc}", file=sys.stderr)
            continue
        added_this_round = 0
        for title in parse_lines(text):
            if not title.lower().startswith("how to"):
                continue
            n = normalize(title)
            if n in seen:
                continue
            seen.add(n)
            out.append(title)
            added_this_round += 1
        print(f"  round {rounds:>3} {category:<35}  +{added_this_round:>2}  "
              f"total={len(out)}/{target}")
    return out


def generate_outfox_prompts(client, rl: RateLimiter, target: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    rng = random.Random(43)
    rounds = 0
    max_rounds = 100
    while len(out) < target and rounds < max_rounds:
        rounds += 1
        topic = rng.choice(OUTFOX_TOPIC_AREAS)
        prompt = (
            f"Generate {PROMPT_BATCH_SIZE} unique student-essay writing prompts "
            f"in the topic area: {topic}. "
            "Each prompt should be 1-3 sentences, suitable for a high-school "
            "or first-year-college argumentative or expository essay. "
            "Vary the angle (pro/con, comparison, policy proposal, personal "
            "reflection grounded in argument). "
            "Output: one prompt per line, no numbering, no extra commentary."
        )
        try:
            text = call_with_retry(client, prompt, rate_limiter=rl)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] outfox batch failed ({topic}): {exc}", file=sys.stderr)
            continue
        added = 0
        for line in parse_lines(text):
            n = normalize(line)
            if n in seen or len(n) < 20:
                continue
            seen.add(n)
            out.append(line)
            added += 1
        print(f"  round {rounds:>3} {topic:<45}  +{added:>2}  total={len(out)}/{target}")
    return out


def has_enough(path: Path, target: int) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data, list) and len(data) >= target
    except Exception:  # noqa: BLE001
        return False


def write_json(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {len(items):>5} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpm", type=int, default=30, help="model RPM cap")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train_prompts.json"
    dev_path = OUT_DIR / "dev_prompts.json"
    test_path = OUT_DIR / "test_prompts.json"

    client = get_client()
    rl = RateLimiter(rpm=args.rpm)

    # --- wikiHow pool (train + dev) ---
    if has_enough(train_path, TRAIN_TARGET) and has_enough(dev_path, DEV_TARGET):
        print(f"train_prompts.json and dev_prompts.json already populated; skipping")
    else:
        wikihow_target = TRAIN_TARGET + DEV_TARGET + WIKIHOW_OVER
        print(f"\n=== Generating wikiHow titles (target {wikihow_target}) ===")
        titles = generate_wikihow_titles(client, rl, wikihow_target)
        if len(titles) < TRAIN_TARGET + DEV_TARGET:
            print(f"[error] only got {len(titles)} unique titles; need "
                  f"{TRAIN_TARGET + DEV_TARGET}", file=sys.stderr)
            sys.exit(1)
        random.Random(42).shuffle(titles)
        write_json(train_path, titles[:TRAIN_TARGET])
        write_json(dev_path, titles[TRAIN_TARGET:TRAIN_TARGET + DEV_TARGET])

    # --- outfox pool (test) ---
    if has_enough(test_path, TEST_TARGET):
        print(f"test_prompts.json already populated; skipping")
    else:
        outfox_target = TEST_TARGET + TEST_OVER
        print(f"\n=== Generating outfox-style prompts (target {outfox_target}) ===")
        prompts = generate_outfox_prompts(client, rl, outfox_target)
        if len(prompts) < TEST_TARGET:
            print(f"[error] only got {len(prompts)} unique prompts; need "
                  f"{TEST_TARGET}", file=sys.stderr)
            sys.exit(1)
        random.Random(44).shuffle(prompts)
        write_json(test_path, prompts[:TEST_TARGET])

    # --- disjointness check ---
    sets = {
        "train": {normalize(s) for s in json.loads(train_path.read_text(encoding="utf-8"))},
        "dev":   {normalize(s) for s in json.loads(dev_path.read_text(encoding="utf-8"))},
        "test":  {normalize(s) for s in json.loads(test_path.read_text(encoding="utf-8"))},
    }
    for a in ("train", "dev", "test"):
        for b in ("train", "dev", "test"):
            if a >= b:
                continue
            inter = sets[a] & sets[b]
            print(f"  {a} ∩ {b}: {len(inter)} shared")
            if inter:
                print(f"    e.g. {next(iter(inter))!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
