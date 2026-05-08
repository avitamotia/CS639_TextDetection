"""Turn prompts into Gemini-generated samples for one M4 split.

Reads:  ood-llm-detect-main/data/gemini-M4/_intermediate/<split>_prompts.json
Writes: ood-llm-detect-main/data/gemini-M4/_intermediate/gemini_<split>.jsonl
        + <split>_done.txt (resumable progress sidecar)

Records have the raw M4 schema (so the existing M4_utils.load_M4 will load
them as-is and apply its own label flip):
  train/dev: {"text", "label": 1, "model": "gemini-3.1-flash-lite", "source": "wikihow", "id"}
  test:      {"text", "label": 1, "model": "gemini-3.1-flash-lite", "domain": "outfox", "id"}
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gemini_common import (
    DEFAULT_MODEL,
    RateLimiter,
    call_with_retry,
    get_client,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INT_DIR = REPO_ROOT / "ood-llm-detect-main" / "data" / "gemini-M4" / "_intermediate"

WIKIHOW_SYSTEM = (
    "You write articles in the style of wikiHow. Given a 'How to ...' title, "
    "produce a complete how-to article between 500 and 1500 words. "
    "Use a short introductory paragraph, then numbered steps with a bold "
    "step heading followed by a one-paragraph explanation. Use a friendly, "
    "instructional voice. End with a brief closing paragraph. "
    "Output ONLY the article body. Do NOT include the title line. "
    "Do NOT include markdown like '#' or '**'. Plain prose only."
)

OUTFOX_SYSTEM = (
    "You are a high-school or first-year college student writing a "
    "well-organized argumentative or expository essay in response to the "
    "given prompt. Aim for 400-800 words across 4-6 paragraphs: "
    "introduction with a clear thesis, body paragraphs each with a topic "
    "sentence and supporting evidence/reasoning, and a conclusion. "
    "Use a natural student voice (clear but not overly polished). "
    "Output ONLY the essay text. Do NOT restate the prompt. "
    "Do NOT include a title or any markdown formatting."
)

MIN_WORDS = 150


def split_meta(split: str) -> tuple[str, str, str]:
    """Returns (system_prompt, schema_extra_field_name, schema_extra_field_value)."""
    if split in ("train", "dev"):
        return WIKIHOW_SYSTEM, "source", "wikihow"
    if split == "test":
        return OUTFOX_SYSTEM, "domain", "outfox"
    raise ValueError(f"unknown split: {split}")


def load_prompts(split: str) -> list[str]:
    p = INT_DIR / f"{split}_prompts.json"
    if not p.exists():
        raise FileNotFoundError(f"prompt file missing: {p} (run generate_prompts.py first)")
    return json.loads(p.read_text(encoding="utf-8"))


def load_done(done_path: Path) -> set[int]:
    if not done_path.exists():
        return set()
    out: set[int] = set()
    for line in done_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.isdigit():
            out.add(int(line))
    return out


def generate_one(client, rl, system: str, prompt: str, model: str) -> str:
    """Single attempt with one short-output retry."""
    user_msg = f"Title/prompt: {prompt}\n\nWrite the full text now."
    text = call_with_retry(client, user_msg, system=system, model=model, rate_limiter=rl)
    if len(text.split()) < MIN_WORDS:
        # one retry with stronger length nudge
        user_msg2 = (
            f"Title/prompt: {prompt}\n\nWrite the full text now. "
            f"It MUST be at least {MIN_WORDS} words."
        )
        text = call_with_retry(client, user_msg2, system=system, model=model, rate_limiter=rl)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rpm", type=int, default=30)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None,
                        help="(debug) only process the first N prompts")
    args = parser.parse_args()

    INT_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = INT_DIR / f"gemini_{args.split}.jsonl"
    done_path = INT_DIR / f"{args.split}_done.txt"

    system, extra_key, extra_val = split_meta(args.split)
    prompts = load_prompts(args.split)
    if args.limit:
        prompts = prompts[: args.limit]
    done = load_done(done_path)
    todo = [(i, p) for i, p in enumerate(prompts) if i not in done]
    print(f"split={args.split}  prompts={len(prompts)}  already_done={len(done)}  "
          f"todo={len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    client = get_client()
    rl = RateLimiter(rpm=args.rpm)

    write_lock = threading.Lock()

    def _process(idx: int, prompt: str) -> tuple[int, str | None, str | None]:
        try:
            text = generate_one(client, rl, system, prompt, args.model)
            return idx, text, None
        except Exception as exc:  # noqa: BLE001
            return idx, None, str(exc)

    n_ok = 0
    n_fail = 0
    with out_jsonl.open("a", encoding="utf-8") as fout, \
         done_path.open("a", encoding="utf-8") as fdone, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process, i, p): i for i, p in todo}
        for fut in as_completed(futs):
            idx, text, err = fut.result()
            if err is not None or not text:
                n_fail += 1
                if n_fail <= 10 or n_fail % 25 == 0:
                    print(f"  [fail idx={idx}] {err}", file=sys.stderr)
                continue
            rec = {
                "text": text,
                "label": 1,
                "model": args.model,
                extra_key: extra_val,
                "id": idx,
            }
            line = json.dumps(rec, ensure_ascii=False)
            with write_lock:
                fout.write(line + "\n")
                fout.flush()
                fdone.write(f"{idx}\n")
                fdone.flush()
            n_ok += 1
            if n_ok % 25 == 0 or n_ok == len(todo):
                print(f"  ok={n_ok}/{len(todo)}  fail={n_fail}")
    print(f"\nDone. ok={n_ok}  fail={n_fail}  -> {out_jsonl}")
    if n_fail:
        print(f"Re-run the same command to retry the {n_fail} failed prompts.")


if __name__ == "__main__":
    main()
