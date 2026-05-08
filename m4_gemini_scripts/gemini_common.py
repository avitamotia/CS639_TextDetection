"""Shared helpers for Gemini-based dataset generation.

Provides:
- load_api_key(): read ./gemini-key.txt
- get_client(): returns a configured google.genai.Client
- RateLimiter: thread-safe token bucket sized for an RPM cap
- call_with_retry(): generation call with exponential backoff
- list_models(): print available models if a chosen one is rejected
"""

from __future__ import annotations

import os
import random
import threading
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = REPO_ROOT / "gemini-key.txt"
DEFAULT_MODEL = "gemini-3.1-flash-lite"


def load_api_key() -> str:
    if not KEY_FILE.exists():
        raise FileNotFoundError(f"Missing API key file: {KEY_FILE}")
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"{KEY_FILE} is empty")
    return key


def get_client():
    from google import genai  # type: ignore

    return genai.Client(api_key=load_api_key())


class RateLimiter:
    """Thread-safe token bucket; blocks until a slot is available."""

    def __init__(self, rpm: int):
        self.capacity = max(1, rpm)
        self.interval = 60.0 / self.capacity
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self.interval


def call_with_retry(
    client,
    prompt: str,
    *,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    rate_limiter: Optional[RateLimiter] = None,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
) -> str:
    """Make a generation call with exponential backoff on transient errors.

    Returns the generated text. Raises after max_retries failures.
    """
    from google.genai import types  # type: ignore

    config = None
    if system is not None:
        config = types.GenerateContentConfig(system_instruction=system)

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt, config=config
            )
            text = getattr(resp, "text", None)
            if not text:
                raise RuntimeError("empty response from model")
            return text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            transient = (
                "429" in msg
                or "rate" in msg
                or "quota" in msg
                or "503" in msg
                or "500" in msg
                or "deadline" in msg
                or "unavailable" in msg
                or "timeout" in msg
            )
            if not transient or attempt == max_retries - 1:
                raise
            backoff = initial_backoff * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(backoff)
    if last_exc:
        raise last_exc
    raise RuntimeError("call_with_retry exited without success or exception")


def list_models() -> None:
    client = get_client()
    print("Available models:")
    for m in client.models.list():
        name = getattr(m, "name", str(m))
        print(f"  {name}")


if __name__ == "__main__":
    list_models()
