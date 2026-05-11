"""GSM8K (Cobbe et al., 2021) loader for the NatComms revision.

GSM8K contains 8 500 grade-school math word problems (1 319 in the test
split). Each entry has::

    {"question": "...", "answer": "...reasoning... #### <final number>"}

This loader follows the same priority order as ``humaneval_loader``:
local cache → HuggingFace ``datasets`` → direct download.

Outputs from this module are pure data — no API calls.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_CACHE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "inputs"
    / "revision"
    / "gsm8k_test.jsonl"
)

# Official mirror by OpenAI (test split).
GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)


def _load_from_jsonl(path: Path) -> list:
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _try_datasets() -> Optional[list]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        ds = load_dataset("gsm8k", "main", split="test")
    except Exception:
        return None
    return [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]


def _try_download(url: str = GSM8K_TEST_URL, timeout: int = 30) -> Optional[list]:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "paper-experiment-revision/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
    except Exception:
        return None
    out = []
    for line in data.splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def cache_jsonl(problems: list, path: Path = DEFAULT_CACHE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


def load_gsm8k(cache_path: Path = DEFAULT_CACHE,
               allow_download: bool = True) -> list:
    """Return the GSM8K test split (1 319 problems)."""
    if cache_path.exists():
        return _load_from_jsonl(cache_path)

    problems = _try_datasets()
    if problems is None and allow_download:
        problems = _try_download()
    if problems is None:
        raise FileNotFoundError(
            f"GSM8K not available. Place a JSONL at {cache_path} or "
            f"`pip install datasets` or allow network access for "
            f"{GSM8K_TEST_URL}."
        )

    cache_jsonl(problems, cache_path)
    return problems


def sample_problems(problems: list, n: int = 30, seed: int = 42) -> list:
    """Random uniform sample of ``n`` problems with deterministic ordering."""
    rng = random.Random(seed)
    sampled = list(problems)
    rng.shuffle(sampled)
    return sampled[:n]


GSM8K_PROMPT = (
    "Solve the following grade-school math problem. Reason step by step, "
    "then on the final line write '#### <answer>' where <answer> is a "
    "single integer (no units, no commas).\n\n"
    "Problem:\n{question}"
)


def format_prompt(problem: dict) -> str:
    """Compose the user-facing prompt for a GSM8K problem."""
    return GSM8K_PROMPT.format(question=problem["question"])


def attach_id(problems: list, prefix: str = "GSM8K") -> list:
    """Return a copy of ``problems`` with stable ``id`` fields injected."""
    out = []
    for i, p in enumerate(problems):
        q = dict(p)
        q.setdefault("id", f"{prefix}/{i:04d}")
        out.append(q)
    return out
