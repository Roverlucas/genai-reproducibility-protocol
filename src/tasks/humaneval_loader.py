"""HumanEval (Chen et al., 2021) loader for the NatComms revision.

HumanEval is a 164-problem benchmark of hand-written Python programming
problems with unit tests. Each entry has the structure::

    {
        "task_id":  "HumanEval/0",
        "prompt":   "...function signature + docstring...",
        "canonical_solution": "...reference body...",
        "test":     "def check(candidate): ...",
        "entry_point": "function_name",
    }

This loader:
  1. Loads from a local cached JSONL if present (preferred — keeps the
     pipeline reproducible and avoids network dependence).
  2. Falls back to ``datasets.load_dataset('openai_humaneval')`` if the
     ``datasets`` package is installed.
  3. Falls back to a direct urllib download from the official GitHub repo.
  4. Stratified sampling by docstring length (proxy for difficulty)
     when ``stratify=True``.

No API calls are made. This module is pure data preparation.
"""

from __future__ import annotations

import gzip
import io
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
    / "humaneval.jsonl"
)

# Official mirror published by OpenAI (gz-compressed JSONL).
HUMANEVAL_GZ_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/"
    "HumanEval.jsonl.gz"
)


def _load_from_jsonl(path: Path) -> list:
    """Load a list of HumanEval problems from a local JSONL file."""
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _try_datasets() -> Optional[list]:
    """Optional path: HuggingFace ``datasets`` library."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        ds = load_dataset("openai_humaneval", split="test")
    except Exception:
        return None
    out = []
    for ex in ds:
        out.append(
            {
                "task_id": ex["task_id"],
                "prompt": ex["prompt"],
                "canonical_solution": ex.get("canonical_solution", ""),
                "test": ex["test"],
                "entry_point": ex["entry_point"],
            }
        )
    return out


def _try_download(url: str = HUMANEVAL_GZ_URL, timeout: int = 30) -> Optional[list]:
    """Direct download as a last resort."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "paper-experiment-revision/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception:
        return None
    try:
        decompressed = gzip.decompress(raw)
        problems = []
        for line in io.BytesIO(decompressed).read().decode("utf-8").splitlines():
            if line.strip():
                problems.append(json.loads(line))
        return problems
    except Exception:
        return None


def cache_jsonl(problems: list, path: Path = DEFAULT_CACHE) -> Path:
    """Persist the loaded set to a JSONL file for offline reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


def load_humaneval(cache_path: Path = DEFAULT_CACHE,
                   allow_download: bool = True) -> list:
    """Return the full HumanEval problem list (164 problems).

    Resolution order:
        1. Local JSONL cache (``cache_path``).
        2. HuggingFace ``datasets`` (cached on first call).
        3. Direct download from the GitHub mirror.
    """
    if cache_path.exists():
        return _load_from_jsonl(cache_path)

    problems = _try_datasets()
    if problems is None and allow_download:
        problems = _try_download()
    if problems is None:
        raise FileNotFoundError(
            f"HumanEval not available. Place a JSONL at {cache_path} or "
            f"`pip install datasets` or allow network access for "
            f"{HUMANEVAL_GZ_URL}."
        )

    cache_jsonl(problems, cache_path)
    return problems


def _difficulty_proxy(problem: dict) -> int:
    """Length of the docstring (longer docstring => harder problem).

    Used as a stratification proxy when no canonical difficulty label is
    available. Buckets: easy (<400 chars), medium (400-800), hard (>=800).
    """
    return len(problem.get("prompt", ""))


def stratified_sample(
    problems: list,
    n: int = 30,
    seed: int = 42,
    stratify: bool = True,
) -> list:
    """Sample ``n`` problems, stratified by docstring length if requested.

    Returns problems in deterministic order based on ``seed``.
    """
    rng = random.Random(seed)

    if not stratify:
        sampled = list(problems)
        rng.shuffle(sampled)
        return sampled[:n]

    # Three-bucket stratification by docstring length.
    by_bucket = {"easy": [], "medium": [], "hard": []}
    for p in problems:
        L = _difficulty_proxy(p)
        if L < 400:
            by_bucket["easy"].append(p)
        elif L < 800:
            by_bucket["medium"].append(p)
        else:
            by_bucket["hard"].append(p)

    # Allocate proportionally, with a minimum of 1 per non-empty bucket.
    total = sum(len(v) for v in by_bucket.values())
    if total == 0:
        return []
    quotas = {}
    for k, items in by_bucket.items():
        if not items:
            quotas[k] = 0
        else:
            quotas[k] = max(1, round(n * len(items) / total))

    # Adjust to exact n by trimming the largest bucket.
    while sum(quotas.values()) > n:
        biggest = max(quotas, key=quotas.get)
        quotas[biggest] -= 1
    while sum(quotas.values()) < n:
        biggest = max(by_bucket, key=lambda k: len(by_bucket[k]))
        quotas[biggest] += 1

    sampled = []
    for k, q in quotas.items():
        bucket = list(by_bucket[k])
        rng.shuffle(bucket)
        sampled.extend(bucket[:q])

    rng.shuffle(sampled)
    return sampled


def format_prompt(problem: dict) -> str:
    """Return the prompt to send to the LLM.

    HumanEval prompts are the function signature + docstring; the model
    is expected to produce only the function body.
    """
    return problem["prompt"]


def get_test_harness(problem: dict) -> str:
    """Return the executable test harness code.

    The harness defines a ``check(candidate)`` function and asserts the
    candidate's behaviour. ``run_pass_at_1`` will combine this with the
    completion to evaluate correctness.
    """
    return problem["test"]
