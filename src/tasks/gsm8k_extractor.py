"""GSM8K final-answer extraction and grading.

GSM8K's gold answers are formatted as::

    <chain of thought>
    #### 42

The convention used by Cobbe et al. (2021) is to extract the integer
following ``####`` as the final answer. We mirror this and add tolerant
fallbacks for model outputs that omit the marker.
"""

from __future__ import annotations

import re
from typing import Optional

# Strict pattern: '#### <maybe minus> digits (with optional commas / dot
# decimal)'. We then strip commas and trailing decimals to compare.
_GSM8K_FINAL_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
# Fallback: last number in the text.
_NUMBER_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def _normalize(num_str: str) -> Optional[float]:
    """Convert a textual number to float; return None on failure."""
    if num_str is None:
        return None
    cleaned = num_str.replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_final_answer(text: str) -> Optional[float]:
    """Extract the GSM8K final answer from ``text``.

    Strategy:
        1. Match the canonical ``####`` marker.
        2. If absent, fall back to the last number in the text.

    Returns the numeric value as ``float`` or ``None`` if no number was
    found. Booleans / words ('forty-two') are NOT supported — they would
    be a graded miss.
    """
    if not text:
        return None
    m = _GSM8K_FINAL_RE.search(text)
    if m:
        return _normalize(m.group(1))
    nums = _NUMBER_RE.findall(text)
    if nums:
        return _normalize(nums[-1])
    return None


def is_correct(prediction: str, gold_answer: str,
               tolerance: float = 1e-6) -> bool:
    """Compare a model's prediction against the GSM8K gold answer.

    Both arguments are raw strings; the canonical answer field looks
    like ``"... reasoning ... #### 42"``.
    """
    pred = extract_final_answer(prediction)
    gold = extract_final_answer(gold_answer)
    if pred is None or gold is None:
        return False
    # Most GSM8K answers are integers; allow a tiny float epsilon.
    return abs(pred - gold) <= tolerance


def grade_runs(runs: list) -> dict:
    """Aggregate accuracy across a list of run dicts.

    Each run dict must contain ``output_text`` (model output) and
    ``gold_answer`` (raw gold string).

    Returns::

        {"n": int, "n_correct": int, "accuracy": float}
    """
    n = len(runs)
    correct = sum(
        1 for r in runs
        if is_correct(r.get("output_text", ""), r.get("gold_answer", ""))
    )
    return {
        "n": n,
        "n_correct": correct,
        "accuracy": correct / n if n else 0.0,
    }
