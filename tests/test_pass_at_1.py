"""Tests for src/tasks/pass_at_1.py — sandboxed code execution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks import pass_at_1  # noqa: E402


# A minimal HumanEval-style problem for testing.
ADD_PROBLEM = {
    "prompt": (
        "def add(a: int, b: int) -> int:\n"
        '    """Return a + b."""\n'
    ),
    "test": (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(0, 0) == 0\n"
        "    assert candidate(-1, 1) == 0\n"
    ),
    "entry_point": "add",
}


def test_correct_completion_passes():
    completion = "    return a + b\n"
    res = pass_at_1.run_pass_at_1(
        ADD_PROBLEM["prompt"], completion, ADD_PROBLEM["test"],
        ADD_PROBLEM["entry_point"], timeout=5.0,
    )
    assert res["passed"] is True
    assert res["timed_out"] is False
    assert res["return_code"] == 0
    assert "duration_ms" in res


def test_wrong_completion_fails():
    completion = "    return a - b\n"  # Buggy.
    res = pass_at_1.run_pass_at_1(
        ADD_PROBLEM["prompt"], completion, ADD_PROBLEM["test"],
        ADD_PROBLEM["entry_point"], timeout=5.0,
    )
    assert res["passed"] is False
    assert res["timed_out"] is False
    assert "AssertionError" in res["stderr"] or res["return_code"] != 0


def test_full_function_redefined():
    """Model returns a complete `def add(...): ...` rather than just body."""
    completion = "def add(a, b):\n    return a + b\n"
    res = pass_at_1.run_pass_at_1(
        ADD_PROBLEM["prompt"], completion, ADD_PROBLEM["test"],
        ADD_PROBLEM["entry_point"], timeout=5.0,
    )
    assert res["passed"] is True


def test_code_fence_stripped():
    """Model wrapped its answer in ``` ... ``` block."""
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    res = pass_at_1.run_pass_at_1(
        ADD_PROBLEM["prompt"], completion, ADD_PROBLEM["test"],
        ADD_PROBLEM["entry_point"], timeout=5.0,
    )
    assert res["passed"] is True


def test_timeout_kills_runaway_completion():
    """Infinite loop is sandbox-killed within the timeout."""
    completion = "    while True:\n        pass\n"
    res = pass_at_1.run_pass_at_1(
        ADD_PROBLEM["prompt"], completion, ADD_PROBLEM["test"],
        ADD_PROBLEM["entry_point"], timeout=2.0,
    )
    assert res["passed"] is False
    assert res["timed_out"] is True


def test_runtime_error_is_caught():
    completion = "    raise RuntimeError('boom')\n"
    res = pass_at_1.run_pass_at_1(
        ADD_PROBLEM["prompt"], completion, ADD_PROBLEM["test"],
        ADD_PROBLEM["entry_point"], timeout=5.0,
    )
    assert res["passed"] is False
    assert res["return_code"] != 0


def test_pass_at_1_metric():
    results = [{"passed": True}, {"passed": False}, {"passed": True}]
    assert pass_at_1.pass_at_1_metric(results) == pytest.approx(2 / 3)


def test_pass_at_1_metric_empty():
    assert pass_at_1.pass_at_1_metric([]) == 0.0


def test_extract_function_body_passes_through_full_def():
    completion = "def add(a, b):\n    return a + b\n"
    program = pass_at_1.extract_function_body(
        ADD_PROBLEM["prompt"], completion, "add"
    )
    assert "def add" in program


def test_extract_function_body_indents_naked_body():
    completion = "return a + b"
    program = pass_at_1.extract_function_body(
        ADD_PROBLEM["prompt"], completion, "add"
    )
    # Should preserve the prompt's def line and indent the body.
    assert "def add" in program
    assert "    return a + b" in program
