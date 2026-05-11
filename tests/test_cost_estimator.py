"""Tests for src/cost_estimator.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import cost_estimator  # noqa: E402


def test_count_tokens_empty():
    assert cost_estimator.count_tokens("") == 0


def test_count_tokens_nonempty_at_least_one():
    assert cost_estimator.count_tokens("hi") >= 1


def test_count_tokens_scales_with_length():
    short = cost_estimator.count_tokens("abc")
    long = cost_estimator.count_tokens("abc" * 100)
    assert long > short


def test_resolve_alias_to_canonical():
    assert cost_estimator.resolve_model("gemini-2-5-pro") == "gemini-2.5-pro"
    assert cost_estimator.resolve_model("claude-sonnet-4-5") \
        == "claude-sonnet-4-5-20250929"


def test_resolve_unknown_returns_input():
    assert cost_estimator.resolve_model("zzz-mystery") == "zzz-mystery"


def test_pricing_known_models():
    assert cost_estimator.get_pricing("gpt-4o")["input"] == 2.50
    assert cost_estimator.get_pricing("gpt-4")["output"] == 60.00
    assert cost_estimator.get_pricing("deepseek-chat")["input"] == 0.27


def test_pricing_unknown_returns_zeros():
    p = cost_estimator.get_pricing("never-heard-of-it")
    assert p == {"input": 0.0, "output": 0.0}


def test_estimate_call_cost_known_stack_nonzero():
    est = cost_estimator.estimate_call_cost(
        stack="gpt-4o",
        input_text="hello world " * 100,
        max_output_tokens=512,
        n_reps=1,
    )
    assert est.total_usd > 0
    assert est.input_tokens > 0
    assert est.output_tokens == 512


def test_estimate_call_cost_local_is_zero():
    est = cost_estimator.estimate_call_cost(
        stack="llama3-8b-local",
        input_text="hello",
        max_output_tokens=100,
        n_reps=5,
    )
    assert est.total_usd == 0.0
    assert "local" in (est.notes[0] if est.notes else "").lower()


def test_estimate_call_cost_unknown_warns():
    est = cost_estimator.estimate_call_cost(
        stack="totally-unknown-stack",
        input_text="hi",
        max_output_tokens=10,
        n_reps=1,
    )
    assert est.total_usd == 0.0
    assert any("UNKNOWN" in n for n in est.notes)


def test_estimate_call_cost_scales_with_reps():
    est_1 = cost_estimator.estimate_call_cost(
        "gpt-4o", "hi" * 200, 256, n_reps=1
    )
    est_5 = cost_estimator.estimate_call_cost(
        "gpt-4o", "hi" * 200, 256, n_reps=5
    )
    assert est_5.total_usd == pytest.approx(est_1.total_usd * 5)


def test_estimate_task_cost_aggregates_prompts():
    prompts = ["a" * 400, "b" * 400, "c" * 400]
    est = cost_estimator.estimate_task_cost(
        stack="claude-sonnet-4-5", prompts=prompts,
        max_output_tokens=200, n_reps=2, n_turns=1,
    )
    assert est.n_calls == 6  # 3 prompts × 2 reps
    assert est.input_tokens > 0
    assert est.output_tokens == 200 * 1 * 2 * 3


def test_estimate_task_cost_multiturn_grows():
    prompts = ["x" * 1000]
    single = cost_estimator.estimate_task_cost(
        "gpt-4o", prompts, 200, n_reps=1, n_turns=1
    )
    triple = cost_estimator.estimate_task_cost(
        "gpt-4o", prompts, 200, n_reps=1, n_turns=3
    )
    assert triple.input_tokens > single.input_tokens
    assert triple.output_tokens == single.output_tokens * 3


# ─── BudgetGuard ─────────────────────────────────────────────────────────────

def test_budget_guard_allows_small_charge():
    g = cost_estimator.BudgetGuard(cap_usd=10.0)
    assert g.would_allow(2.0)
    g.charge(2.0, label="run1")
    assert g.spent_usd == 2.0
    assert g.remaining() == 8.0


def test_budget_guard_blocks_over_cap():
    g = cost_estimator.BudgetGuard(cap_usd=5.0)
    g.charge(4.0)
    assert g.would_allow(0.5)
    assert not g.would_allow(2.0)


def test_budget_guard_summary_round_trip():
    g = cost_estimator.BudgetGuard(cap_usd=10.0)
    g.charge(3.0, label="x")
    g.charge(1.0, label="y")
    s = g.summary()
    assert s["cap_usd"] == 10.0
    assert s["spent_usd"] == 4.0
    assert s["remaining_usd"] == 6.0
    assert s["n_charges"] == 2
