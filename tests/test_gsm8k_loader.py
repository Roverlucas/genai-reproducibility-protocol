"""Tests for src/tasks/gsm8k_loader.py and gsm8k_extractor.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks import gsm8k_loader, gsm8k_extractor  # noqa: E402


@pytest.fixture
def fake_gsm8k(tmp_path: Path) -> Path:
    problems = [
        {"question": f"What is {i} + {i}?",
         "answer": f"reasoning... #### {2 * i}"} for i in range(5)
    ]
    path = tmp_path / "gsm8k_test.jsonl"
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    return path


# ─── Loader ──────────────────────────────────────────────────────────────────

def test_load_from_local_jsonl(fake_gsm8k):
    out = gsm8k_loader.load_gsm8k(cache_path=fake_gsm8k)
    assert len(out) == 5
    assert all("question" in p and "answer" in p for p in out)


def test_sample_problems_deterministic(fake_gsm8k):
    out = gsm8k_loader.load_gsm8k(cache_path=fake_gsm8k)
    s1 = gsm8k_loader.sample_problems(out, n=3, seed=42)
    s2 = gsm8k_loader.sample_problems(out, n=3, seed=42)
    assert s1 == s2
    assert len(s1) == 3


def test_sample_problems_n_too_large(fake_gsm8k):
    out = gsm8k_loader.load_gsm8k(cache_path=fake_gsm8k)
    s = gsm8k_loader.sample_problems(out, n=99, seed=1)
    assert len(s) == len(out)


def test_attach_id_adds_id(fake_gsm8k):
    out = gsm8k_loader.load_gsm8k(cache_path=fake_gsm8k)
    out = gsm8k_loader.attach_id(out)
    assert all(p["id"].startswith("GSM8K/") for p in out)
    assert len({p["id"] for p in out}) == len(out)


def test_format_prompt_includes_question(fake_gsm8k):
    out = gsm8k_loader.load_gsm8k(cache_path=fake_gsm8k)
    p = out[0]
    prompt = gsm8k_loader.format_prompt(p)
    assert p["question"] in prompt
    assert "####" in prompt


def test_missing_cache_no_network(tmp_path):
    with pytest.raises(FileNotFoundError):
        gsm8k_loader.load_gsm8k(
            cache_path=tmp_path / "x.jsonl", allow_download=False
        )


# ─── Extractor ───────────────────────────────────────────────────────────────

def test_extract_with_marker():
    text = "step 1\nstep 2\n#### 42"
    assert gsm8k_extractor.extract_final_answer(text) == 42


def test_extract_negative():
    assert gsm8k_extractor.extract_final_answer("#### -7") == -7


def test_extract_with_commas():
    assert gsm8k_extractor.extract_final_answer("#### 1,234") == 1234


def test_extract_decimal():
    assert gsm8k_extractor.extract_final_answer("#### 3.5") == 3.5


def test_extract_no_marker_falls_back_to_last_number():
    text = "I think it is 12 or maybe 18 final."
    assert gsm8k_extractor.extract_final_answer(text) == 18


def test_extract_empty():
    assert gsm8k_extractor.extract_final_answer("") is None
    assert gsm8k_extractor.extract_final_answer("no numbers here") is None


def test_is_correct_true():
    assert gsm8k_extractor.is_correct(
        "answer is #### 42", "long reasoning #### 42"
    )


def test_is_correct_false():
    assert not gsm8k_extractor.is_correct(
        "answer is #### 41", "long reasoning #### 42"
    )


def test_is_correct_handles_format_diff():
    assert gsm8k_extractor.is_correct(
        "the answer is 12.", "computation #### 12"
    )


def test_grade_runs_aggregates():
    runs = [
        {"output_text": "#### 1", "gold_answer": "#### 1"},
        {"output_text": "#### 2", "gold_answer": "#### 2"},
        {"output_text": "#### 9", "gold_answer": "#### 3"},
    ]
    summary = gsm8k_extractor.grade_runs(runs)
    assert summary == {"n": 3, "n_correct": 2, "accuracy": pytest.approx(2 / 3)}


def test_grade_runs_empty():
    assert gsm8k_extractor.grade_runs([]) == {
        "n": 0, "n_correct": 0, "accuracy": 0.0
    }
