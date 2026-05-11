"""Tests for src/tasks/humaneval_loader.py."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks import humaneval_loader  # noqa: E402


@pytest.fixture
def fake_humaneval_jsonl(tmp_path: Path) -> Path:
    """Write a 6-problem JSONL covering the three difficulty buckets."""
    problems = []
    # 2 easy (short prompts), 2 medium, 2 hard.
    for i in range(2):
        problems.append({
            "task_id": f"HumanEval/E{i}",
            "prompt": "def f():\n    \"\"\"easy\"\"\"\n",
            "canonical_solution": "    pass\n",
            "test": "def check(c): pass",
            "entry_point": "f",
        })
    for i in range(2):
        problems.append({
            "task_id": f"HumanEval/M{i}",
            "prompt": "def g():\n    \"\"\"" + ("medium " * 60) + "\"\"\"\n",
            "canonical_solution": "    pass\n",
            "test": "def check(c): pass",
            "entry_point": "g",
        })
    for i in range(2):
        problems.append({
            "task_id": f"HumanEval/H{i}",
            "prompt": "def h():\n    \"\"\"" + ("hard " * 200) + "\"\"\"\n",
            "canonical_solution": "    pass\n",
            "test": "def check(c): pass",
            "entry_point": "h",
        })
    cache = tmp_path / "humaneval.jsonl"
    with open(cache, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    return cache


def test_load_from_local_jsonl(fake_humaneval_jsonl):
    out = humaneval_loader.load_humaneval(cache_path=fake_humaneval_jsonl)
    assert len(out) == 6
    assert all("task_id" in p for p in out)
    assert all("prompt" in p for p in out)
    assert all("test" in p for p in out)
    assert all("entry_point" in p for p in out)


def test_stratified_sample_returns_n(fake_humaneval_jsonl):
    problems = humaneval_loader.load_humaneval(cache_path=fake_humaneval_jsonl)
    sampled = humaneval_loader.stratified_sample(problems, n=4, seed=42)
    assert len(sampled) == 4
    # Stratification should pull from at least 2 buckets.
    bucket_ids = {
        "easy": "HumanEval/E", "medium": "HumanEval/M", "hard": "HumanEval/H"
    }
    sampled_buckets = set()
    for p in sampled:
        for k, prefix in bucket_ids.items():
            if p["task_id"].startswith(prefix):
                sampled_buckets.add(k)
    assert len(sampled_buckets) >= 2


def test_stratified_sample_deterministic(fake_humaneval_jsonl):
    problems = humaneval_loader.load_humaneval(cache_path=fake_humaneval_jsonl)
    s1 = humaneval_loader.stratified_sample(problems, n=4, seed=42)
    s2 = humaneval_loader.stratified_sample(problems, n=4, seed=42)
    assert [p["task_id"] for p in s1] == [p["task_id"] for p in s2]


def test_unstratified_sample(fake_humaneval_jsonl):
    problems = humaneval_loader.load_humaneval(cache_path=fake_humaneval_jsonl)
    sampled = humaneval_loader.stratified_sample(
        problems, n=3, seed=42, stratify=False
    )
    assert len(sampled) == 3


def test_format_prompt_returns_problem_prompt(fake_humaneval_jsonl):
    problems = humaneval_loader.load_humaneval(cache_path=fake_humaneval_jsonl)
    p = problems[0]
    assert humaneval_loader.format_prompt(p) == p["prompt"]


def test_get_test_harness(fake_humaneval_jsonl):
    problems = humaneval_loader.load_humaneval(cache_path=fake_humaneval_jsonl)
    p = problems[0]
    assert humaneval_loader.get_test_harness(p) == p["test"]


def test_missing_cache_no_network_raises(tmp_path):
    """No cache + downloads disabled => FileNotFoundError."""
    cache = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError):
        humaneval_loader.load_humaneval(
            cache_path=cache, allow_download=False
        )


def test_cache_jsonl_round_trip(tmp_path):
    problems = [
        {"task_id": "HumanEval/0", "prompt": "x", "canonical_solution": "y",
         "test": "z", "entry_point": "f"},
    ]
    path = tmp_path / "cache.jsonl"
    humaneval_loader.cache_jsonl(problems, path)
    out = humaneval_loader._load_from_jsonl(path)
    assert out == problems
