"""Core test suite for the GenAI reproducibility protocol.

Tests cover:
  - compute_per_abstract_emr (bootstrap_analysis)
  - bootstrap_ci (bootstrap_analysis)
  - parse_filename (bootstrap_analysis)
  - hash_text, hash_file, hash_dict (protocol.hasher)
  - exact_match, exact_match_all_pairs, rouge_l_scores, edit_distance_stats (metrics.variability)
"""

import hashlib
import math
import os
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Path setup: the project has no pyproject.toml / setup.py, so we add the
# necessary directories to sys.path so that imports resolve.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)                  # for 'src' package
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))  # for 'metrics', 'protocol'
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))  # for bootstrap_analysis

from bootstrap_analysis import compute_per_abstract_emr, bootstrap_ci, parse_filename
from protocol.hasher import hash_text, hash_file, hash_dict
from metrics.variability import (
    exact_match,
    exact_match_all_pairs,
    edit_distance_stats,
    rouge_l_scores,
)


# ===================================================================
# 1. compute_per_abstract_emr
# ===================================================================
class TestComputePerAbstractEMR:
    """Tests for the pairwise exact-match-rate function."""

    def test_all_identical(self):
        """All hashes identical -> EMR must be 1.0."""
        hashes = ["abc123"] * 10
        assert compute_per_abstract_emr(hashes) == 1.0

    def test_all_different(self):
        """Every hash unique -> EMR must be 0.0."""
        hashes = [f"hash_{i}" for i in range(5)]
        assert compute_per_abstract_emr(hashes) == 0.0

    def test_mixed_case_two_groups(self):
        """Two groups of identical hashes give a predictable EMR.

        3 copies of 'a' and 2 copies of 'b' -> C(5,2)=10 pairs total.
        Matching pairs: C(3,2) + C(2,1) = 3 + 1 = 4  -> EMR = 4/10 = 0.4
        """
        hashes = ["a", "a", "a", "b", "b"]
        assert compute_per_abstract_emr(hashes) == pytest.approx(0.4)

    def test_mixed_case_half_and_half(self):
        """2 of each -> C(4,2)=6 pairs, 2 matching -> EMR = 2/6 = 1/3."""
        hashes = ["x", "x", "y", "y"]
        assert compute_per_abstract_emr(hashes) == pytest.approx(1.0 / 3.0)

    def test_edge_single_element(self):
        """n < 2 should return None (cannot form pairs)."""
        assert compute_per_abstract_emr(["only_one"]) is None

    def test_edge_empty(self):
        """Empty list -> None."""
        assert compute_per_abstract_emr([]) is None

    def test_exactly_two_identical(self):
        """Two identical hashes -> exactly 1 pair, EMR = 1.0."""
        assert compute_per_abstract_emr(["same", "same"]) == 1.0

    def test_exactly_two_different(self):
        """Two different hashes -> exactly 1 pair, EMR = 0.0."""
        assert compute_per_abstract_emr(["a", "b"]) == 0.0

    def test_large_identical_set(self):
        """100 identical hashes -> EMR must still be 1.0."""
        hashes = ["deterministic"] * 100
        assert compute_per_abstract_emr(hashes) == 1.0


# ===================================================================
# 2. bootstrap_ci
# ===================================================================
class TestBootstrapCI:
    """Tests for the bootstrap confidence-interval function."""

    def test_basic_structure(self):
        """Returned dict has required keys with sensible ordering."""
        emrs = [1.0, 0.8, 0.9, 1.0, 0.7, 0.95, 0.85]
        result = bootstrap_ci(emrs, n_boot=500, seed=99)
        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "n_abstracts" in result
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_deterministic_with_seed(self):
        """Same seed must produce identical results."""
        emrs = [0.5, 0.6, 0.7, 0.8, 0.9]
        r1 = bootstrap_ci(emrs, n_boot=2000, seed=42)
        r2 = bootstrap_ci(emrs, n_boot=2000, seed=42)
        assert r1 == r2

    def test_different_seed_may_differ(self):
        """Different seeds should (almost certainly) give different CIs.

        We use high-variance data and compare multiple fields so that
        rounding to 4 decimal places does not mask the difference.
        """
        emrs = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                0.05, 0.95, 0.12, 0.88, 0.33, 0.67, 0.01, 0.99, 0.45, 0.55]
        r1 = bootstrap_ci(emrs, n_boot=10000, seed=1)
        r2 = bootstrap_ci(emrs, n_boot=10000, seed=7)
        # With high variance and 20 data points, at least one bound should differ
        differs = (
            r1["ci_lower"] != r2["ci_lower"]
            or r1["ci_upper"] != r2["ci_upper"]
            or r1["std"] != r2["std"]
        )
        assert differs

    def test_all_same_emr(self):
        """Constant EMR values -> CI should collapse to a single point."""
        emrs = [1.0] * 20
        result = bootstrap_ci(emrs, n_boot=1000, seed=7)
        assert result["mean"] == 1.0
        assert result["ci_lower"] == 1.0
        assert result["ci_upper"] == 1.0

    def test_single_value(self):
        """Single observation -> mean returned, CI collapses."""
        result = bootstrap_ci([0.75], n_boot=500, seed=0)
        assert result["mean"] == 0.75
        assert result["ci_lower"] == result["ci_upper"] == 0.75
        assert result["n_abstracts"] == 1

    def test_empty_input(self):
        """Empty list -> mean is None."""
        result = bootstrap_ci([], n_boot=100, seed=0)
        assert result["mean"] is None
        assert result["n_abstracts"] == 0

    def test_ci_width_decreases_with_more_data(self):
        """Adding more (identical-distribution) data should narrow the CI."""
        import numpy as np
        rng = np.random.RandomState(123)
        small = list(rng.uniform(0.4, 0.6, size=5))
        large = list(rng.uniform(0.4, 0.6, size=100))
        ci_small = bootstrap_ci(small, n_boot=5000, seed=42)
        ci_large = bootstrap_ci(large, n_boot=5000, seed=42)
        width_small = ci_small["ci_upper"] - ci_small["ci_lower"]
        width_large = ci_large["ci_upper"] - ci_large["ci_lower"]
        assert width_large < width_small


# ===================================================================
# 3. parse_filename
# ===================================================================
class TestParseFilename:
    """Tests for run-file filename parsing."""

    def test_valid_local_model(self):
        """Standard local-model filename parses correctly."""
        result = parse_filename("llama3_8b_extraction_abs_001_C1_fixed_seed_rep1.json")
        assert result is not None
        model, task, abs_num, condition, rep = result
        assert model == "llama3_8b"
        assert task == "extraction"
        assert abs_num == 1
        assert condition == "C1_fixed_seed"
        assert rep == 1

    def test_valid_api_model(self):
        """API model (sonnet-4-5) filename parses correctly."""
        result = parse_filename("sonnet-4-5_summarization_abs_015_C1_fixed_seed_rep3.json")
        assert result is not None
        model, task, abs_num, condition, rep = result
        assert model == "claude_sonnet"
        assert task == "summarization"
        assert abs_num == 15
        assert rep == 3

    def test_gpt4_filename(self):
        """GPT-4 filename with C2_same_params."""
        result = parse_filename("gpt-4_extraction_abs_010_C2_same_params_rep5.json")
        assert result is not None
        model, task, abs_num, condition, rep = result
        assert model == "gpt4"
        assert task == "extraction"
        assert abs_num == 10
        assert condition == "C2_same_params"
        assert rep == 5

    def test_multiturn_task(self):
        """Multiturn refinement task filename."""
        result = parse_filename("mistral_7b_multiturn_refinement_abs_020_C1_fixed_seed_rep2.json")
        assert result is not None
        model, task, abs_num, condition, rep = result
        assert model == "mistral_7b"
        assert task == "multiturn_refinement"
        assert abs_num == 20
        assert rep == 2

    def test_rejects_non_json(self):
        """Non-.json extension returns None."""
        assert parse_filename("llama3_8b_extraction_abs_001_C1_fixed_seed_rep1.txt") is None

    def test_rejects_no_abs_marker(self):
        """Filename without '_abs_' marker returns None."""
        assert parse_filename("llama3_8b_extraction_001_C1_fixed_seed_rep1.json") is None

    def test_rejects_no_rep(self):
        """Filename without _repN returns None."""
        assert parse_filename("llama3_8b_extraction_abs_001_C1_fixed_seed.json") is None

    def test_rejects_unknown_model(self):
        """Unknown model prefix returns None."""
        assert parse_filename("unknownmodel_extraction_abs_001_C1_fixed_seed_rep1.json") is None

    def test_rejects_chat_control(self):
        """Chat-control tasks are explicitly skipped (returns None)."""
        assert parse_filename("llama3_8b_chat_control_abs_001_C1_fixed_seed_rep1.json") is None


# ===================================================================
# 4. Hashing functions (protocol.hasher)
# ===================================================================
class TestHasher:
    """Tests for SHA-256 hashing utilities."""

    def test_hash_text_deterministic(self):
        """Same text always produces same hash."""
        assert hash_text("hello world") == hash_text("hello world")

    def test_hash_text_different_inputs(self):
        """Different texts produce different hashes."""
        assert hash_text("hello") != hash_text("world")

    def test_hash_text_matches_stdlib(self):
        """hash_text result must match direct hashlib computation."""
        text = "reproducibility matters"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert hash_text(text) == expected

    def test_hash_text_empty_string(self):
        """Empty string is a valid input and has a known SHA-256."""
        expected = hashlib.sha256(b"").hexdigest()
        assert hash_text("") == expected

    def test_hash_text_unicode(self):
        """Unicode text hashes without error and is deterministic."""
        h1 = hash_text("cafe\u0301")  # e + combining accent
        h2 = hash_text("cafe\u0301")
        assert h1 == h2
        # Different Unicode normalization forms should differ
        assert hash_text("caf\u00e9") != hash_text("cafe\u0301")

    def test_hash_file(self):
        """hash_file on a temp file matches expected SHA-256."""
        content = b"test content for hashing"
        expected = hashlib.sha256(content).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            path = f.name
        try:
            assert hash_file(path) == expected
        finally:
            os.unlink(path)

    def test_hash_dict_deterministic(self):
        """Same dict always produces same hash regardless of insertion order."""
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert hash_dict(d1) == hash_dict(d2)

    def test_hash_dict_different_values(self):
        """Different dict values produce different hashes."""
        assert hash_dict({"key": "value1"}) != hash_dict({"key": "value2"})


# ===================================================================
# 5. EMR / variability edge cases (metrics.variability)
# ===================================================================
class TestVariabilityMetrics:
    """Tests for exact_match, exact_match_all_pairs, rouge_l, edit_distance."""

    # --- exact_match (first-element reference) ---
    def test_exact_match_all_same(self):
        assert exact_match(["a", "a", "a"]) == 1.0

    def test_exact_match_all_different(self):
        assert exact_match(["a", "b", "c"]) == 0.0

    def test_exact_match_single(self):
        """Single element -> vacuously 1.0."""
        assert exact_match(["only"]) == 1.0

    def test_exact_match_empty(self):
        """Empty list -> 1.0 (no outputs to disagree)."""
        assert exact_match([]) == 1.0

    # --- exact_match_all_pairs ---
    def test_all_pairs_identical(self):
        assert exact_match_all_pairs(["x"] * 5) == 1.0

    def test_all_pairs_unique(self):
        assert exact_match_all_pairs(["a", "b", "c", "d"]) == 0.0

    def test_all_pairs_single(self):
        assert exact_match_all_pairs(["one"]) == 1.0

    def test_all_pairs_empty(self):
        assert exact_match_all_pairs([]) == 1.0

    def test_all_pairs_large_identical(self):
        """100 identical outputs -> EMR = 1.0."""
        assert exact_match_all_pairs(["same"] * 100) == 1.0

    def test_all_pairs_mixed(self):
        """3x 'a' + 2x 'b' -> C(3,2)+C(2,1) = 4 out of C(5,2)=10 -> 0.4."""
        outputs = ["a", "a", "a", "b", "b"]
        assert exact_match_all_pairs(outputs) == pytest.approx(0.4)

    # --- edit_distance_stats ---
    def test_edit_distance_identical(self):
        stats = edit_distance_stats(["hello", "hello", "hello"])
        assert stats["mean"] == 0.0
        assert stats["normalized_mean"] == 0.0

    def test_edit_distance_single(self):
        stats = edit_distance_stats(["only"])
        assert stats["mean"] == 0.0

    # --- rouge_l_scores ---
    def test_rouge_l_identical(self):
        scores = rouge_l_scores(["the cat sat", "the cat sat"])
        assert scores["mean"] == pytest.approx(1.0)

    def test_rouge_l_completely_different(self):
        scores = rouge_l_scores(["aaa bbb ccc", "xxx yyy zzz"])
        assert scores["mean"] == pytest.approx(0.0)

    def test_rouge_l_single(self):
        """Single output -> vacuously perfect."""
        scores = rouge_l_scores(["just one"])
        assert scores["mean"] == 1.0

    def test_rouge_l_partial_overlap(self):
        """Partial word overlap should give 0 < ROUGE-L < 1."""
        scores = rouge_l_scores(["the quick brown fox", "the slow brown dog"])
        assert 0.0 < scores["mean"] < 1.0
