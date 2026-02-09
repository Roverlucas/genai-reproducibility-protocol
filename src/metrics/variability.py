"""Variability metrics for comparing outputs across experimental runs.

Implements ROUGE-L, cosine similarity, edit distance, and exact match
metrics used to quantify output variability in the GenAI reproducibility study.
"""

from typing import List, Tuple

import Levenshtein
import numpy as np


def exact_match(outputs: List[str]) -> float:
    """Calculate exact match rate: fraction of outputs identical to the first."""
    if len(outputs) < 2:
        return 1.0
    reference = outputs[0]
    matches = sum(1 for o in outputs[1:] if o == reference)
    return matches / (len(outputs) - 1)


def exact_match_all_pairs(outputs: List[str]) -> float:
    """Calculate exact match rate across all unique pairs."""
    if len(outputs) < 2:
        return 1.0
    n = len(outputs)
    total_pairs = 0
    match_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if outputs[i] == outputs[j]:
                match_count += 1
    return match_count / total_pairs if total_pairs > 0 else 1.0


def edit_distance_stats(outputs: List[str]) -> dict:
    """Calculate Levenshtein edit distance statistics across all pairs."""
    if len(outputs) < 2:
        return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "normalized_mean": 0.0}

    distances = []
    normalized = []
    n = len(outputs)
    for i in range(n):
        for j in range(i + 1, n):
            d = Levenshtein.distance(outputs[i], outputs[j])
            distances.append(d)
            max_len = max(len(outputs[i]), len(outputs[j]), 1)
            normalized.append(d / max_len)

    return {
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "min": int(np.min(distances)),
        "max": int(np.max(distances)),
        "normalized_mean": float(np.mean(normalized)),
    }


def rouge_l_scores(outputs: List[str]) -> dict:
    """Calculate ROUGE-L F1 scores across all pairs using a simple LCS implementation."""
    if len(outputs) < 2:
        return {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}

    scores = []
    n = len(outputs)
    for i in range(n):
        for j in range(i + 1, n):
            score = _rouge_l_f1(outputs[i], outputs[j])
            scores.append(score)

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def _lcs_length(x: str, y: str) -> int:
    """Compute length of longest common subsequence (word-level)."""
    x_words = x.split()
    y_words = y.split()
    m, n = len(x_words), len(y_words)
    if m == 0 or n == 0:
        return 0
    # Space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x_words[i - 1] == y_words[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _rouge_l_f1(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 between two texts (word-level LCS)."""
    lcs = _lcs_length(hypothesis, reference)
    h_len = len(hypothesis.split())
    r_len = len(reference.split())
    if h_len == 0 or r_len == 0:
        return 0.0
    precision = lcs / h_len
    recall = lcs / r_len
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_bert_scorer():
    """Create and cache a BERTScorer instance (loads model once)."""
    from bert_score import BERTScorer
    scorer = BERTScorer(lang="en", rescale_with_baseline=False)
    return scorer


def bert_score_stats(outputs: list, scorer=None) -> dict:
    """Compute pairwise BERTScore (P, R, F1) for all output pairs."""
    import itertools

    if len(outputs) < 2:
        return {"bertscore_f1_mean": None, "bertscore_f1_std": None}

    pairs = list(itertools.combinations(range(len(outputs)), 2))
    refs = [outputs[i] for i, j in pairs]
    cands = [outputs[j] for i, j in pairs]

    if scorer is not None:
        P, R, F1 = scorer.score(cands, refs)
    else:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(cands, refs, lang="en", verbose=False)

    f1_list = F1.tolist()

    return {
        "bertscore_f1_mean": float(np.mean(f1_list)),
        "bertscore_f1_std": float(np.std(f1_list)),
        "bertscore_f1_min": float(np.min(f1_list)),
        "bertscore_f1_max": float(np.max(f1_list)),
        "bertscore_precision_mean": float(np.mean(P.tolist())),
        "bertscore_recall_mean": float(np.mean(R.tolist())),
    }


def compute_all_metrics(outputs: List[str], scorer=None) -> dict:
    """Compute all variability metrics for a set of outputs."""
    return {
        "n_outputs": len(outputs),
        "exact_match_rate": exact_match_all_pairs(outputs),
        "edit_distance": edit_distance_stats(outputs),
        "rouge_l": rouge_l_scores(outputs),
        "bert_score": bert_score_stats(outputs, scorer=scorer),
        "avg_output_length_chars": float(np.mean([len(o) for o in outputs])),
        "avg_output_length_words": float(np.mean([len(o.split()) for o in outputs])),
    }
