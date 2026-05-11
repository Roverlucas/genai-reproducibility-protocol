"""
Compute Cohen's κ and Fleiss' κ for the human-rater validation.

Inputs:
  human_validation/rater_A_verdicts.json  — Yara's verdicts
  human_validation/rater_B_verdicts.json  — external rater's verdicts
  human_validation/selected_10_cases.json — manifest with hidden LLM-judge verdicts

Output:
  human_validation/human_kappa_results.json — full analysis summary

Supports:
  - Cohen's κ pairwise: A-vs-B, A-vs-Claude, A-vs-gpt-4o, B-vs-Claude, B-vs-gpt-4o,
                       human-consensus-vs-Claude, human-consensus-vs-gpt-4o
  - Fleiss' κ across {A, B, Claude, gpt-4o}
  - Plain agreement rates
  - Per-case verdict matrix (printed)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

HV = Path(__file__).resolve().parent

RATER_A_PATH = HV / "rater_A_verdicts.json"
RATER_B_PATH = HV / "rater_B_verdicts.json"
MANIFEST_PATH = HV / "selected_10_cases.json"
OUT_PATH = HV / "human_kappa_results.json"


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters, categorical."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    p_e = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if p_e == 1.0:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def fleiss_kappa(rater_matrix: list[list[str]]) -> float:
    """Fleiss' kappa for >=2 raters.
    rater_matrix: list of rater verdict lists; all same length.
    """
    n_raters = len(rater_matrix)
    n_subjects = len(rater_matrix[0])
    cats = sorted(set(v for r in rater_matrix for v in r))
    # n_ij: number of raters who assigned subject i to category j
    counts = [[0] * len(cats) for _ in range(n_subjects)]
    for r in rater_matrix:
        for i, v in enumerate(r):
            counts[i][cats.index(v)] += 1
    # P_i: agreement for subject i
    p_i = [(sum(c * c for c in counts[i]) - n_raters) / (n_raters * (n_raters - 1))
           for i in range(n_subjects)]
    p_bar = sum(p_i) / n_subjects
    # P_e
    p_j = [sum(counts[i][j] for i in range(n_subjects)) / (n_subjects * n_raters)
           for j in range(len(cats))]
    p_e = sum(p * p for p in p_j)
    if p_e == 1.0:
        return float("nan")
    return (p_bar - p_e) / (1 - p_e)


def landis_koch(k: float) -> str:
    if k != k:  # NaN
        return "undefined"
    if k < 0.0: return "poor"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def human_consensus(a: list[str], b: list[str]) -> list[str]:
    """If both agree, return their verdict; otherwise return 'ambiguous'."""
    return [x if x == y else "ambiguous" for x, y in zip(a, b)]


def main():
    if not RATER_A_PATH.exists():
        sys.exit(f"❌ Rater A verdicts not found at {RATER_A_PATH}. Transcribe Yara's form first.")
    if not RATER_B_PATH.exists():
        sys.exit(f"❌ Rater B verdicts not found at {RATER_B_PATH}. Transcribe external rater's form first.")
    if not MANIFEST_PATH.exists():
        sys.exit(f"❌ Manifest not found at {MANIFEST_PATH}. Run select_10_for_human_validation.py first.")

    with open(RATER_A_PATH) as f: rater_a = json.load(f)
    with open(RATER_B_PATH) as f: rater_b = json.load(f)
    with open(MANIFEST_PATH) as f: manifest = json.load(f)

    case_nums = [f"case_{i+1:02d}" for i in range(len(manifest["cases"]))]
    verdict_a = [rater_a["case_verdicts"][cn] for cn in case_nums]
    verdict_b = [rater_b["case_verdicts"][cn] for cn in case_nums]
    verdict_claude = [m["claude_verdict_hidden"] for m in manifest["cases"]]
    verdict_gpt4o = [m["gpt4o_verdict_hidden"] for m in manifest["cases"]]

    # Filter to common-domain categories
    consensus = human_consensus(verdict_a, verdict_b)

    print("=" * 70)
    print(f" Human-rater validation results (n={len(case_nums)} cases)")
    print("=" * 70)
    print()
    print(f"Rater A ({rater_a.get('rater_name', '?')}):")
    print(f"  verdict counts: {dict(Counter(verdict_a))}")
    print(f"Rater B ({rater_b.get('rater_name', '?')}):")
    print(f"  verdict counts: {dict(Counter(verdict_b))}")
    print(f"Claude Opus 4.7 (hidden during rating):")
    print(f"  verdict counts: {dict(Counter(verdict_claude))}")
    print(f"gpt-4o (hidden during rating):")
    print(f"  verdict counts: {dict(Counter(verdict_gpt4o))}")
    print()

    # Pairwise Cohen's kappa
    pairs = {
        "rater_A_vs_rater_B": (verdict_a, verdict_b),
        "rater_A_vs_claude_opus": (verdict_a, verdict_claude),
        "rater_A_vs_gpt4o": (verdict_a, verdict_gpt4o),
        "rater_B_vs_claude_opus": (verdict_b, verdict_claude),
        "rater_B_vs_gpt4o": (verdict_b, verdict_gpt4o),
        "human_consensus_vs_claude_opus": (consensus, verdict_claude),
        "human_consensus_vs_gpt4o": (consensus, verdict_gpt4o),
    }
    kappa_results = {}
    print("Pairwise Cohen's kappa:")
    for label, (x, y) in pairs.items():
        k = cohens_kappa(x, y)
        agree = sum(1 for a, b in zip(x, y) if a == b) / len(x)
        kappa_results[label] = {"kappa": k, "interpretation": landis_koch(k), "agreement_rate": agree}
        print(f"  {label:42s}  κ={k:+.3f}  ({landis_koch(k)})  agree={agree:.0%}")

    # Fleiss' kappa across 4 raters
    fleiss = fleiss_kappa([verdict_a, verdict_b, verdict_claude, verdict_gpt4o])
    print(f"\nFleiss' kappa across 4 raters (A, B, Claude, gpt-4o):  κ={fleiss:+.3f}  ({landis_koch(fleiss)})")

    # Per-case verdict matrix
    print("\nPer-case verdict matrix:")
    print(f"  {'Case':6s}  {'A':24s}  {'B':24s}  {'Claude':24s}  {'gpt-4o':24s}")
    for cn, a, b, c, g in zip(case_nums, verdict_a, verdict_b, verdict_claude, verdict_gpt4o):
        marker = "" if a == b == c == g else (" ⚠" if (a == b) and (c == g) and (a != c) else "")
        print(f"  {cn:6s}  {a:24s}  {b:24s}  {c:24s}  {g:24s}{marker}")

    # Save aggregate
    out = {
        "n_cases": len(case_nums),
        "raters": {
            "A": {"name": rater_a.get("rater_name"), "verdict_counts": dict(Counter(verdict_a))},
            "B": {"name": rater_b.get("rater_name"), "verdict_counts": dict(Counter(verdict_b))},
            "claude_opus_4_7": {"verdict_counts": dict(Counter(verdict_claude))},
            "gpt_4o": {"verdict_counts": dict(Counter(verdict_gpt4o))},
        },
        "pairwise_cohens_kappa": kappa_results,
        "fleiss_kappa_4_raters": {"kappa": fleiss, "interpretation": landis_koch(fleiss)},
        "human_consensus_verdict_counts": dict(Counter(consensus)),
        "per_case_matrix": [
            {"case_num": cn, "case_id": m["case_id"],
             "rater_A": a, "rater_B": b, "claude_opus_4_7": c, "gpt_4o": g}
            for cn, m, a, b, c, g in zip(case_nums, manifest["cases"],
                                          verdict_a, verdict_b, verdict_claude, verdict_gpt4o)
        ],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to: {OUT_PATH}")
    print("\nNext: Sage will insert these κ values into manuscript §2.7, supplementary §S12.4,")
    print("response letter R3.6, and cover letter item 5 (see 00_PROTOCOL.md §8).")


if __name__ == "__main__":
    main()
