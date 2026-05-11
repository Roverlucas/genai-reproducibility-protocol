"""
R2 — Extended T3 validation: 20 new cases + 2nd judge (gpt-4o) + Cohen's kappa.

Phases:
  1. Sample 20 additional disagreement cases (distinct from original 10)
  2. Run Claude Opus 4.7 on the 20 new cases
  3. Run gpt-4o on ALL 30 cases (10 original + 20 new) for inter-judge agreement
  4. Compute Cohen's kappa between Claude Opus and gpt-4o
  5. Save aggregate with the combined N=30 verdicts

Cost estimate: ~$1.10 USD
  - 20 new Claude Opus calls: ~$0.48
  - 30 gpt-4o calls: ~$0.60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.tasks import llm_judge, pm25_case_loader, gpt4o_judge  # noqa: E402

CASES_PATH_V1 = ROOT / "data" / "inputs" / "revision" / "t3_judge_cases.json"
CASES_PATH_V2 = ROOT / "data" / "inputs" / "revision" / "t3_judge_cases_extended.json"
JUDGE_OUTPUT_DIR = ROOT / "outputs" / "revision" / "t3_judge"


def _load_existing_case_ids() -> set[str]:
    with open(CASES_PATH_V1) as f:
        bundle = json.load(f)
    return {c["case_id"] for c in bundle["cases"]}


def sample_additional_cases(n_more: int = 20, seed: int = 4242) -> Path:
    """Sample n_more additional cases distinct from the original 10."""
    existing = _load_existing_case_ids()
    # Use a different seed and sample more, then filter
    cases = pm25_case_loader.sample_cases_for_judge(n=n_more + 10, seed=seed)
    new_cases = [c for c in cases if c.case_id not in existing][:n_more]
    if len(new_cases) < n_more:
        # Try a larger pool
        cases = pm25_case_loader.sample_cases_for_judge(n=50, seed=seed + 1)
        new_cases = [c for c in cases if c.case_id not in existing][:n_more]

    # Serialize same format as v1
    serializable = [{
        "case_id": c.case_id, "model": c.model, "corpus_id": c.corpus_id,
        "estimate_idx": c.estimate_idx, "disagreement_kind": c.disagreement_kind,
        "run_a": {
            "run_id": c.run_a.run_id, "effect_measure": c.run_a.effect_measure,
            "effect_estimate": c.run_a.effect_estimate, "ci_lower": c.run_a.ci_lower,
            "ci_upper": c.run_a.ci_upper, "lag": c.run_a.lag,
            "outcome_specific": c.run_a.outcome_specific,
            "exposure_increment": c.run_a.exposure_increment,
            "output_hash": c.run_a.output_hash,
        },
        "run_b": {
            "run_id": c.run_b.run_id, "effect_measure": c.run_b.effect_measure,
            "effect_estimate": c.run_b.effect_estimate, "ci_lower": c.run_b.ci_lower,
            "ci_upper": c.run_b.ci_upper, "lag": c.run_b.lag,
            "outcome_specific": c.run_b.outcome_specific,
            "exposure_increment": c.run_b.exposure_increment,
            "output_hash": c.run_b.output_hash,
        },
        "abstract_text": c.abstract_text,
    } for c in new_cases]

    CASES_PATH_V2.parent.mkdir(parents=True, exist_ok=True)
    with open(CASES_PATH_V2, "w") as f:
        json.dump({"n_cases": len(serializable), "seed": seed,
                   "source": str(pm25_case_loader.EXTRACTION_LONG),
                   "excluded_case_ids": sorted(existing),
                   "cases": serializable}, f, indent=2)
    print(f"  -> {len(serializable)} additional cases at {CASES_PATH_V2}")
    return CASES_PATH_V2


def cohens_kappa(rater_a: list[str], rater_b: list[str]) -> float:
    """Cohen's kappa for two raters with categorical verdicts."""
    assert len(rater_a) == len(rater_b)
    n = len(rater_a)
    if n == 0:
        return float("nan")
    categories = sorted(set(rater_a) | set(rater_b))
    # observed agreement
    p_o = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    # expected agreement under independence
    p_e = 0.0
    for c in categories:
        p_a = rater_a.count(c) / n
        p_b = rater_b.count(c) / n
        p_e += p_a * p_b
    if p_e == 1.0:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-only", action="store_true",
                        help="Just sample additional cases, do not run judges.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually call APIs (required to run).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan + cost only, no API calls.")
    args = parser.parse_args()

    print("=" * 70)
    print("R2 — Extended T3 Validation (N=30, 2 judges, Cohen's kappa)")
    print("=" * 70)

    # PHASE 1: sample
    print("\n[PHASE 1] Sampling 20 additional cases...")
    sample_additional_cases(n_more=20, seed=4242)

    if args.sample_only:
        print("Sample-only mode. Exiting.")
        return

    # PHASE 2: Claude Opus on new 20 cases
    print("\n[PHASE 2] Claude Opus 4.7 on 20 new cases...")
    if args.execute:
        llm_judge.judge_all_cases(
            cases_file=CASES_PATH_V2,
            output_dir=JUDGE_OUTPUT_DIR / "extended_claude",
            dry_run=False,
        )

    # PHASE 3: gpt-4o on all 30 cases (10 original + 20 new)
    print("\n[PHASE 3] gpt-4o on original 10 cases...")
    if args.execute:
        gpt4o_judge.judge_all_cases(
            cases_file=CASES_PATH_V1,
            output_dir=JUDGE_OUTPUT_DIR / "gpt4o_original10",
            dry_run=False,
        )
    print("\n[PHASE 3b] gpt-4o on 20 new cases...")
    if args.execute:
        gpt4o_judge.judge_all_cases(
            cases_file=CASES_PATH_V2,
            output_dir=JUDGE_OUTPUT_DIR / "gpt4o_extended20",
            dry_run=False,
        )

    # PHASE 4: Cohen's kappa
    if not args.execute:
        if args.dry_run:
            print("\nDry-run complete.")
        return

    print("\n[PHASE 4] Computing Cohen's kappa across N=30...")
    # Load Claude verdicts (original 10 + new 20)
    with open(JUDGE_OUTPUT_DIR / "t3_judge_results.json") as f:
        claude_v1 = json.load(f)
    with open(JUDGE_OUTPUT_DIR / "extended_claude" / "t3_judge_results.json") as f:
        claude_v2 = json.load(f)
    # Load gpt-4o verdicts
    with open(JUDGE_OUTPUT_DIR / "gpt4o_original10" / "t3_judge_gpt4o_results.json") as f:
        gpt4o_v1 = json.load(f)
    with open(JUDGE_OUTPUT_DIR / "gpt4o_extended20" / "t3_judge_gpt4o_results.json") as f:
        gpt4o_v2 = json.load(f)

    claude_by_case = {r["case_id"]: r["verdict"] for r in claude_v1["results"] + claude_v2["results"]}
    gpt4o_by_case = {r["case_id"]: r["verdict"] for r in gpt4o_v1["results"] + gpt4o_v2["results"]}

    case_ids = sorted(set(claude_by_case) & set(gpt4o_by_case))
    claude_verdicts = [claude_by_case[c] for c in case_ids]
    gpt4o_verdicts = [gpt4o_by_case[c] for c in case_ids]
    k = cohens_kappa(claude_verdicts, gpt4o_verdicts)

    # Verdict distribution
    from collections import Counter
    claude_counts = Counter(claude_verdicts)
    gpt4o_counts = Counter(gpt4o_verdicts)

    summary = {
        "n_cases": len(case_ids),
        "claude_opus_4_7": dict(claude_counts),
        "gpt_4o": dict(gpt4o_counts),
        "cohens_kappa": k,
        "agreement_rate": sum(1 for a, b in zip(claude_verdicts, gpt4o_verdicts) if a == b) / len(case_ids),
    }
    out = JUDGE_OUTPUT_DIR / "t3_extended_kappa.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
