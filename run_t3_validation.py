"""
T3 PM2.5 protected triangulation runner — Major Revision NatComms.

Two phases:
  PHASE A (sample, ~zero cost): pull 10 disagreement cases from the
  companion paper's extraction_long.json (cloud models only), distinct
  from the 23 effects reported in that paper, stratified by model and
  disagreement kind.

  PHASE B (judge, ~$3-5 USD): run Claude Opus 4.7 LLM-as-judge on each
  case with three pre-registered criteria (direction, magnitude +/- 20%,
  CI overlap), blind and run-order-randomised. Save per-case Run Cards
  and an aggregate JSON ready for the Extended Data table.

Usage:
  # Step 1: sample + cache (zero cost, no API)
  python run_t3_validation.py --sample

  # Step 2: dry-run judge to check cost
  python run_t3_validation.py --judge --dry-run

  # Step 3: actually judge (after authorisation)
  python run_t3_validation.py --judge --execute

This script is RESUMABLE. The --sample step caches the 10 cases at
data/inputs/revision/t3_judge_cases.json. The --judge step writes per-case
Run Cards in outputs/revision/t3_judge/ and an aggregate at
outputs/revision/t3_judge/t3_judge_results.json.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.tasks import llm_judge, pm25_case_loader  # noqa: E402

CASES_PATH = ROOT / "data" / "inputs" / "revision" / "t3_judge_cases.json"
JUDGE_OUTPUT_DIR = ROOT / "outputs" / "revision" / "t3_judge"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true",
                        help="Phase A: sample 10 cases from companion paper data.")
    parser.add_argument("--judge", action="store_true",
                        help="Phase B: run Claude Opus LLM-as-judge.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan + cost only, no API calls (with --judge).")
    parser.add_argument("--execute", action="store_true",
                        help="Actually call the API (with --judge).")
    parser.add_argument("--n-cases", type=int, default=10,
                        help="Number of cases to sample (default 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default 42).")
    args = parser.parse_args()

    if not (args.sample or args.judge):
        parser.error("Pass --sample, --judge, or both.")

    # PHASE A: sample
    if args.sample:
        print("=" * 70)
        print(f"PHASE A: Sampling {args.n_cases} cases from companion paper")
        print(f"  Source: {pm25_case_loader.EXTRACTION_LONG}")
        print(f"  Cache: {CASES_PATH}")
        print(f"  Models: {sorted(pm25_case_loader.CLOUD_MODELS)}")
        print(f"  Seed: {args.seed}")
        print("=" * 70)
        if not pm25_case_loader.EXTRACTION_LONG.exists():
            print(f"ERROR: companion paper data not found at {pm25_case_loader.EXTRACTION_LONG}")
            sys.exit(1)
        out = pm25_case_loader.cache_sample(CASES_PATH, n=args.n_cases, seed=args.seed)
        print(f"\n  -> Cached {args.n_cases} cases at {out}")

        # Briefly summarise what we sampled
        import json
        with open(out) as f:
            bundle = json.load(f)
        print("\n  Stratification summary:")
        by_model: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        for c in bundle["cases"]:
            by_model[c["model"]] = by_model.get(c["model"], 0) + 1
            by_kind[c["disagreement_kind"]] = by_kind.get(c["disagreement_kind"], 0) + 1
        print(f"    by model: {by_model}")
        print(f"    by disagreement kind: {by_kind}")

    # PHASE B: judge
    if args.judge:
        print("\n" + "=" * 70)
        print("PHASE B: LLM-as-judge")
        print(f"  Cases: {CASES_PATH}")
        print(f"  Output: {JUDGE_OUTPUT_DIR}")
        print(f"  Mode: {'DRY-RUN (no API)' if args.dry_run else 'EXECUTE (real API)'}")
        print("=" * 70)
        if not CASES_PATH.exists():
            print(f"ERROR: cases not cached. Run --sample first.")
            sys.exit(1)
        if not args.dry_run and not args.execute:
            print("ERROR: must pass --dry-run OR --execute")
            sys.exit(1)
        if args.execute and not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not set in environment")
            sys.exit(1)

        llm_judge.judge_all_cases(
            cases_file=CASES_PATH,
            output_dir=JUDGE_OUTPUT_DIR,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
