#!/usr/bin/env python3
"""Run the complete analysis pipeline after experiments are done.

Steps:
  1. Merge all run files into all_runs.json
  2. Compute variability metrics (EMR, NED, ROUGE-L, BERTScore)
  3. Compute JSON validation metrics for extraction tasks
  4. Compute overhead metrics
  5. Generate LaTeX tables (v2)
  6. Generate publication figures (v2)
  7. Run ablation study
  8. Run statistical tests

Usage:
    python run_full_analysis.py
    python run_full_analysis.py --skip-bertscore  # Skip slow BERTScore computation
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).parent
VENV_PYTHON = str(PROJ / ".venv" / "bin" / "python")


def run_step(name, script, args=None):
    """Run a Python script as a subprocess."""
    cmd = [VENV_PYTHON, str(script)]
    if args:
        cmd.extend(args)
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJ), capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: {name} returned code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bertscore", action="store_true",
                        help="Skip BERTScore computation (slow)")
    args = parser.parse_args()

    print("="*60)
    print("FULL ANALYSIS PIPELINE")
    print("="*60)

    # Step 1: Merge all runs
    run_step("Merge all runs", PROJ / "analysis" / "merge_all_runs.py")

    # Step 2: Compute metrics
    run_step("Compute all metrics", PROJ / "analysis" / "compute_metrics.py")

    # Step 3: Generate tables
    run_step("Generate LaTeX tables", PROJ / "analysis" / "generate_tables_v2.py")

    # Step 4: Generate figures
    run_step("Generate figures", PROJ / "analysis" / "generate_figures_v2.py")

    # Step 5: Run ablation study
    run_step("Protocol ablation study", PROJ / "analysis" / "ablation_study.py")

    # Step 6: Statistical tests
    run_step("Statistical tests", PROJ / "analysis" / "statistical_tests.py")

    print(f"\n{'='*60}")
    print("ALL ANALYSIS STEPS COMPLETE")
    print(f"{'='*60}")
    print(f"  Runs merged:    outputs/all_runs.json")
    print(f"  Full analysis:  analysis/full_analysis.json")
    print(f"  LaTeX tables:   analysis/latex_tables_v2.tex")
    print(f"  Figures:        analysis/figures/")
    print(f"  Ablation:       analysis/ablation_results.json")


if __name__ == "__main__":
    main()
