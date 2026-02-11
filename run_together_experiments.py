#!/usr/bin/env python3
"""Experiment runner for LLaMA 3 8B via Together AI (cloud-served).

Runs single-turn extraction and summarization under conditions C1 and C2
to enable direct comparison with the locally-served LLaMA 3 8B (Ollama).

Design rationale:
  - Same model family (LLaMA 3 8B) served locally vs cloud
  - Same prompts, same abstracts, same seeds, same temperature
  - If local is deterministic and cloud is not → infrastructure is the causal variable

Conditions:
  C1: fixed seed=42, temp=0.0, 5 repetitions per abstract
  C2: variable seeds=[42,123,456,789,1024], temp=0.0, 1 run each

Total: 10 abstracts × 2 tasks × (5 C1 + 5 C2) = 200 runs

Usage:
    python run_together_experiments.py                        # All experiments
    python run_together_experiments.py --task extraction      # Only extraction
    python run_together_experiments.py --task summarization   # Only summarization
    python run_together_experiments.py --condition C1         # Only C1
    python run_together_experiments.py --abstracts 5          # Fewer abstracts
"""

import argparse
import json
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.protocol.logger import RunLogger
from src.protocol.run_card import RunCard
from src.models import together_runner
from src.experiments.config import (
    RESEARCHER_ID,
    AFFILIATION,
    SEEDS,
    SUMMARIZATION_PROMPT,
    EXTRACTION_PROMPT,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"

DEFAULT_MODEL = together_runner.DEFAULT_MODEL
DEFAULT_N_ABSTRACTS = 10
N_REPS = 5
MAX_RETRIES = 3
DELAY_BETWEEN_CALLS = 1.0  # seconds, to respect rate limits


# ─── Retry Helper ────────────────────────────────────────────────────────────

def _api_call_with_retry(call_fn, max_retries: int = MAX_RETRIES) -> dict:
    """Execute an API call with exponential backoff on failure."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return call_fn()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError, json.JSONDecodeError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"      [RETRY] Attempt {attempt + 1}/{max_retries} "
                      f"failed: {e}. Waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"      [FAIL] All {max_retries} attempts exhausted: {e}",
                      flush=True)
    raise last_error


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_abstracts(n: int = DEFAULT_N_ABSTRACTS) -> list:
    """Load scientific abstracts (first n)."""
    with open(Path(__file__).parent / "data" / "inputs" / "abstracts.json") as f:
        data = json.load(f)
    return data["abstracts"][:n]


def run_exists(run_id: str) -> bool:
    """Check if a run output file already exists."""
    return (OUTPUT_DIR / "runs" / f"{run_id}.json").exists()


def make_run_id(model_tag: str, task_id: str, abstract_id: str,
                condition: str, rep: int) -> str:
    """Build a standardized run_id."""
    run_id = f"{model_tag}_{task_id}_{abstract_id}_{condition}_rep{rep}"
    return run_id.replace("/", "_").replace(":", "_").replace(" ", "_")


# ─── Single Experiment Run ───────────────────────────────────────────────────

def run_single_experiment(
    model_name: str,
    model_tag: str,
    prompt_text: str,
    task_id: str,
    task_category: str,
    abstract: dict,
    condition: str,
    rep: int,
    temperature: float = 0.0,
    seed: int = None,
) -> dict:
    """Run a single extraction or summarization experiment via Together AI."""
    run_id = make_run_id(model_tag, task_id, abstract["id"], condition, rep)

    if run_exists(run_id):
        return None

    inference_params = together_runner.get_inference_params(
        temperature=temperature,
        seed=seed,
        max_tokens=1024,
    )

    model_info = together_runner.get_model_info(model_name)

    logger = RunLogger(str(OUTPUT_DIR / "runs"))
    logger.start_run(
        run_id=run_id,
        task_id=task_id,
        task_category=task_category,
        prompt_text=prompt_text,
        model_name=model_info.get("model_name", model_name),
        model_version=model_info.get("model_version", "unknown"),
        inference_params=inference_params,
        researcher_id=RESEARCHER_ID,
        affiliation=AFFILIATION,
        input_text=abstract["text"],
        weights_hash=model_info.get("weights_hash", ""),
        model_source=model_info.get("model_source", ""),
    )

    try:
        result = _api_call_with_retry(
            lambda: together_runner.run_inference(
                prompt=prompt_text,
                input_text=abstract["text"],
                model=model_name,
                temperature=temperature,
                seed=seed,
                max_tokens=1024,
            )
        )

        output_text = result["output_text"]
        system_logs = json.dumps(
            {k: v for k, v in result.items() if k != "output_text"},
            default=str,
        )
        errors = []

    except Exception as e:
        output_text = ""
        system_logs = json.dumps({"error": str(e)}, default=str)
        errors = [str(e)]

    logger.log_output(output_text=output_text, system_logs=system_logs, errors=errors)
    logger.save()

    rc = RunCard(str(OUTPUT_DIR / "run_cards"))
    prompt_card_ref = (
        "prompt_card_extraction_v1_0.json" if "extraction" in task_id
        else "prompt_card_summarization_v1_0.json"
    )
    run_card = rc.create_from_run(logger.run_data, prompt_card_ref=prompt_card_ref)
    rc.save(run_card)

    return logger.run_data


# ─── Progress Printing ───────────────────────────────────────────────────────

def _print_progress(run_data, done, total):
    run_id = run_data.get("run_id", "?")
    duration = run_data.get("execution_duration_ms", 0)
    overhead = run_data.get("logging_overhead_ms", 0)
    has_error = len(run_data.get("errors", [])) > 0
    status = "ERR" if has_error else "OK"
    out_len = len(run_data.get("output_text", ""))
    pct = (done / total * 100) if total > 0 else 0
    print(f"    [{status}] ({done}/{total} {pct:.0f}%) {run_id} | "
          f"{duration:.0f}ms | oh={overhead:.1f}ms | out={out_len}c",
          flush=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Together AI LLaMA 3 8B Experiment Runner (Tasks 1 & 2)"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--abstracts", type=int, default=DEFAULT_N_ABSTRACTS,
                        help=f"Number of abstracts (default: {DEFAULT_N_ABSTRACTS})")
    parser.add_argument("--task", type=str, default=None,
                        choices=["extraction", "summarization"],
                        help="Run only this task (default: both)")
    parser.add_argument("--condition", type=str, default=None,
                        choices=["C1", "C2"],
                        help="Run only this condition (default: both)")
    args = parser.parse_args()

    tasks_to_run = [args.task] if args.task else ["extraction", "summarization"]
    conditions = [args.condition] if args.condition else ["C1", "C2"]

    # Compute total expected runs
    n_per_condition = args.abstracts * N_REPS
    total_expected = n_per_condition * len(tasks_to_run) * len(conditions)

    # Model tag for filenames (no slashes)
    model_tag = "together_llama3_8b"

    print("=" * 70)
    print("GenAI Reproducibility - Together AI LLaMA 3 8B (Cloud-Served)")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Model: {args.model}")
    print(f"Model tag: {model_tag}")
    print(f"Abstracts: {args.abstracts}")
    print(f"Tasks: {tasks_to_run}")
    print(f"Conditions: {conditions}")
    print(f"Total expected runs: {total_expected}")
    print("=" * 70)

    # Test API connectivity
    print("\nTesting API connectivity...")
    try:
        test = together_runner.run_inference(
            prompt="Say OK",
            model=args.model,
            max_tokens=5,
            seed=42,
            timeout=15,
        )
        print(f"  [OK] API works. Model: {test.get('model_id_returned', '?')}")
        print(f"  System fingerprint: {test.get('system_fingerprint', 'N/A')}")
    except Exception as e:
        print(f"\nERROR: Cannot reach Together AI API: {e}")
        sys.exit(1)

    # Load data
    abstracts = load_abstracts(args.abstracts)
    print(f"\nLoaded {len(abstracts)} abstracts")

    # Task configs
    task_configs = {
        "extraction": {
            "task_id": "extraction",
            "task_category": "structured_extraction",
            "prompt_text": EXTRACTION_PROMPT,
        },
        "summarization": {
            "task_id": "summarization",
            "task_category": "scientific_summarization",
            "prompt_text": SUMMARIZATION_PROMPT,
        },
    }

    all_runs = []
    start = time.time()

    for task_name in tasks_to_run:
        cfg = task_configs[task_name]
        print(f"\n{'=' * 70}")
        print(f"TASK: {task_name.upper()}")
        print("=" * 70)

        # Condition C1: fixed seed=42, temp=0.0, 5 reps
        if "C1" in conditions:
            print(f"\n  --- Condition C1: fixed seed=42, temp=0.0, {N_REPS} reps ---")
            total = len(abstracts) * N_REPS
            done = 0
            for abstract in abstracts:
                for rep in range(N_REPS):
                    run_data = run_single_experiment(
                        model_name=args.model,
                        model_tag=model_tag,
                        prompt_text=cfg["prompt_text"],
                        task_id=cfg["task_id"],
                        task_category=cfg["task_category"],
                        abstract=abstract,
                        condition="C1_fixed_seed",
                        rep=rep,
                        temperature=0.0,
                        seed=SEEDS[0],  # 42
                    )
                    done += 1
                    if run_data:
                        all_runs.append(run_data)
                        _print_progress(run_data, done, total)
                        time.sleep(DELAY_BETWEEN_CALLS)
                    else:
                        run_id = make_run_id(model_tag, cfg["task_id"],
                                             abstract["id"], "C1_fixed_seed", rep)
                        pct = (done / total * 100) if total > 0 else 0
                        print(f"    [SKIP] ({done}/{total} {pct:.0f}%) {run_id}",
                              flush=True)

        # Condition C2: variable seeds, temp=0.0
        if "C2" in conditions:
            print(f"\n  --- Condition C2: variable seeds, temp=0.0, {N_REPS} seeds ---")
            total = len(abstracts) * N_REPS
            done = 0
            for abstract in abstracts:
                for rep, seed in enumerate(SEEDS):
                    run_data = run_single_experiment(
                        model_name=args.model,
                        model_tag=model_tag,
                        prompt_text=cfg["prompt_text"],
                        task_id=cfg["task_id"],
                        task_category=cfg["task_category"],
                        abstract=abstract,
                        condition="C2_variable_seeds",
                        rep=rep,
                        temperature=0.0,
                        seed=seed,
                    )
                    done += 1
                    if run_data:
                        all_runs.append(run_data)
                        _print_progress(run_data, done, total)
                        time.sleep(DELAY_BETWEEN_CALLS)
                    else:
                        run_id = make_run_id(model_tag, cfg["task_id"],
                                             abstract["id"], "C2_variable_seeds", rep)
                        pct = (done / total * 100) if total > 0 else 0
                        print(f"    [SKIP] ({done}/{total} {pct:.0f}%) {run_id}",
                              flush=True)

    elapsed = time.time() - start

    # Summary
    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {len(all_runs)} new runs in {elapsed:.1f}s")
    ext_count = sum(1 for r in all_runs if "extraction" in r.get("task_id", ""))
    sum_count = sum(1 for r in all_runs if "summarization" in r.get("task_id", ""))
    err_count = sum(1 for r in all_runs if len(r.get("errors", [])) > 0)
    print(f"  Extraction:     {ext_count}")
    print(f"  Summarization:  {sum_count}")
    if err_count > 0:
        print(f"  Errors:         {err_count}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
