#!/usr/bin/env python3
"""Experiment runner for Claude (Anthropic API).

Runs the same experimental conditions as the other models using 10
representative abstracts to enable cross-model reproducibility comparison.

Skips any run whose output file already exists.

Usage:
    python run_claude.py                                    # Default (Sonnet 4.5)
    python run_claude.py --model claude-sonnet-4-5-20250929  # Specific model
    python run_claude.py --abstracts 30                     # All 30 abstracts
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.protocol.logger import RunLogger
from src.protocol.run_card import RunCard
from src.models import claude_runner
from src.experiments.config import (
    RESEARCHER_ID,
    AFFILIATION,
    N_REPS,
    SEEDS,
    TEMPERATURES,
    SUMMARIZATION_PROMPT,
    EXTRACTION_PROMPT,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_N_ABSTRACTS = 10


def load_abstracts(n: int = DEFAULT_N_ABSTRACTS) -> list:
    """Load scientific abstracts (first n)."""
    with open(Path(__file__).parent / "data" / "inputs" / "abstracts.json") as f:
        data = json.load(f)
    return data["abstracts"][:n]


def load_prompt_cards():
    """Load existing prompt cards."""
    pc_dir = OUTPUT_DIR / "prompt_cards"
    sum_card = ext_card = None
    for f in pc_dir.glob("*.json"):
        with open(f) as fp:
            card = json.load(fp)
        if "summarization" in card.get("prompt_id", ""):
            sum_card = card
        elif "extraction" in card.get("prompt_id", ""):
            ext_card = card
    return sum_card, ext_card


def run_exists(run_id: str) -> bool:
    """Check if a run output file already exists."""
    return (OUTPUT_DIR / "runs" / f"{run_id}.json").exists()


def make_run_id(model_name: str, task_id: str, abstract_id: str,
                condition: str, rep: int) -> str:
    """Build a standardized run_id."""
    # Shorten model name for file naming
    short = model_name.replace("claude-", "").replace("-20250929", "")
    run_id = f"{short}_{task_id}_{abstract_id}_{condition}_rep{rep}"
    return run_id.replace(":", "_").replace(" ", "_").replace(".", "_")


def run_single(model_name: str, prompt_text: str, prompt_card_ref: str,
               task_id: str, task_category: str, abstract: dict,
               condition: str, rep: int, temperature: float = 0.0,
               seed=None) -> dict:
    """Run a single experiment with full protocol logging."""
    run_id = make_run_id(model_name, task_id, abstract["id"], condition, rep)

    if run_exists(run_id):
        return None

    inference_params = claude_runner.get_inference_params(
        temperature=temperature, max_tokens=1024,
    )
    # Claude API doesn't support seed parameter, but we log it for protocol
    if seed is not None:
        inference_params["seed"] = seed
        inference_params["seed_note"] = "logged-only-not-sent-to-api"

    model_info = claude_runner.get_model_info(model_name)

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
        result = claude_runner.run_inference(
            prompt=prompt_text,
            input_text=abstract["text"],
            model=model_name,
            temperature=temperature,
            max_tokens=1024,
        )
        output_text = result["output_text"]
        system_logs = json.dumps(
            {k: v for k, v in result.items() if k != "output_text"}, default=str
        )
        errors = []
    except Exception as e:
        output_text = ""
        system_logs = ""
        errors = [str(e)]

    logger.log_output(output_text=output_text, system_logs=system_logs, errors=errors)
    logger.save()

    rc = RunCard(str(OUTPUT_DIR / "run_cards"))
    run_card = rc.create_from_run(logger.run_data, prompt_card_ref=prompt_card_ref)
    rc.save(run_card)

    return logger.run_data


def run_claude_experiments(model_name: str, abstracts: list,
                           sum_card: dict, ext_card: dict) -> list:
    """Run all conditions for Claude."""
    all_runs = []
    tasks = [
        ("summarization", "scientific_summarization", SUMMARIZATION_PROMPT, sum_card),
        ("extraction", "structured_extraction", EXTRACTION_PROMPT, ext_card),
    ]

    n_per_task = len(abstracts) * (N_REPS + N_REPS + len(TEMPERATURES) * 3)
    total = n_per_task * len(tasks)
    done = 0

    for task_id, task_cat, prompt, card in tasks:
        card_ref = f"prompt_card_{card['prompt_id']}_v{card['version'].replace('.', '_')}.json"

        # C1: Fixed seed=42, temp=0 (baseline determinism)
        # Note: Claude API doesn't accept seed, so all C1 runs use identical params
        print(f"\n  {model_name}/{task_id} C1: temp=0.0 (greedy baseline)")
        for abstract in abstracts:
            for rep in range(N_REPS):
                run_data = run_single(
                    model_name=model_name, prompt_text=prompt,
                    prompt_card_ref=card_ref, task_id=task_id,
                    task_category=task_cat, abstract=abstract,
                    condition="C1_fixed_seed", rep=rep,
                    temperature=0.0, seed=SEEDS[0],
                )
                done += 1
                if run_data:
                    all_runs.append(run_data)
                    _print_progress(run_data, done, total)

        # C2: Variable seeds, temp=0
        # Note: seed not sent to API but logged for protocol parity
        print(f"\n  {model_name}/{task_id} C2: temp=0.0 (variable seed logged)")
        for abstract in abstracts:
            for rep, seed in enumerate(SEEDS):
                run_data = run_single(
                    model_name=model_name, prompt_text=prompt,
                    prompt_card_ref=card_ref, task_id=task_id,
                    task_category=task_cat, abstract=abstract,
                    condition="C2_var_seed", rep=rep,
                    temperature=0.0, seed=seed,
                )
                done += 1
                if run_data:
                    all_runs.append(run_data)
                    _print_progress(run_data, done, total)

        # C3: Temperature sweep
        print(f"\n  {model_name}/{task_id} C3: Temperature sweep")
        for abstract in abstracts:
            for temp in TEMPERATURES:
                for rep in range(3):
                    run_data = run_single(
                        model_name=model_name, prompt_text=prompt,
                        prompt_card_ref=card_ref, task_id=task_id,
                        task_category=task_cat, abstract=abstract,
                        condition=f"C3_temp{temp}", rep=rep,
                        temperature=temp, seed=SEEDS[rep],
                    )
                    done += 1
                    if run_data:
                        all_runs.append(run_data)
                        _print_progress(run_data, done, total)

    return all_runs


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


def main():
    parser = argparse.ArgumentParser(description="Claude Experiment Runner")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Claude model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--abstracts", type=int, default=DEFAULT_N_ABSTRACTS,
                        help=f"Number of abstracts (default: {DEFAULT_N_ABSTRACTS})")
    args = parser.parse_args()

    print("=" * 70)
    print("GenAI Reproducibility - Claude (Anthropic API) Experiments")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Model: {args.model}")
    print(f"Abstracts: {args.abstracts}")
    print("=" * 70)

    # Test API connectivity
    print("\nTesting API connectivity...")
    try:
        test = claude_runner.run_inference(
            prompt="Say OK", model=args.model, max_tokens=5, timeout=15
        )
        print(f"  [OK] API works. Model returned: {test.get('model_id_returned', '?')}")
    except Exception as e:
        print(f"\nERROR: Cannot reach Anthropic API: {e}")
        sys.exit(1)

    # Load data
    abstracts = load_abstracts(args.abstracts)
    print(f"\nLoaded {len(abstracts)} abstracts")

    sum_card, ext_card = load_prompt_cards()
    if not sum_card or not ext_card:
        print("ERROR: Prompt cards not found. Run original experiments first.")
        sys.exit(1)

    # Run experiments
    print(f"\n{'=' * 70}")
    print(f"MODEL: {args.model}")
    print("=" * 70)

    start = time.time()
    runs = run_claude_experiments(args.model, abstracts, sum_card, ext_card)
    elapsed = time.time() - start

    # Summary
    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {len(runs)} new runs in {elapsed:.1f}s")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
