#!/usr/bin/env python3
"""Experiment runner for expanded model coverage (Mistral 7B + Gemma 2 9B).

Runs the same experimental conditions as the original study but with
two additional open-weight models via Ollama, using 10 representative
abstracts to validate cross-model reproducibility patterns.

Skips any run whose output file already exists.

Usage:
    python run_new_models.py                          # Both models
    python run_new_models.py --model mistral:7b       # Mistral only
    python run_new_models.py --model gemma2:9b        # Gemma 2 only
    python run_new_models.py --abstracts 30           # All 30 abstracts
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
from src.models import llama_runner  # Works with any Ollama model
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

# New models to test (all served via Ollama)
NEW_MODELS = ["mistral:7b", "gemma2:9b"]

# Number of abstracts to use (10 representative subset by default)
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
    run_id = f"{model_name}_{task_id}_{abstract_id}_{condition}_rep{rep}"
    return run_id.replace(":", "_").replace(" ", "_")


def run_single(model_name: str, prompt_text: str, prompt_card_ref: str,
               task_id: str, task_category: str, abstract: dict,
               condition: str, rep: int, temperature: float = 0.0,
               seed=None) -> dict:
    """Run a single experiment with full protocol logging."""
    run_id = make_run_id(model_name, task_id, abstract["id"], condition, rep)

    if run_exists(run_id):
        return None

    inference_params = llama_runner.get_inference_params(
        temperature=temperature, seed=seed, max_tokens=1024,
    )
    model_info = llama_runner.get_model_info(model_name)

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
        result = llama_runner.run_inference(
            prompt=prompt_text,
            input_text=abstract["text"],
            model=model_name,
            temperature=temperature,
            seed=seed,
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


def run_model_experiments(model_name: str, abstracts: list,
                          sum_card: dict, ext_card: dict) -> list:
    """Run all conditions for a single model."""
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

        # C1: Fixed seed, temp=0 (baseline determinism)
        print(f"\n  {model_name}/{task_id} C1: Fixed seed=42, temp=0.0")
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

        # C2: Variable seeds, temp=0 (seed sensitivity)
        print(f"\n  {model_name}/{task_id} C2: Variable seeds, temp=0.0")
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

        # C3: Temperature sweep (variability by sampling)
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
          f"{duration:.0f}ms | oh={overhead:.1f}ms | out={out_len}c")


def verify_model_available(model_name: str) -> bool:
    """Check if a model is available in Ollama."""
    try:
        info = llama_runner.get_model_info(model_name)
        return "error" not in info
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="New Models Experiment Runner")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (e.g., mistral:7b)")
    parser.add_argument("--abstracts", type=int, default=DEFAULT_N_ABSTRACTS,
                        help=f"Number of abstracts to use (default: {DEFAULT_N_ABSTRACTS})")
    args = parser.parse_args()

    models = [args.model] if args.model else NEW_MODELS

    print("=" * 70)
    print("GenAI Reproducibility - Expanded Model Experiments")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Models: {models}")
    print(f"Abstracts: {args.abstracts}")
    print("=" * 70)

    # Verify models are available
    for model in models:
        if not verify_model_available(model):
            print(f"\nERROR: Model '{model}' not available in Ollama.")
            print(f"  Pull it first: ollama pull {model}")
            sys.exit(1)
        print(f"  [OK] {model} available")

    # Load data
    abstracts = load_abstracts(args.abstracts)
    print(f"\nLoaded {len(abstracts)} abstracts")

    sum_card, ext_card = load_prompt_cards()
    if not sum_card or not ext_card:
        print("ERROR: Prompt cards not found. Run original experiments first.")
        sys.exit(1)

    # Run experiments for each model
    all_new_runs = []
    for model in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model}")
        print("=" * 70)

        start = time.time()
        runs = run_model_experiments(model, abstracts, sum_card, ext_card)
        elapsed = time.time() - start

        all_new_runs.extend(runs)
        print(f"\n  {model}: {len(runs)} new runs in {elapsed:.1f}s")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {len(all_new_runs)} new runs")
    for model in models:
        count = sum(1 for r in all_new_runs
                    if model.replace(":", "_") in r.get("run_id", ""))
        print(f"  {model}: {count} runs")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
