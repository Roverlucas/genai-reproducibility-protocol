#!/usr/bin/env python3
"""Main experiment runner for the GenAI reproducibility study.

Executes all experimental conditions across two tasks and two models,
applying the proposed protocol for logging, versioning, and provenance.

Usage:
    python run_experiments.py --llama-only    # Run only LLaMA 3 experiments
    python run_experiments.py --gpt4-only     # Run only GPT-4 experiments
    python run_experiments.py                 # Run all experiments
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.protocol.logger import RunLogger
from src.protocol.prompt_card import PromptCard
from src.protocol.run_card import RunCard
from src.protocol.prov_generator import ProvGenerator
from src.experiments.config import (
    RESEARCHER_ID,
    AFFILIATION,
    LLAMA_MODEL,
    GPT4_MODEL,
    N_REPS,
    SEEDS,
    TEMPERATURES,
    SUMMARIZATION_PROMPT,
    EXTRACTION_PROMPT,
)


OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_abstracts() -> list:
    """Load the 5 scientific abstracts."""
    with open(Path(__file__).parent / "data" / "inputs" / "abstracts.json") as f:
        data = json.load(f)
    return data["abstracts"]


def create_prompt_cards():
    """Create and save Prompt Cards for both experimental prompts."""
    pc = PromptCard(str(OUTPUT_DIR / "prompt_cards"))

    # Summarization prompt card
    sum_card = pc.create(
        prompt_id="summarization_v1",
        prompt_text=SUMMARIZATION_PROMPT,
        task_category="scientific_summarization",
        objective="Generate a concise 3-sentence summary of a scientific abstract, preserving key contributions, methods, and results.",
        designed_by=RESEARCHER_ID,
        version="1.0",
        assumptions=[
            "Input is a well-formed English scientific abstract",
            "Output should be exactly 3 sentences",
            "No external knowledge should be added",
        ],
        limitations=[
            "Designed for single abstracts, not full papers",
            "English-only",
            "May struggle with highly technical domain-specific terminology",
        ],
        target_models=[LLAMA_MODEL, GPT4_MODEL],
        expected_output_format="3 sentences of plain text",
        interaction_regime="single-turn",
    )
    pc.save(sum_card)

    # Extraction prompt card
    ext_card = pc.create(
        prompt_id="extraction_v1",
        prompt_text=EXTRACTION_PROMPT,
        task_category="structured_extraction",
        objective="Extract structured JSON with 5 fields from a scientific abstract.",
        designed_by=RESEARCHER_ID,
        version="1.0",
        assumptions=[
            "Input is a well-formed English scientific abstract",
            "Output must be valid JSON",
            "Fields not present in the abstract should be null",
        ],
        limitations=[
            "Fixed schema - does not capture all possible metadata",
            "English-only",
            "JSON parsing may fail if model adds explanation text",
        ],
        target_models=[LLAMA_MODEL, GPT4_MODEL],
        expected_output_format="JSON object with 5 string fields",
        interaction_regime="single-turn",
    )
    pc.save(ext_card)

    print("[OK] Prompt Cards created")
    return sum_card, ext_card


def run_single_experiment(
    runner_module,
    model_name: str,
    prompt_text: str,
    prompt_card_ref: str,
    task_id: str,
    task_category: str,
    abstract: dict,
    condition: str,
    rep: int,
    temperature: float = 0.0,
    seed=None,
    top_p: float = 1.0,
) -> dict:
    """Run a single experimental run with full protocol logging."""
    run_id = f"{model_name}_{task_id}_{abstract['id']}_{condition}_rep{rep}"
    run_id = run_id.replace(":", "_").replace(" ", "_")

    # Get inference params
    inference_params = runner_module.get_inference_params(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=1024,
    )

    # Get model info
    model_info = runner_module.get_model_info(
        model_name if "llama" in model_name else model_name
    )

    # Initialize logger
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

    # Run inference
    try:
        result = runner_module.run_inference(
            prompt=prompt_text,
            input_text=abstract["text"],
            model=model_name,
            temperature=temperature,
            top_p=top_p,
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

    # Log output
    logger.log_output(
        output_text=output_text,
        system_logs=system_logs,
        errors=errors,
    )

    # Save run record
    filepath = logger.save()

    # Create Run Card
    rc = RunCard(str(OUTPUT_DIR / "run_cards"))
    run_card = rc.create_from_run(logger.run_data, prompt_card_ref=prompt_card_ref)
    rc.save(run_card)

    return logger.run_data


def run_llama_experiments(abstracts: list, sum_card: dict, ext_card: dict) -> list:
    """Run all LLaMA 3 experimental conditions."""
    from src.models import llama_runner

    all_runs = []
    tasks = [
        ("summarization", "scientific_summarization", SUMMARIZATION_PROMPT, sum_card),
        ("extraction", "structured_extraction", EXTRACTION_PROMPT, ext_card),
    ]

    for task_id, task_cat, prompt, card in tasks:
        print(f"\n--- LLaMA 3: Task={task_id} ---")
        card_ref = f"prompt_card_{card['prompt_id']}_v{card['version'].replace('.', '_')}.json"

        # C1: Fixed seed, temperature=0 (deterministic)
        print(f"  Condition C1: Fixed seed, temp=0.0")
        for abstract in abstracts:
            for rep in range(N_REPS):
                seed = SEEDS[0]  # Always same seed
                run_data = run_single_experiment(
                    runner_module=llama_runner,
                    model_name=LLAMA_MODEL,
                    prompt_text=prompt,
                    prompt_card_ref=card_ref,
                    task_id=task_id,
                    task_category=task_cat,
                    abstract=abstract,
                    condition="C1_fixed_seed",
                    rep=rep,
                    temperature=0.0,
                    seed=seed,
                )
                all_runs.append(run_data)
                _print_run_status(run_data)

        # C2: Variable seeds, temperature=0
        print(f"  Condition C2: Variable seeds, temp=0.0")
        for abstract in abstracts:
            for rep, seed in enumerate(SEEDS):
                run_data = run_single_experiment(
                    runner_module=llama_runner,
                    model_name=LLAMA_MODEL,
                    prompt_text=prompt,
                    prompt_card_ref=card_ref,
                    task_id=task_id,
                    task_category=task_cat,
                    abstract=abstract,
                    condition="C2_var_seed",
                    rep=rep,
                    temperature=0.0,
                    seed=seed,
                )
                all_runs.append(run_data)
                _print_run_status(run_data)

        # C3: Variable temperature
        print(f"  Condition C3: Variable temperatures")
        for abstract in abstracts:
            for temp in TEMPERATURES:
                for rep in range(3):  # 3 reps per temperature
                    run_data = run_single_experiment(
                        runner_module=llama_runner,
                        model_name=LLAMA_MODEL,
                        prompt_text=prompt,
                        prompt_card_ref=card_ref,
                        task_id=task_id,
                        task_category=task_cat,
                        abstract=abstract,
                        condition=f"C3_temp{temp}",
                        rep=rep,
                        temperature=temp,
                        seed=SEEDS[rep],
                    )
                    all_runs.append(run_data)
                    _print_run_status(run_data)

    return all_runs


def run_gpt4_experiments(abstracts: list, sum_card: dict, ext_card: dict) -> list:
    """Run all GPT-4 experimental conditions."""
    from src.models import gpt4_runner

    all_runs = []
    tasks = [
        ("summarization", "scientific_summarization", SUMMARIZATION_PROMPT, sum_card),
        ("extraction", "structured_extraction", EXTRACTION_PROMPT, ext_card),
    ]

    for task_id, task_cat, prompt, card in tasks:
        print(f"\n--- GPT-4: Task={task_id} ---")
        card_ref = f"prompt_card_{card['prompt_id']}_v{card['version'].replace('.', '_')}.json"

        # C2: Same params, same session (no seed control from API)
        print(f"  Condition C2: Same params, temp=0.0")
        for abstract in abstracts:
            for rep in range(N_REPS):
                run_data = run_single_experiment(
                    runner_module=gpt4_runner,
                    model_name=GPT4_MODEL,
                    prompt_text=prompt,
                    prompt_card_ref=card_ref,
                    task_id=task_id,
                    task_category=task_cat,
                    abstract=abstract,
                    condition="C2_same_params",
                    rep=rep,
                    temperature=0.0,
                    seed=42,  # GPT-4 supports seed param
                )
                all_runs.append(run_data)
                _print_run_status(run_data)
                time.sleep(1)  # Rate limiting

        # C3: Variable temperature
        print(f"  Condition C3: Variable temperatures")
        for abstract in abstracts:
            for temp in TEMPERATURES:
                for rep in range(3):
                    run_data = run_single_experiment(
                        runner_module=gpt4_runner,
                        model_name=GPT4_MODEL,
                        prompt_text=prompt,
                        prompt_card_ref=card_ref,
                        task_id=task_id,
                        task_category=task_cat,
                        abstract=abstract,
                        condition=f"C3_temp{temp}",
                        rep=rep,
                        temperature=temp,
                        seed=SEEDS[rep],
                    )
                    all_runs.append(run_data)
                    _print_run_status(run_data)
                    time.sleep(1)

    return all_runs


def generate_provenance(all_runs: list):
    """Generate PROV-JSON documents from all runs."""
    prov = ProvGenerator(str(OUTPUT_DIR / "prov"))

    # Individual provenance per run
    for run_data in all_runs:
        doc = prov.generate_from_run(run_data)
        run_id = run_data["run_id"]
        prov.save(doc, f"prov_{run_id}")

    # Merged provenance graph
    merged = prov.generate_multi_run(all_runs)
    prov.save(merged, "provenance_complete")

    print(f"\n[OK] Provenance generated: {len(all_runs)} individual + 1 merged graph")


def save_experiment_summary(llama_runs: list, gpt4_runs: list):
    """Save a summary of all experiments."""
    summary = {
        "experiment_date": datetime.now(timezone.utc).isoformat(),
        "researcher": RESEARCHER_ID,
        "total_runs": len(llama_runs) + len(gpt4_runs),
        "llama_runs": len(llama_runs),
        "gpt4_runs": len(gpt4_runs),
        "tasks": ["summarization", "extraction"],
        "models": [LLAMA_MODEL, GPT4_MODEL],
        "conditions": ["C1_fixed_seed", "C2_var_seed/same_params", "C3_var_temp"],
        "n_abstracts": len(load_abstracts()),
    }

    filepath = OUTPUT_DIR / "experiment_summary.json"
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Experiment summary saved: {filepath}")


def _print_run_status(run_data: dict):
    """Print a brief status line for a completed run."""
    run_id = run_data.get("run_id", "?")
    duration = run_data.get("execution_duration_ms", 0)
    overhead = run_data.get("logging_overhead_ms", 0)
    has_error = len(run_data.get("errors", [])) > 0
    status = "ERROR" if has_error else "OK"
    out_len = len(run_data.get("output_text", ""))
    print(f"    [{status}] {run_id} | {duration:.0f}ms | overhead={overhead:.1f}ms | output={out_len}chars")


def main():
    parser = argparse.ArgumentParser(description="GenAI Reproducibility Experiments")
    parser.add_argument("--llama-only", action="store_true", help="Run only LLaMA experiments")
    parser.add_argument("--gpt4-only", action="store_true", help="Run only GPT-4 experiments")
    args = parser.parse_args()

    print("=" * 60)
    print("GenAI Reproducibility Protocol - Experiment Runner")
    print("=" * 60)

    # Load abstracts
    abstracts = load_abstracts()
    print(f"\nLoaded {len(abstracts)} abstracts")

    # Create Prompt Cards
    sum_card, ext_card = create_prompt_cards()

    llama_runs = []
    gpt4_runs = []

    # Run experiments
    if not args.gpt4_only:
        print("\n" + "=" * 60)
        print("PHASE 1: LLaMA 3 Experiments")
        print("=" * 60)
        llama_runs = run_llama_experiments(abstracts, sum_card, ext_card)

    if not args.llama_only:
        print("\n" + "=" * 60)
        print("PHASE 2: GPT-4 Experiments")
        print("=" * 60)
        gpt4_runs = run_gpt4_experiments(abstracts, sum_card, ext_card)

    # Generate provenance
    all_runs = llama_runs + gpt4_runs
    if all_runs:
        generate_provenance(all_runs)
        save_experiment_summary(llama_runs, gpt4_runs)

    # Save all run data for metrics analysis
    all_runs_path = OUTPUT_DIR / "all_runs.json"
    with open(all_runs_path, "w") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(all_runs)} total runs executed")
    print(f"  LLaMA: {len(llama_runs)} runs")
    print(f"  GPT-4: {len(gpt4_runs)} runs")
    print(f"  All data saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
