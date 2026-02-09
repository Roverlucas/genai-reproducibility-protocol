#!/usr/bin/env python3
"""Expanded experiment runner for the GenAI reproducibility study.

Runs experiments for:
  1. New abstracts (abs_006 through abs_030) across all existing conditions
  2. GPT-4 C1 condition (fixed seed) for ALL 30 abstracts (new condition)

Skips any run whose output file already exists.

Usage:
    python run_expanded_experiments.py --llama-only
    python run_expanded_experiments.py --gpt4-only
    python run_expanded_experiments.py --gpt4-c1-only  # Only new GPT-4 C1 condition
    python run_expanded_experiments.py                 # Run everything
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
EXISTING_ABSTRACT_IDS = {f"abs_{i:03d}" for i in range(1, 6)}


def load_abstracts():
    with open(Path(__file__).parent / "data" / "inputs" / "abstracts.json") as f:
        data = json.load(f)
    return data["abstracts"]


def run_exists(run_id: str) -> bool:
    """Check if a run output file already exists."""
    filepath = OUTPUT_DIR / "runs" / f"{run_id}.json"
    return filepath.exists()


def load_prompt_cards():
    """Load existing prompt cards."""
    pc_dir = OUTPUT_DIR / "prompt_cards"
    sum_card = None
    ext_card = None
    for f in pc_dir.glob("*.json"):
        with open(f) as fp:
            card = json.load(fp)
        if "summarization" in card.get("prompt_id", ""):
            sum_card = card
        elif "extraction" in card.get("prompt_id", ""):
            ext_card = card
    return sum_card, ext_card


def run_single_experiment(
    runner_module,
    model_name,
    prompt_text,
    prompt_card_ref,
    task_id,
    task_category,
    abstract,
    condition,
    rep,
    temperature=0.0,
    seed=None,
    top_p=1.0,
):
    """Run a single experimental run with full protocol logging."""
    run_id = f"{model_name}_{task_id}_{abstract['id']}_{condition}_rep{rep}"
    run_id = run_id.replace(":", "_").replace(" ", "_")

    # Skip if already exists
    if run_exists(run_id):
        return None

    inference_params = runner_module.get_inference_params(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=1024,
    )

    model_info = runner_module.get_model_info(model_name)

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

    logger.log_output(output_text=output_text, system_logs=system_logs, errors=errors)
    logger.save()

    rc = RunCard(str(OUTPUT_DIR / "run_cards"))
    run_card = rc.create_from_run(logger.run_data, prompt_card_ref=prompt_card_ref)
    rc.save(run_card)

    return logger.run_data


def run_llama_new_abstracts(abstracts, sum_card, ext_card):
    """Run LLaMA experiments for new abstracts only."""
    from src.models import llama_runner

    new_abstracts = [a for a in abstracts if a["id"] not in EXISTING_ABSTRACT_IDS]
    print(f"\nLLaMA 3: Running {len(new_abstracts)} new abstracts")

    all_runs = []
    tasks = [
        ("summarization", "scientific_summarization", SUMMARIZATION_PROMPT, sum_card),
        ("extraction", "structured_extraction", EXTRACTION_PROMPT, ext_card),
    ]

    total = len(new_abstracts) * len(tasks) * (N_REPS + N_REPS + len(TEMPERATURES) * 3)
    done = 0

    for task_id, task_cat, prompt, card in tasks:
        card_ref = f"prompt_card_{card['prompt_id']}_v{card['version'].replace('.', '_')}.json"

        # C1: Fixed seed, temp=0
        print(f"\n  LLaMA/{task_id} C1: Fixed seed, temp=0.0")
        for abstract in new_abstracts:
            for rep in range(N_REPS):
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
                    seed=SEEDS[0],
                )
                done += 1
                if run_data:
                    all_runs.append(run_data)
                    _print_progress(run_data, done, total)

        # C2: Variable seeds, temp=0
        print(f"\n  LLaMA/{task_id} C2: Variable seeds, temp=0.0")
        for abstract in new_abstracts:
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
                done += 1
                if run_data:
                    all_runs.append(run_data)
                    _print_progress(run_data, done, total)

        # C3: Variable temperature
        print(f"\n  LLaMA/{task_id} C3: Variable temperatures")
        for abstract in new_abstracts:
            for temp in TEMPERATURES:
                for rep in range(3):
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
                    done += 1
                    if run_data:
                        all_runs.append(run_data)
                        _print_progress(run_data, done, total)

    return all_runs


def run_gpt4_new_abstracts(abstracts, sum_card, ext_card):
    """Run GPT-4 experiments for new abstracts (C2, C3)."""
    from src.models import gpt4_runner

    new_abstracts = [a for a in abstracts if a["id"] not in EXISTING_ABSTRACT_IDS]
    print(f"\nGPT-4: Running {len(new_abstracts)} new abstracts (C2, C3)")

    all_runs = []
    tasks = [
        ("summarization", "scientific_summarization", SUMMARIZATION_PROMPT, sum_card),
        ("extraction", "structured_extraction", EXTRACTION_PROMPT, ext_card),
    ]

    total = len(new_abstracts) * len(tasks) * (N_REPS + len(TEMPERATURES) * 3)
    done = 0

    for task_id, task_cat, prompt, card in tasks:
        card_ref = f"prompt_card_{card['prompt_id']}_v{card['version'].replace('.', '_')}.json"

        # C2: Same params, temp=0
        print(f"\n  GPT-4/{task_id} C2: Same params, temp=0.0")
        for abstract in new_abstracts:
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
                    seed=42,
                )
                done += 1
                if run_data:
                    all_runs.append(run_data)
                    _print_progress(run_data, done, total)
                    time.sleep(3)

        # C3: Variable temperature
        print(f"\n  GPT-4/{task_id} C3: Variable temperatures")
        for abstract in new_abstracts:
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
                    done += 1
                    if run_data:
                        all_runs.append(run_data)
                        _print_progress(run_data, done, total)
                        time.sleep(3)

    return all_runs


def run_gpt4_c1_all(abstracts, sum_card, ext_card):
    """Run the NEW GPT-4 C1 condition (fixed seed) for ALL 30 abstracts."""
    from src.models import gpt4_runner

    print(f"\nGPT-4 C1 (NEW): Running {len(abstracts)} abstracts with fixed seed")

    all_runs = []
    tasks = [
        ("summarization", "scientific_summarization", SUMMARIZATION_PROMPT, sum_card),
        ("extraction", "structured_extraction", EXTRACTION_PROMPT, ext_card),
    ]

    total = len(abstracts) * len(tasks) * N_REPS
    done = 0

    for task_id, task_cat, prompt, card in tasks:
        card_ref = f"prompt_card_{card['prompt_id']}_v{card['version'].replace('.', '_')}.json"

        print(f"\n  GPT-4/{task_id} C1: Fixed seed=42, temp=0.0")
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
                    condition="C1_fixed_seed",
                    rep=rep,
                    temperature=0.0,
                    seed=42,
                )
                done += 1
                if run_data:
                    all_runs.append(run_data)
                    _print_progress(run_data, done, total)
                    time.sleep(3)

    return all_runs


def _print_progress(run_data, done, total):
    run_id = run_data.get("run_id", "?")
    duration = run_data.get("execution_duration_ms", 0)
    overhead = run_data.get("logging_overhead_ms", 0)
    has_error = len(run_data.get("errors", [])) > 0
    status = "ERROR" if has_error else "OK"
    out_len = len(run_data.get("output_text", ""))
    pct = (done / total * 100) if total > 0 else 0
    print(f"    [{status}] ({done}/{total} {pct:.0f}%) {run_id} | {duration:.0f}ms | oh={overhead:.1f}ms | out={out_len}c")


def merge_all_runs():
    """Merge all individual run files into all_runs.json."""
    runs_dir = OUTPUT_DIR / "runs"
    all_runs = []
    for f in sorted(runs_dir.glob("*.json")):
        with open(f) as fp:
            all_runs.append(json.load(fp))

    with open(OUTPUT_DIR / "all_runs.json", "w") as fp:
        json.dump(all_runs, fp, indent=2, ensure_ascii=False)

    llama = sum(1 for r in all_runs if "llama" in r.get("model_name", "").lower())
    gpt4 = sum(1 for r in all_runs if "gpt" in r.get("model_name", "").lower())
    print(f"\n[OK] Merged {len(all_runs)} runs: LLaMA={llama}, GPT-4={gpt4}")
    return all_runs


def main():
    parser = argparse.ArgumentParser(description="Expanded GenAI Experiments")
    parser.add_argument("--llama-only", action="store_true")
    parser.add_argument("--gpt4-only", action="store_true")
    parser.add_argument("--gpt4-c1-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("GenAI Reproducibility - Expanded Experiments")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    abstracts = load_abstracts()
    print(f"Loaded {len(abstracts)} abstracts (5 original + {len(abstracts)-5} new)")

    sum_card, ext_card = load_prompt_cards()
    if not sum_card or not ext_card:
        print("ERROR: Prompt cards not found. Run original experiments first.")
        sys.exit(1)

    llama_runs = []
    gpt4_runs = []
    gpt4_c1_runs = []

    if args.gpt4_c1_only:
        gpt4_c1_runs = run_gpt4_c1_all(abstracts, sum_card, ext_card)
    else:
        if not args.gpt4_only:
            print("\n" + "=" * 60)
            print("PHASE A: LLaMA 3 - New Abstracts")
            print("=" * 60)
            llama_runs = run_llama_new_abstracts(abstracts, sum_card, ext_card)

        if not args.llama_only:
            print("\n" + "=" * 60)
            print("PHASE B: GPT-4 - New Abstracts (C2, C3)")
            print("=" * 60)
            gpt4_runs = run_gpt4_new_abstracts(abstracts, sum_card, ext_card)

            print("\n" + "=" * 60)
            print("PHASE C: GPT-4 - C1 Fixed Seed (ALL 30 abstracts)")
            print("=" * 60)
            gpt4_c1_runs = run_gpt4_c1_all(abstracts, sum_card, ext_card)

    # Merge all runs
    print("\n" + "=" * 60)
    print("Merging all runs...")
    all_runs = merge_all_runs()

    total_new = len(llama_runs) + len(gpt4_runs) + len(gpt4_c1_runs)
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {total_new} new runs executed")
    print(f"  LLaMA new:    {len(llama_runs)}")
    print(f"  GPT-4 C2/C3:  {len(gpt4_runs)}")
    print(f"  GPT-4 C1:     {len(gpt4_c1_runs)}")
    print(f"  Total merged:  {len(all_runs)}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
