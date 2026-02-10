#!/usr/bin/env python3
"""Experiment runner for new API models: DeepSeek and Perplexity.

Runs Task 1 (extraction) and Task 2 (summarization) under condition C1
(greedy decoding, temperature=0.0, 5 repetitions) for 10 abstracts.

Total per model: 10 abstracts × 5 reps × 2 tasks = 100 runs.
Grand total: 2 models × 100 = 200 runs.

Skips any run whose output file already exists.

Usage:
    python run_new_api_models.py                     # Run both models
    python run_new_api_models.py --model deepseek    # DeepSeek only
    python run_new_api_models.py --model perplexity  # Perplexity only
    python run_new_api_models.py --abstracts 5       # Fewer abstracts
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.protocol.logger import RunLogger
from src.protocol.run_card import RunCard
from src.protocol.hasher import hash_text
from src.models import deepseek_runner, perplexity_runner
from src.experiments.config import (
    RESEARCHER_ID,
    AFFILIATION,
    SEEDS,
    SUMMARIZATION_PROMPT,
    EXTRACTION_PROMPT,
    DEEPSEEK_MODEL,
    PERPLEXITY_MODEL,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
DEFAULT_N_ABSTRACTS = 10
N_REPS = 5
MAX_RETRIES = 3
RETRY_DELAY = 5


def load_abstracts(n: int = DEFAULT_N_ABSTRACTS) -> list:
    with open(Path(__file__).parent / "data" / "inputs" / "abstracts.json") as f:
        data = json.load(f)
    return data["abstracts"][:n]


def load_prompt_cards():
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
    return (OUTPUT_DIR / "runs" / f"{run_id}.json").exists()


def run_single(runner_module, model_name, prompt_text, prompt_card_ref,
               task_id, task_category, abstract, condition, rep,
               temperature=0.0, seed=None):
    """Run a single experimental run with full protocol logging."""
    run_id = (
        f"{model_name}_{task_id}_{abstract['id']}_{condition}_rep{rep}"
        .replace(":", "_").replace(" ", "_")
    )

    if run_exists(run_id):
        return None

    inference_params = runner_module.get_inference_params(
        temperature=temperature,
        top_p=1.0,
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
        model_name=model_info["model_name"],
        model_version=model_info["model_version"],
        inference_params=inference_params,
        researcher_id=RESEARCHER_ID,
        affiliation=AFFILIATION,
        input_text=abstract["text"],
        weights_hash=model_info.get("weights_hash", ""),
        model_source=model_info.get("model_source", ""),
    )

    for attempt in range(MAX_RETRIES):
        try:
            result = runner_module.run_inference(
                prompt=prompt_text,
                input_text=abstract["text"],
                model=model_name,
                temperature=temperature,
                top_p=1.0,
                seed=seed,
                max_tokens=1024,
            )
            output_text = result["output_text"]
            system_logs = json.dumps(
                {k: v for k, v in result.items() if k != "output_text"}, default=str
            )
            errors = []
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    Retry {attempt+1}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                output_text = ""
                system_logs = ""
                errors = [traceback.format_exc()]

    logger.log_output(output_text=output_text, system_logs=system_logs, errors=errors)
    logger.save()

    rc = RunCard(str(OUTPUT_DIR / "run_cards"))
    run_card = rc.create_from_run(logger.run_data, prompt_card_ref=prompt_card_ref)
    rc.save(run_card)

    return logger.run_data


def run_model_experiments(runner_module, model_name, abstracts, sum_card, ext_card):
    """Run all C1 experiments for a single model."""
    tasks = [
        ("extraction", "information_extraction", EXTRACTION_PROMPT, ext_card),
        ("summarization", "text_summarization", SUMMARIZATION_PROMPT, sum_card),
    ]

    total = len(abstracts) * N_REPS * len(tasks)
    completed = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Abstracts: {len(abstracts)}, Reps: {N_REPS}, Tasks: {len(tasks)}")
    print(f"Total runs: {total}")
    print(f"{'='*60}")

    for task_id, task_cat, prompt, card in tasks:
        for abstract in abstracts:
            for rep in range(N_REPS):
                run_id = (
                    f"{model_name}_{task_id}_{abstract['id']}_C1_fixed_seed_rep{rep}"
                    .replace(":", "_").replace(" ", "_")
                )
                if run_exists(run_id):
                    skipped += 1
                    continue

                seed = SEEDS[rep]
                print(f"  [{completed+skipped+1}/{total}] {task_id} | "
                      f"{abstract['id']} | rep{rep} (seed={seed})")

                result = run_single(
                    runner_module=runner_module,
                    model_name=model_name,
                    prompt_text=prompt,
                    prompt_card_ref=card.get("prompt_id", "") if card else "",
                    task_id=task_id,
                    task_category=task_cat,
                    abstract=abstract,
                    condition="C1_fixed_seed",
                    rep=rep,
                    temperature=0.0,
                    seed=seed,
                )

                completed += 1
                # Rate limiting: 1 second between calls
                time.sleep(1.0)

    print(f"\nDone: {completed} new, {skipped} skipped")
    return completed


def main():
    parser = argparse.ArgumentParser(description="Run new API model experiments")
    parser.add_argument("--model", choices=["deepseek", "perplexity", "both"],
                        default="both", help="Which model(s) to run")
    parser.add_argument("--abstracts", type=int, default=DEFAULT_N_ABSTRACTS,
                        help="Number of abstracts to use")
    args = parser.parse_args()

    abstracts = load_abstracts(args.abstracts)
    sum_card, ext_card = load_prompt_cards()

    print(f"Loaded {len(abstracts)} abstracts")
    print(f"Prompt cards: sum={'yes' if sum_card else 'no'}, ext={'yes' if ext_card else 'no'}")

    total_new = 0

    if args.model in ("deepseek", "both"):
        n = run_model_experiments(
            deepseek_runner, DEEPSEEK_MODEL, abstracts, sum_card, ext_card
        )
        total_new += n

    if args.model in ("perplexity", "both"):
        n = run_model_experiments(
            perplexity_runner, PERPLEXITY_MODEL, abstracts, sum_card, ext_card
        )
        total_new += n

    print(f"\n{'='*60}")
    print(f"GRAND TOTAL: {total_new} new runs completed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
