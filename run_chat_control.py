#!/usr/bin/env python3
"""Chat-format control experiment for prompt-format confound analysis.

Runs LLaMA 3 via /api/chat (chat template) to compare with the original
/api/generate (completion) results. Uses a subset of 10 abstracts with
conditions C1 (fixed seed) and C2 (variable seeds), both at temp=0.

Design: 10 abstracts × 2 tasks × 2 conditions × 5 reps = 200 runs.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.protocol.logger import RunLogger
from src.protocol.run_card import RunCard
from src.experiments.config import (
    RESEARCHER_ID,
    AFFILIATION,
    LLAMA_MODEL,
    N_REPS,
    SEEDS,
    SUMMARIZATION_PROMPT,
    EXTRACTION_PROMPT,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
CHAT_CONTROL_ABSTRACTS = [f"abs_{i:03d}" for i in range(1, 11)]  # First 10


def load_abstracts():
    with open(Path(__file__).parent / "data" / "inputs" / "abstracts.json") as f:
        data = json.load(f)
    return [a for a in data["abstracts"] if a["id"] in CHAT_CONTROL_ABSTRACTS]


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


def run_single(abstract, task_id, task_category, prompt_text, prompt_card_ref,
               condition, rep, seed, temperature=0.0):
    """Run a single chat-mode experiment."""
    from src.models import llama_runner

    run_id = f"llama3_8b_chat_{task_id}_{abstract['id']}_{condition}_rep{rep}"
    run_id = run_id.replace(":", "_").replace(" ", "_")

    # Skip if exists
    filepath = OUTPUT_DIR / "runs" / f"{run_id}.json"
    if filepath.exists():
        print(f"    [SKIP] {run_id}")
        return None

    inference_params = llama_runner.get_inference_params(
        temperature=temperature, seed=seed, max_tokens=1024,
    )
    inference_params["api_mode"] = "chat"  # Mark as chat mode

    model_info = llama_runner.get_model_info(LLAMA_MODEL)

    logger = RunLogger(str(OUTPUT_DIR / "runs"))
    logger.start_run(
        run_id=run_id,
        task_id=task_id,
        task_category=task_category,
        prompt_text=prompt_text,
        model_name=model_info.get("model_name", LLAMA_MODEL),
        model_version=model_info.get("model_version", "unknown"),
        inference_params=inference_params,
        researcher_id=RESEARCHER_ID,
        affiliation=AFFILIATION,
        input_text=abstract["text"],
        weights_hash=model_info.get("weights_hash", ""),
        model_source=model_info.get("model_source", ""),
    )

    try:
        result = llama_runner.run_inference_chat(
            prompt=prompt_text,
            input_text=abstract["text"],
            model=LLAMA_MODEL,
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

    duration = logger.run_data.get("execution_duration_ms", 0)
    overhead = logger.run_data.get("logging_overhead_ms", 0)
    out_len = len(output_text)
    print(f"    [OK] {run_id} | {duration:.0f}ms | oh={overhead:.1f}ms | out={out_len}c")

    return logger.run_data


def main():
    print("=" * 60)
    print("Chat-Format Control Experiment (LLaMA 3 via /api/chat)")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    abstracts = load_abstracts()
    print(f"Loaded {len(abstracts)} abstracts for control experiment")

    sum_card, ext_card = load_prompt_cards()
    if not sum_card or not ext_card:
        print("ERROR: Prompt cards not found.")
        sys.exit(1)

    tasks = [
        ("summarization", "scientific_summarization", SUMMARIZATION_PROMPT, sum_card),
        ("extraction", "structured_extraction", EXTRACTION_PROMPT, ext_card),
    ]

    all_runs = []
    total = len(abstracts) * len(tasks) * N_REPS * 2  # C1 + C2
    done = 0

    for task_id, task_cat, prompt, card in tasks:
        card_ref = f"prompt_card_{card['prompt_id']}_v{card['version'].replace('.', '_')}.json"

        # C1: Fixed seed (seed=42 for all reps)
        print(f"\n  Chat/{task_id} C1: Fixed seed=42, temp=0.0")
        for abstract in abstracts:
            for rep in range(N_REPS):
                run_data = run_single(
                    abstract, task_id, task_cat, prompt, card_ref,
                    "C1_fixed_seed", rep, seed=SEEDS[0], temperature=0.0,
                )
                done += 1
                if run_data:
                    all_runs.append(run_data)

        # C2: Variable seeds
        print(f"\n  Chat/{task_id} C2: Variable seeds, temp=0.0")
        for abstract in abstracts:
            for rep, seed in enumerate(SEEDS):
                run_data = run_single(
                    abstract, task_id, task_cat, prompt, card_ref,
                    "C2_var_seed", rep, seed=seed, temperature=0.0,
                )
                done += 1
                if run_data:
                    all_runs.append(run_data)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(all_runs)} new chat-mode runs executed ({done} total checked)")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'=' * 60}")

    # Quick analysis
    if all_runs:
        analyze_results(all_runs)


def analyze_results(runs):
    """Quick EMR analysis of chat control results."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in runs:
        task = r.get("task_id", "")
        condition = r.get("run_id", "").split("_")
        # Extract abstract ID
        abs_id = None
        for part_idx, part in enumerate(condition):
            if part.startswith("abs"):
                abs_id = f"{part}_{condition[part_idx+1]}"
                break
        cond = "C1" if "C1_fixed" in r.get("run_id", "") else "C2"
        key = (task, cond, abs_id)
        groups[key].append(r.get("output_text", ""))

    print("\n--- Chat Control: Quick EMR Analysis ---")
    for (task, cond, abs_id), outputs in sorted(groups.items()):
        if len(outputs) < 2:
            continue
        ref = outputs[0]
        matches = sum(1 for o in outputs if o == ref)
        emr = matches / len(outputs)
        if emr < 1.0:
            print(f"  {task}/{cond}/{abs_id}: EMR={emr:.3f} ({matches}/{len(outputs)}) ***")

    # Aggregate EMR
    for task in ["summarization", "extraction"]:
        for cond in ["C1", "C2"]:
            matching_groups = [(k, v) for k, v in groups.items()
                              if k[0] == task and k[1] == cond and len(v) >= 2]
            if not matching_groups:
                continue
            emrs = []
            for (_, _, _), outputs in matching_groups:
                ref = outputs[0]
                emrs.append(1.0 if all(o == ref for o in outputs) else 0.0)
            mean_emr = sum(emrs) / len(emrs) if emrs else 0
            print(f"\n  AGGREGATE {task}/{cond}: EMR={mean_emr:.3f} ({sum(emrs):.0f}/{len(emrs)} abstracts match)")


if __name__ == "__main__":
    main()
