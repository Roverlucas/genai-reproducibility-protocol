#!/usr/bin/env python3
"""Compute variability metrics for chat-control experiment vs original completion runs.

Compares LLaMA 3 8B chat-format (/api/chat) vs completion-format (/api/generate)
for the same 10 abstracts under greedy decoding (C1 + C2 combined).
Outputs: EMR, NED, ROUGE-L for each task and format.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.variability import exact_match_all_pairs, edit_distance_stats, rouge_l_scores

RUNS_DIR = ROOT / "outputs" / "runs"
ABSTRACTS = [f"abs_{i:03d}" for i in range(1, 11)]


def load_outputs(mode: str, task: str) -> dict:
    """Load outputs grouped by abstract for a given mode and task.

    mode: 'chat' or 'completion'
    task: 'summarization' or 'extraction'
    Returns: {abs_id: [output_text, ...]}
    """
    groups = defaultdict(list)

    for f in sorted(RUNS_DIR.glob("*.json")):
        name = f.stem

        # Filter by mode
        if mode == "chat":
            if "chat" not in name or "llama" not in name.lower():
                continue
        else:  # completion
            if "chat" in name:
                continue
            if "llama" not in name.lower():
                continue

        # Filter by task
        if task not in name:
            continue

        # Extract abstract ID
        abs_id = None
        for aid in ABSTRACTS:
            if aid in name:
                abs_id = aid
                break
        if abs_id is None:
            continue

        # Only C1 and C2 (greedy conditions)
        if "C1_fixed" not in name and "C2_" not in name:
            continue
        # Exclude C2 from C3 temperature runs
        if "C3_" in name:
            continue

        with open(f) as fp:
            data = json.load(fp)

        output = data.get("output_text", "")
        if output:
            groups[abs_id].append(output)

    return dict(groups)


def compute_metrics_for_groups(groups: dict) -> dict:
    """Compute per-abstract and aggregate metrics."""
    all_emrs = []
    all_neds = []
    all_rouges = []

    for abs_id in sorted(groups.keys()):
        outputs = groups[abs_id]
        if len(outputs) < 2:
            continue

        emr = exact_match_all_pairs(outputs)
        ned = edit_distance_stats(outputs)["normalized_mean"]
        rouge = rouge_l_scores(outputs)["mean"]

        all_emrs.append(emr)
        all_neds.append(ned)
        all_rouges.append(rouge)

    n = len(all_emrs)
    if n == 0:
        return {"emr": 0.0, "ned": 0.0, "rouge_l": 0.0, "n_abstracts": 0}

    return {
        "emr": sum(all_emrs) / n,
        "ned": sum(all_neds) / n,
        "rouge_l": sum(all_rouges) / n,
        "n_abstracts": n,
        "per_abstract_emr": {abs_id: emr for abs_id, emr in zip(sorted(groups.keys()), all_emrs)},
    }


def main():
    results = {}

    for task in ["summarization", "extraction"]:
        for mode in ["completion", "chat"]:
            groups = load_outputs(mode, task)
            print(f"\n{task}/{mode}: {len(groups)} abstracts, "
                  f"outputs per abstract: {[len(v) for v in groups.values()]}")

            metrics = compute_metrics_for_groups(groups)
            results[f"{task}_{mode}"] = metrics

            print(f"  EMR     = {metrics['emr']:.3f}")
            print(f"  NED     = {metrics['ned']:.4f}")
            print(f"  ROUGE-L = {metrics['rouge_l']:.4f}")

    # Print LaTeX table values
    print("\n" + "=" * 60)
    print("LaTeX Table Values (Completion vs Chat):")
    print("=" * 60)

    for task in ["summarization", "extraction"]:
        comp = results[f"{task}_completion"]
        chat = results[f"{task}_chat"]
        print(f"\n{task.title()}:")
        print(f"  EMR:     {comp['emr']:.3f}  vs  {chat['emr']:.3f}")
        print(f"  NED:     {comp['ned']:.4f}  vs  {chat['ned']:.4f}")
        print(f"  ROUGE-L: {comp['rouge_l']:.4f}  vs  {chat['rouge_l']:.4f}")

    # Save results
    out_path = ROOT / "analysis" / "chat_control_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
