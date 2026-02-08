#!/usr/bin/env python3
"""Compute all metrics from experimental runs and generate analysis report.

Processes all runs (LLaMA 3 + GPT-4) across 2 tasks, multiple conditions,
and 5 abstracts to produce variability metrics, overhead metrics, and
aggregated statistics per model.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.variability import compute_all_metrics
from src.metrics.overhead import (
    compute_logging_overhead,
    compute_storage_overhead,
    compute_overhead_ratio,
    compute_directory_size,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
ANALYSIS_DIR = Path(__file__).parent


def load_all_runs():
    """Load all run records."""
    with open(OUTPUT_DIR / "all_runs.json") as f:
        return json.load(f)


def parse_condition(run_id):
    """Parse condition from run_id."""
    if "C1_fixed_seed" in run_id:
        return "C1"
    elif "C2_var_seed" in run_id or "C2_same_params" in run_id:
        return "C2"
    elif "C3_temp0.0" in run_id:
        return "C3_t0.0"
    elif "C3_temp0.3" in run_id:
        return "C3_t0.3"
    elif "C3_temp0.7" in run_id:
        return "C3_t0.7"
    return "unknown"


def parse_model(run):
    """Parse model from run record."""
    model = run.get("model_name", "")
    if "llama" in model.lower():
        return "llama3_8b"
    elif "gpt" in model.lower():
        return "gpt4"
    return "unknown"


def parse_abstract(run_id):
    """Parse abstract id from run_id."""
    for abs_id in ["abs_001", "abs_002", "abs_003", "abs_004", "abs_005"]:
        if abs_id in run_id:
            return abs_id
    return "unknown"


def group_runs(all_runs):
    """Group runs by (model, task, condition) and by (model, task, condition, abstract)."""
    by_model_task_condition = defaultdict(list)
    by_model_task_condition_abstract = defaultdict(list)

    for run in all_runs:
        task = run["task_id"]
        run_id = run["run_id"]
        model = parse_model(run)
        condition = parse_condition(run_id)
        abstract = parse_abstract(run_id)

        by_model_task_condition[(model, task, condition)].append(run)
        by_model_task_condition_abstract[(model, task, condition, abstract)].append(run)

    return by_model_task_condition, by_model_task_condition_abstract


def compute_variability_analysis(by_mtca):
    """Compute variability metrics for each (model, task, condition, abstract) group."""
    results = {}

    for key, runs in sorted(by_mtca.items()):
        model, task, condition, abstract = key
        outputs = [r["output_text"] for r in runs if r["output_text"]]

        if len(outputs) < 2:
            continue

        metrics = compute_all_metrics(outputs)
        metrics["model"] = model
        metrics["task"] = task
        metrics["condition"] = condition
        metrics["abstract"] = abstract
        results[f"{model}_{task}_{condition}_{abstract}"] = metrics

    return results


def compute_aggregated_variability(variability_results):
    """Aggregate variability metrics per (model, task, condition) across all abstracts."""
    import numpy as np

    aggregated = defaultdict(lambda: {
        "exact_match_rates": [],
        "edit_dist_means": [],
        "edit_dist_normalized_means": [],
        "rouge_l_means": [],
        "avg_lengths_chars": [],
        "avg_lengths_words": [],
    })

    for key, metrics in variability_results.items():
        group_key = (metrics["model"], metrics["task"], metrics["condition"])
        agg = aggregated[group_key]
        agg["exact_match_rates"].append(metrics["exact_match_rate"])
        agg["edit_dist_means"].append(metrics["edit_distance"]["mean"])
        agg["edit_dist_normalized_means"].append(metrics["edit_distance"]["normalized_mean"])
        agg["rouge_l_means"].append(metrics["rouge_l"]["mean"])
        agg["avg_lengths_chars"].append(metrics["avg_output_length_chars"])
        agg["avg_lengths_words"].append(metrics["avg_output_length_words"])

    summary = {}
    for (model, task, condition), agg in sorted(aggregated.items()):
        summary[f"{model}_{task}_{condition}"] = {
            "model": model,
            "task": task,
            "condition": condition,
            "n_abstracts": len(agg["exact_match_rates"]),
            "exact_match_rate": {
                "mean": float(np.mean(agg["exact_match_rates"])),
                "std": float(np.std(agg["exact_match_rates"])),
            },
            "edit_distance_normalized": {
                "mean": float(np.mean(agg["edit_dist_normalized_means"])),
                "std": float(np.std(agg["edit_dist_normalized_means"])),
            },
            "edit_distance_raw": {
                "mean": float(np.mean(agg["edit_dist_means"])),
                "std": float(np.std(agg["edit_dist_means"])),
            },
            "rouge_l": {
                "mean": float(np.mean(agg["rouge_l_means"])),
                "std": float(np.std(agg["rouge_l_means"])),
            },
            "avg_output_length_chars": {
                "mean": float(np.mean(agg["avg_lengths_chars"])),
                "std": float(np.std(agg["avg_lengths_chars"])),
            },
            "avg_output_length_words": {
                "mean": float(np.mean(agg["avg_lengths_words"])),
                "std": float(np.std(agg["avg_lengths_words"])),
            },
        }

    return summary


def compute_overhead_analysis(all_runs):
    """Compute overhead metrics for the entire experiment."""
    logging_oh = compute_logging_overhead(all_runs)
    storage_oh = compute_storage_overhead(all_runs)
    ratio_oh = compute_overhead_ratio(all_runs)

    # Directory sizes
    runs_dir_size = compute_directory_size(str(OUTPUT_DIR / "runs"))
    prov_dir_size = compute_directory_size(str(OUTPUT_DIR / "prov"))
    cards_dir_size = compute_directory_size(str(OUTPUT_DIR / "run_cards"))
    prompt_cards_size = compute_directory_size(str(OUTPUT_DIR / "prompt_cards"))
    total_dir_size = compute_directory_size(str(OUTPUT_DIR))

    return {
        "logging_overhead": logging_oh,
        "storage_overhead": storage_oh,
        "overhead_ratio": ratio_oh,
        "directory_sizes": {
            "runs": runs_dir_size,
            "provenance": prov_dir_size,
            "run_cards": cards_dir_size,
            "prompt_cards": prompt_cards_size,
            "total_output": total_dir_size,
        },
    }


def compute_execution_time_analysis(all_runs):
    """Analyze execution times by model, task and condition."""
    import numpy as np

    by_group = defaultdict(list)
    for run in all_runs:
        task = run["task_id"]
        run_id = run["run_id"]
        model = parse_model(run)
        condition = parse_condition(run_id)
        by_group[(model, task, condition)].append(run["execution_duration_ms"])

    results = {}
    for (model, task, condition), durations in sorted(by_group.items()):
        results[f"{model}_{task}_{condition}"] = {
            "model": model,
            "task": task,
            "condition": condition,
            "n_runs": len(durations),
            "mean_ms": float(np.mean(durations)),
            "std_ms": float(np.std(durations)),
            "min_ms": float(np.min(durations)),
            "max_ms": float(np.max(durations)),
            "median_ms": float(np.median(durations)),
        }

    return results


def main():
    print("=" * 60)
    print("GenAI Reproducibility Protocol - Metrics Analysis")
    print("=" * 60)

    # Load data
    all_runs = load_all_runs()
    print(f"\nLoaded {len(all_runs)} run records")

    # Group runs
    by_mtc, by_mtca = group_runs(all_runs)
    print(f"Groups by (model, task, condition): {len(by_mtc)}")
    print(f"Groups by (model, task, condition, abstract): {len(by_mtca)}")

    # Print group sizes
    print("\n--- Run Counts per Group ---")
    for key, runs in sorted(by_mtc.items()):
        print(f"  {key[0]:12s} | {key[1]:15s} | {key[2]:10s} | {len(runs)} runs")

    # Variability analysis
    print("\n--- Computing Variability Metrics ---")
    var_results = compute_variability_analysis(by_mtca)

    # Aggregated variability
    agg_var = compute_aggregated_variability(var_results)

    print("\n--- Aggregated Variability (mean across 5 abstracts) ---")
    print(f"{'Model':<12} {'Task':<15} {'Condition':<10} {'ExactMatch':>12} {'EditDist(norm)':>15} {'ROUGE-L':>10}")
    print("-" * 80)
    for key, v in sorted(agg_var.items()):
        print(
            f"{v['model']:<12} {v['task']:<15} {v['condition']:<10} "
            f"{v['exact_match_rate']['mean']:>11.3f}  "
            f"{v['edit_distance_normalized']['mean']:>14.4f}  "
            f"{v['rouge_l']['mean']:>9.4f}"
        )

    # Overhead analysis
    print("\n--- Computing Overhead Metrics ---")
    overhead = compute_overhead_analysis(all_runs)
    oh = overhead["logging_overhead"]
    print(f"  Logging overhead: mean={oh['mean_ms']:.2f}ms, std={oh['std_ms']:.2f}ms")
    print(f"  Total logging overhead: {oh['total_ms']:.1f}ms across {oh['n_runs']} runs")

    sr = overhead["overhead_ratio"]
    print(f"  Overhead ratio: mean={sr['mean_percent']:.3f}%, max={sr['max_percent']:.3f}%")

    ds = overhead["directory_sizes"]
    print(f"  Run files: {ds['runs']['file_count']} files, {ds['runs']['total_kb']:.1f} KB")
    print(f"  Provenance: {ds['provenance']['file_count']} files, {ds['provenance']['total_kb']:.1f} KB")
    print(f"  Run Cards: {ds['run_cards']['file_count']} files, {ds['run_cards']['total_kb']:.1f} KB")
    print(f"  Total output: {ds['total_output']['file_count']} files, {ds['total_output']['total_mb']:.2f} MB")

    # Execution time analysis
    print("\n--- Execution Time Analysis ---")
    exec_times = compute_execution_time_analysis(all_runs)
    print(f"{'Model':<12} {'Task':<15} {'Condition':<10} {'N':>4} {'Mean(ms)':>10} {'Std(ms)':>10} {'Median(ms)':>11}")
    print("-" * 80)
    for key, v in sorted(exec_times.items()):
        print(
            f"{v['model']:<12} {v['task']:<15} {v['condition']:<10} "
            f"{v['n_runs']:>4} "
            f"{v['mean_ms']:>10.1f} "
            f"{v['std_ms']:>10.1f} "
            f"{v['median_ms']:>11.1f}"
        )

    # Save full analysis
    full_analysis = {
        "n_total_runs": len(all_runs),
        "variability_per_abstract": var_results,
        "variability_aggregated": agg_var,
        "overhead": overhead,
        "execution_times": exec_times,
    }

    output_path = ANALYSIS_DIR / "full_analysis.json"
    with open(output_path, "w") as f:
        json.dump(full_analysis, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Full analysis saved to: {output_path}")

    return full_analysis


if __name__ == "__main__":
    main()
