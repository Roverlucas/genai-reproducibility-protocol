#!/usr/bin/env python3
"""Unified analysis for all expanded experiments.

Processes runs from all models (LLaMA 3 8B, Mistral 7B, Gemma 2 9B, GPT-4,
Claude Sonnet 4.5) and all scenarios (single-turn extraction/summarization,
multi-turn, RAG), computing reproducibility metrics across all conditions.

Computes per-abstract metrics then averages across abstracts per
(model, task, condition) — the correct approach for EMR.

Outputs:
  - analysis/expanded_metrics.json: Full metrics for all model/condition combos
  - analysis/cross_model_comparison.json: Cross-model reproducibility comparison
  - Console summary tables

Usage:
    python analysis/analyze_expanded.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.variability import compute_all_metrics
from metrics.validation import json_validity_rate, schema_compliance_rate, field_level_accuracy

RUNS_DIR = Path(__file__).parent.parent / "outputs" / "runs"
ANALYSIS_DIR = Path(__file__).parent


def identify_model(run_data: dict) -> str:
    """Identify model group from run data."""
    model_name = run_data.get("model_name", "").lower()
    run_id = run_data.get("run_id", "").lower()

    if "mistral" in model_name or "mistral" in run_id:
        return "mistral_7b"
    elif "gemma" in model_name or "gemma" in run_id:
        return "gemma2_9b"
    elif "claude" in model_name or "sonnet" in run_id:
        return "claude_sonnet"
    elif "gpt" in model_name:
        return "gpt4"
    elif "llama" in model_name or "llama" in run_id:
        return "llama3_8b"
    return "unknown"


def identify_task(run_data: dict) -> str:
    """Identify task from run data."""
    task_id = run_data.get("task_id", "").lower()
    run_id = run_data.get("run_id", "").lower()

    if "multiturn" in task_id or "refinement" in task_id:
        return "multiturn_refinement"
    elif "rag" in task_id:
        return "rag_extraction"
    elif "extraction" in task_id or "extraction" in run_id:
        return "extraction"
    elif "summarization" in task_id or "summarization" in run_id:
        return "summarization"
    return "unknown"


def identify_condition(run_data: dict) -> str:
    """Identify experimental condition from run data."""
    run_id = run_data.get("run_id", "")
    parts = run_id.split("_")
    for i, part in enumerate(parts):
        if part.startswith("C1") or part.startswith("C2") or part.startswith("C3"):
            cond_parts = [part]
            for j in range(i + 1, len(parts)):
                if parts[j].startswith("rep"):
                    break
                cond_parts.append(parts[j])
            return "_".join(cond_parts)
    return "unknown"


def identify_abstract(run_data: dict) -> str:
    """Extract abstract ID from run data."""
    run_id = run_data.get("run_id", "")
    parts = run_id.split("_")
    for i, part in enumerate(parts):
        if part == "abs" and i + 1 < len(parts):
            return f"abs_{parts[i + 1]}"
    return "unknown"


def load_all_runs() -> List[dict]:
    """Load all run records."""
    runs = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                runs.append(json.load(fp))
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")
    return runs


def group_runs(runs: List[dict]) -> Dict[tuple, List[dict]]:
    """Group runs by (model, task, condition, abstract)."""
    groups = defaultdict(list)
    for run in runs:
        model = identify_model(run)
        task = identify_task(run)
        condition = identify_condition(run)
        abstract = identify_abstract(run)
        groups[(model, task, condition, abstract)].append(run)
    return groups


def compute_group_metrics(runs: List[dict], scorer=None) -> dict:
    """Compute all reproducibility metrics for a group of runs (same abstract)."""
    outputs = [r.get("output_text", "") for r in runs]
    # Filter out empty outputs (e.g. API timeouts)
    outputs = [o for o in outputs if o.strip()]
    n = len(outputs)

    if n < 2:
        return {"n_runs": n, "insufficient_data": True}

    # Core variability metrics
    variability = compute_all_metrics(outputs, scorer=scorer)

    # For extraction tasks, also compute JSON-specific metrics
    task = identify_task(runs[0])
    json_metrics = {}
    if "extraction" in task or "rag" in task or "refinement" in task:
        json_metrics = {
            "validity": json_validity_rate(outputs),
            "compliance": schema_compliance_rate(outputs),
            "field_accuracy": field_level_accuracy(outputs),
        }

    # Timing stats
    durations = [r.get("execution_duration_ms", 0) for r in runs if r.get("execution_duration_ms")]
    overheads = [r.get("logging_overhead_ms", 0) for r in runs if r.get("logging_overhead_ms")]

    return {
        "n_runs": n,
        "variability": variability,
        "json_metrics": json_metrics,
        "timing": {
            "mean_duration_ms": sum(durations) / len(durations) if durations else 0,
            "mean_overhead_ms": sum(overheads) / len(overheads) if overheads else 0,
            "overhead_pct": (
                (sum(overheads) / sum(durations) * 100)
                if durations and sum(durations) > 0
                else 0
            ),
        },
    }


def compute_model_task_summary(groups: Dict[tuple, List[dict]]) -> List[dict]:
    """Compute summary metrics per (model, task, condition).

    Correctly: compute per-abstract metrics first, then average across abstracts.
    This avoids cross-abstract comparisons that would deflate EMR.
    """
    # First, load BERTScore scorer once
    try:
        from bert_score import BERTScorer
        scorer = BERTScorer(lang="en", rescale_with_baseline=False)
        print("BERTScorer loaded successfully")
    except Exception as e:
        print(f"Warning: BERTScorer unavailable ({e}), skipping BERTScore")
        scorer = None

    # Collect per-abstract metrics keyed by (model, task, condition)
    mtc_abstract_metrics = defaultdict(list)
    total_groups = len(groups)
    for idx, ((model, task, condition, abstract), runs) in enumerate(sorted(groups.items())):
        if idx % 50 == 0:
            print(f"  Processing group {idx+1}/{total_groups}...", flush=True)
        metrics = compute_group_metrics(runs, scorer=scorer)
        if not metrics.get("insufficient_data"):
            mtc_abstract_metrics[(model, task, condition)].append(metrics)

    # Average across abstracts
    results = []
    for (model, task, condition), abstract_metrics_list in sorted(mtc_abstract_metrics.items()):
        if not abstract_metrics_list:
            continue

        # Collect per-abstract values
        emrs = []
        neds = []
        rouges = []
        bert_f1s = []
        n_runs_total = 0
        durations = []
        overheads = []
        overhead_pcts = []

        for am in abstract_metrics_list:
            var = am.get("variability", {})
            emrs.append(var.get("exact_match_rate", 0))

            ed = var.get("edit_distance", {})
            ned_val = ed.get("normalized_mean")
            if ned_val is not None:
                neds.append(ned_val)

            rl = var.get("rouge_l", {})
            rouge_val = rl.get("mean")
            if rouge_val is not None:
                rouges.append(rouge_val)

            bs = var.get("bert_score", {})
            bs_val = bs.get("bertscore_f1_mean")
            if bs_val is not None:
                bert_f1s.append(bs_val)

            n_runs_total += am.get("n_runs", 0)

            t = am.get("timing", {})
            if t.get("mean_duration_ms"):
                durations.append(t["mean_duration_ms"])
            if t.get("mean_overhead_ms"):
                overheads.append(t["mean_overhead_ms"])
            if t.get("overhead_pct"):
                overhead_pcts.append(t["overhead_pct"])

        results.append({
            "model": model,
            "task": task,
            "condition": condition,
            "n_abstracts": len(abstract_metrics_list),
            "n_runs": n_runs_total,
            "emr_mean": float(np.mean(emrs)) if emrs else None,
            "emr_std": float(np.std(emrs)) if emrs else None,
            "ned_mean": float(np.mean(neds)) if neds else None,
            "rouge_l_mean": float(np.mean(rouges)) if rouges else None,
            "bertscore_f1_mean": float(np.mean(bert_f1s)) if bert_f1s else None,
            "timing": {
                "mean_duration_ms": float(np.mean(durations)) if durations else 0,
                "mean_overhead_ms": float(np.mean(overheads)) if overheads else 0,
                "overhead_pct": float(np.mean(overhead_pcts)) if overhead_pcts else 0,
            },
            "per_abstract": abstract_metrics_list,
        })

    return results


def compute_cross_model_comparison(results: List[dict]) -> dict:
    """Build cross-model comparison summary."""
    comparison = {}

    for r in results:
        model = r["model"]
        task = r["task"]
        condition = r["condition"]

        emr = r.get("emr_mean")
        if emr is None:
            continue

        key = f"{task}/{condition}"
        if key not in comparison:
            comparison[key] = {}
        comparison[key][model] = {
            "emr": emr,
            "emr_std": r.get("emr_std", 0),
            "n_runs": r["n_runs"],
            "n_abstracts": r.get("n_abstracts", 0),
            "ned": r.get("ned_mean"),
            "rouge_l": r.get("rouge_l_mean"),
            "bertscore_f1": r.get("bertscore_f1_mean"),
        }

    return comparison


def print_summary(results: List[dict], comparison: dict):
    """Print formatted summary tables."""
    print("\n" + "=" * 110)
    print("EXPANDED EXPERIMENTS ANALYSIS — CROSS-MODEL REPRODUCIBILITY")
    print("=" * 110)

    # Table 1: EMR by model x task x condition
    print(f"\n{'Model':<14} {'Task':<24} {'Condition':<18} {'N':>5} {'Abs':>4} "
          f"{'EMR':>8} {'NED':>8} {'ROUGE-L':>8} {'BERT-F1':>8}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: (x["model"], x["task"], x["condition"])):
        emr = r.get("emr_mean")
        ned = r.get("ned_mean")
        rouge = r.get("rouge_l_mean")
        bert = r.get("bertscore_f1_mean")

        emr_s = f"{emr:.4f}" if emr is not None else "N/A"
        ned_s = f"{ned:.4f}" if ned is not None else "N/A"
        rouge_s = f"{rouge:.4f}" if rouge is not None else "N/A"
        bert_s = f"{bert:.4f}" if bert is not None else "N/A"

        print(f"{r['model']:<14} {r['task']:<24} {r['condition']:<18} "
              f"{r['n_runs']:>5} {r.get('n_abstracts', 0):>4} "
              f"{emr_s:>8} {ned_s:>8} {rouge_s:>8} {bert_s:>8}")

    # Table 2: Cross-model comparison for greedy conditions
    print("\n" + "=" * 110)
    print("CROSS-MODEL COMPARISON (C1 Fixed Seed / C2 Variable Seed)")
    print("=" * 110)

    for key in sorted(comparison.keys()):
        if "C3" in key:
            continue
        models = comparison[key]
        print(f"\n  {key}:")
        for model, metrics in sorted(models.items()):
            ned_s = f"{metrics['ned']:.4f}" if metrics.get('ned') is not None else "N/A"
            bert_s = f"{metrics['bertscore_f1']:.4f}" if metrics.get('bertscore_f1') is not None else "N/A"
            print(f"    {model:<14}: EMR={metrics['emr']:.4f}, NED={ned_s}, "
                  f"BERT-F1={bert_s}, N={metrics['n_runs']}")

    # Count totals
    total_runs = sum(r["n_runs"] for r in results)
    models_seen = set(r["model"] for r in results)
    tasks_seen = set(r["task"] for r in results)

    print(f"\n{'=' * 110}")
    print(f"TOTAL: {total_runs} runs across {len(models_seen)} models, "
          f"{len(tasks_seen)} tasks")
    print(f"Models: {', '.join(sorted(models_seen))}")
    print(f"Tasks: {', '.join(sorted(tasks_seen))}")
    print("=" * 110)


def main():
    print("=" * 110)
    print("EXPANDED EXPERIMENTS — UNIFIED ANALYSIS")
    print("=" * 110)

    # Load all runs
    runs = load_all_runs()
    print(f"\nLoaded {len(runs)} total runs")

    # Count by model
    model_counts = defaultdict(int)
    for r in runs:
        model_counts[identify_model(r)] += 1
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count} runs")

    # Group and compute
    groups = group_runs(runs)
    print(f"\n{len(groups)} unique (model, task, condition, abstract) groups")

    results = compute_model_task_summary(groups)
    comparison = compute_cross_model_comparison(results)

    # Save results (without per_abstract detail for smaller file)
    results_slim = []
    for r in results:
        slim = {k: v for k, v in r.items() if k != "per_abstract"}
        results_slim.append(slim)

    output_file = ANALYSIS_DIR / "expanded_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results_slim, f, indent=2, default=str)
    print(f"\nMetrics saved: {output_file}")

    comparison_file = ANALYSIS_DIR / "cross_model_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"Comparison saved: {comparison_file}")

    # Print summary
    print_summary(results, comparison)


if __name__ == "__main__":
    main()
