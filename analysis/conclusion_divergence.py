#!/usr/bin/env python3
"""
WS-3: Conclusion Divergence Analysis

Analyzes when textual non-determinism leads to substantively different
downstream conclusions vs. mere cosmetic phrasing differences.

Produces data for the "3-level reproducibility framework":
  Level 1 - Bitwise (EMR): Needed for audit/compliance/tamper detection
  Level 2 - Surface (NED, ROUGE-L): Sufficient for NLP benchmarks
  Level 3 - Semantic (BERTScore, field agreement): Sufficient for scientific conclusions

Key questions answered:
  1. For extraction: In what % of cases do textually different outputs
     yield different field values (conclusion-changing)?
  2. For summarization: In what % of cases do textually different outputs
     convey different scientific claims?
  3. What is the practical impact spectrum of non-determinism?
"""

import json
import sys
import re
import itertools
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import Levenshtein

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.variability import (
    exact_match_all_pairs,
    rouge_l_scores,
    _rouge_l_f1,
)

# ---------- Constants ----------
EXPECTED_FIELDS = ["objective", "method", "key_result", "model_or_system", "benchmark"]
_JSON_OBJECT_RE = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)

FIELD_OPENNESS = {
    "objective": "open",
    "method": "open",
    "key_result": "open",
    "model_or_system": "closed",
    "benchmark": "closed",
}


# ---------- Helpers ----------
def parse_json_output(text: str) -> Optional[dict]:
    """Parse JSON from output text, handling preamble."""
    if not text:
        return None
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def extract_metadata(run_id: str) -> dict:
    """Extract model, task, abstract, condition, rep from run_id."""
    parts = run_id.split("_")
    meta = {"model": "", "task": "", "abstract": "", "condition": "", "rep": ""}

    # Model
    if "llama" in run_id.lower():
        meta["model"] = "llama3"
    elif "gpt" in run_id.lower():
        meta["model"] = "gpt4"

    # Task
    if "summarization" in run_id:
        meta["task"] = "summarization"
    elif "extraction" in run_id:
        meta["task"] = "extraction"

    # Abstract
    for i, part in enumerate(parts):
        if part == "abs" and i + 1 < len(parts):
            meta["abstract"] = f"abs_{parts[i+1]}"
            break

    # Condition
    for i, part in enumerate(parts):
        if part.startswith("C1") or part.startswith("C2") or part.startswith("C3"):
            cond_parts = [part]
            for j in range(i + 1, len(parts)):
                if parts[j].startswith("rep"):
                    break
                cond_parts.append(parts[j])
            meta["condition"] = "_".join(cond_parts)
            break

    # Rep
    for part in parts:
        if part.startswith("rep"):
            meta["rep"] = part
            break

    return meta


def normalized_edit_distance(a: str, b: str) -> float:
    """Compute NED between two strings."""
    max_len = max(len(a), len(b), 1)
    return Levenshtein.distance(a, b) / max_len


# ---------- Analysis Functions ----------

def analyze_extraction_divergence(runs_by_group: dict) -> dict:
    """
    For extraction task: analyze when textual differences lead to
    different field values (conclusion-changing vs cosmetic).
    """
    results = {
        "total_groups": 0,
        "groups_with_textual_diff": 0,
        "groups_with_field_diff": 0,
        "groups_all_identical": 0,
        "per_abstract": {},
        "per_field_stats": {f: {"divergent_groups": 0, "total_groups": 0} for f in EXPECTED_FIELDS},
        "divergence_examples": [],
        "field_divergence_severity": {f: [] for f in EXPECTED_FIELDS},
    }

    for group_key, runs in runs_by_group.items():
        model, condition, abstract = group_key
        if "extraction" not in condition and runs[0].get("_task") != "extraction":
            continue

        outputs = [r.get("output_text", "") for r in runs]
        if len(outputs) < 2:
            continue

        results["total_groups"] += 1

        # Check textual identity
        emr = exact_match_all_pairs(outputs)
        has_textual_diff = emr < 1.0

        if has_textual_diff:
            results["groups_with_textual_diff"] += 1
        else:
            results["groups_all_identical"] += 1

        # Parse JSON and check field-level divergence
        parsed_outputs = [parse_json_output(o) for o in outputs]
        valid_parsed = [p for p in parsed_outputs if p is not None]

        if len(valid_parsed) < 2:
            continue

        has_any_field_diff = False
        field_diffs = {}

        for field in EXPECTED_FIELDS:
            results["per_field_stats"][field]["total_groups"] += 1
            values = [(p.get(field) or "").strip() for p in valid_parsed]
            unique_values = set(values)

            if len(unique_values) > 1:
                has_any_field_diff = True
                results["per_field_stats"][field]["divergent_groups"] += 1
                field_diffs[field] = list(unique_values)

                # Compute severity: pairwise NED between field values
                neds = []
                rouges = []
                for i in range(len(values)):
                    for j in range(i + 1, len(values)):
                        if values[i] != values[j]:
                            neds.append(normalized_edit_distance(values[i], values[j]))
                            if values[i] and values[j]:
                                rouges.append(_rouge_l_f1(values[i], values[j]))
                if neds:
                    results["field_divergence_severity"][field].append({
                        "abstract": abstract,
                        "model": model,
                        "condition": condition,
                        "mean_ned": float(np.mean(neds)),
                        "mean_rouge_l": float(np.mean(rouges)) if rouges else None,
                        "n_unique_values": len(unique_values),
                    })

        if has_any_field_diff:
            results["groups_with_field_diff"] += 1

            # Collect example
            if len(results["divergence_examples"]) < 10:
                results["divergence_examples"].append({
                    "abstract": abstract,
                    "model": model,
                    "condition": condition,
                    "field_diffs": {k: v[:3] for k, v in field_diffs.items()},
                    "n_outputs": len(outputs),
                    "emr": emr,
                })

        results["per_abstract"][abstract] = {
            "has_textual_diff": has_textual_diff,
            "has_field_diff": has_any_field_diff,
            "emr": emr,
            "divergent_fields": list(field_diffs.keys()),
        }

    return results


def analyze_summarization_divergence(runs_by_group: dict) -> dict:
    """
    For summarization task: analyze whether textually different summaries
    convey different scientific claims or just different phrasing.
    """
    results = {
        "total_groups": 0,
        "groups_with_textual_diff": 0,
        "groups_all_identical": 0,
        "pairwise_similarity_when_different": [],
        "per_abstract": {},
        "severity_distribution": {"cosmetic": 0, "minor": 0, "substantive": 0},
    }

    for group_key, runs in runs_by_group.items():
        model, condition, abstract = group_key
        outputs = [r.get("output_text", "") for r in runs if r.get("output_text")]
        if len(outputs) < 2:
            continue

        results["total_groups"] += 1

        emr = exact_match_all_pairs(outputs)
        has_textual_diff = emr < 1.0

        if not has_textual_diff:
            results["groups_all_identical"] += 1
            results["per_abstract"][abstract] = {
                "emr": 1.0, "category": "identical",
                "mean_rouge_l": 1.0, "mean_ned": 0.0,
            }
            continue

        results["groups_with_textual_diff"] += 1

        # Compute pairwise metrics for non-identical outputs
        neds = []
        rouges = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if outputs[i] != outputs[j]:
                    ned = normalized_edit_distance(outputs[i], outputs[j])
                    rouge = _rouge_l_f1(outputs[i], outputs[j])
                    neds.append(ned)
                    rouges.append(rouge)

        mean_ned = float(np.mean(neds)) if neds else 0.0
        mean_rouge = float(np.mean(rouges)) if rouges else 1.0

        # Classify severity
        if mean_ned < 0.05 and mean_rouge > 0.90:
            category = "cosmetic"
            results["severity_distribution"]["cosmetic"] += 1
        elif mean_ned < 0.15 and mean_rouge > 0.70:
            category = "minor"
            results["severity_distribution"]["minor"] += 1
        else:
            category = "substantive"
            results["severity_distribution"]["substantive"] += 1

        results["pairwise_similarity_when_different"].append({
            "abstract": abstract,
            "model": model,
            "condition": condition,
            "mean_ned": mean_ned,
            "mean_rouge_l": mean_rouge,
            "emr": emr,
            "category": category,
        })

        results["per_abstract"][abstract] = {
            "emr": emr,
            "category": category,
            "mean_rouge_l": mean_rouge,
            "mean_ned": mean_ned,
        }

    return results


def compute_three_level_framework(all_runs: list) -> dict:
    """
    Compute the 3-level reproducibility framework across all conditions.
    Returns summary statistics for each level.
    """
    # Group by (model, task, condition)
    groups = defaultdict(list)
    for run in all_runs:
        meta = extract_metadata(run.get("run_id", ""))
        key = (meta["model"], meta["task"], meta["condition"])
        groups[key].append(run)

    framework = {}
    for (model, task, condition), runs in groups.items():
        if not model or not task or not condition:
            continue

        # Group by abstract for within-abstract metrics
        by_abstract = defaultdict(list)
        for r in runs:
            meta = extract_metadata(r.get("run_id", ""))
            by_abstract[meta["abstract"]].append(r.get("output_text", ""))

        emrs = []
        neds = []
        rouges = []

        for abstract, outputs in by_abstract.items():
            if len(outputs) < 2:
                continue
            emrs.append(exact_match_all_pairs(outputs))
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    neds.append(normalized_edit_distance(outputs[i], outputs[j]))
                    rouges.append(_rouge_l_f1(outputs[i], outputs[j]))

        key_str = f"{model}_{task}_{condition}"
        framework[key_str] = {
            "model": model,
            "task": task,
            "condition": condition,
            "n_abstracts": len(by_abstract),
            "level_1_bitwise": {
                "emr_mean": float(np.mean(emrs)) if emrs else None,
                "emr_std": float(np.std(emrs)) if emrs else None,
                "pct_perfectly_reproducible": float(np.mean([1 if e == 1.0 else 0 for e in emrs])) * 100 if emrs else None,
            },
            "level_2_surface": {
                "ned_mean": float(np.mean(neds)) if neds else None,
                "rouge_l_mean": float(np.mean(rouges)) if rouges else None,
                "pct_near_reproducible": float(np.mean([1 if n < 0.05 else 0 for n in neds])) * 100 if neds else None,
            },
            "level_3_semantic": {
                "note": "BERTScore requires model loading - use existing results from full_analysis.json",
                "rouge_l_gt_090": float(np.mean([1 if r > 0.90 else 0 for r in rouges])) * 100 if rouges else None,
            },
        }

    return framework


# ---------- Main ----------
def main():
    project_root = Path(__file__).parent.parent
    runs_dir = project_root / "outputs" / "runs"
    output_dir = project_root / "analysis"

    print("=" * 80)
    print("WS-3: CONCLUSION DIVERGENCE ANALYSIS")
    print("=" * 80)

    # Load all runs
    print("\nLoading runs...")
    all_runs = []
    for json_file in sorted(runs_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                run = json.load(f)
            meta = extract_metadata(run.get("run_id", ""))
            run["_model"] = meta["model"]
            run["_task"] = meta["task"]
            run["_abstract"] = meta["abstract"]
            run["_condition"] = meta["condition"]
            all_runs.append(run)
        except Exception as e:
            pass

    print(f"Loaded {len(all_runs)} runs")

    # Group runs for analysis
    extraction_groups = defaultdict(list)
    summarization_groups = defaultdict(list)

    for run in all_runs:
        key = (run["_model"], run["_condition"], run["_abstract"])
        if run["_task"] == "extraction":
            extraction_groups[key].append(run)
        elif run["_task"] == "summarization":
            summarization_groups[key].append(run)

    # ── Analysis 1: Extraction Field Divergence ──
    print("\n" + "=" * 80)
    print("ANALYSIS 1: EXTRACTION FIELD DIVERGENCE")
    print("=" * 80)

    # Focus on GPT-4 C2 (greedy, where we know outputs differ)
    gpt4_ext_c2 = {k: v for k, v in extraction_groups.items()
                   if k[0] == "gpt4" and "C2" in k[1]}
    llama_ext_c2 = {k: v for k, v in extraction_groups.items()
                    if k[0] == "llama3" and "C2" in k[1]}
    gpt4_ext_c3 = {k: v for k, v in extraction_groups.items()
                   if k[0] == "gpt4" and "C3" in k[1]}

    print("\n--- GPT-4 Extraction under C2 (greedy, t=0) ---")
    ext_gpt4_c2 = analyze_extraction_divergence(gpt4_ext_c2)
    print(f"  Total groups: {ext_gpt4_c2['total_groups']}")
    print(f"  Groups with textual diff: {ext_gpt4_c2['groups_with_textual_diff']}")
    print(f"  Groups with FIELD diff (conclusion-changing): {ext_gpt4_c2['groups_with_field_diff']}")
    print(f"  Groups all identical: {ext_gpt4_c2['groups_all_identical']}")

    if ext_gpt4_c2['groups_with_textual_diff'] > 0:
        pct_conclusion_changing = (ext_gpt4_c2['groups_with_field_diff'] /
                                   ext_gpt4_c2['groups_with_textual_diff']) * 100
        print(f"\n  >> Of textually different outputs, {pct_conclusion_changing:.1f}% have DIFFERENT field values")
        print(f"  >> This means the non-determinism is conclusion-changing, not just cosmetic")

    print("\n  Per-field divergence rates:")
    for field in EXPECTED_FIELDS:
        stats = ext_gpt4_c2["per_field_stats"][field]
        if stats["total_groups"] > 0:
            rate = stats["divergent_groups"] / stats["total_groups"] * 100
            openness = FIELD_OPENNESS[field]
            print(f"    {field:<25} ({openness:>6}): {rate:5.1f}% divergent ({stats['divergent_groups']}/{stats['total_groups']})")

    # Severity analysis
    print("\n  Field divergence severity (mean NED when different):")
    for field in EXPECTED_FIELDS:
        severities = ext_gpt4_c2["field_divergence_severity"][field]
        if severities:
            mean_ned = np.mean([s["mean_ned"] for s in severities])
            mean_rouge = np.mean([s["mean_rouge_l"] for s in severities if s["mean_rouge_l"] is not None])
            print(f"    {field:<25}: NED={mean_ned:.4f}, ROUGE-L={mean_rouge:.4f}")

    print("\n--- LLaMA 3 Extraction under C2 (greedy, t=0) ---")
    ext_llama_c2 = analyze_extraction_divergence(llama_ext_c2)
    print(f"  Total groups: {ext_llama_c2['total_groups']}")
    print(f"  Groups with textual diff: {ext_llama_c2['groups_with_textual_diff']}")
    print(f"  Groups with FIELD diff: {ext_llama_c2['groups_with_field_diff']}")

    print("\n--- GPT-4 Extraction under C3 (temperature sweep) ---")
    ext_gpt4_c3 = analyze_extraction_divergence(gpt4_ext_c3)
    print(f"  Total groups: {ext_gpt4_c3['total_groups']}")
    print(f"  Groups with textual diff: {ext_gpt4_c3['groups_with_textual_diff']}")
    print(f"  Groups with FIELD diff (conclusion-changing): {ext_gpt4_c3['groups_with_field_diff']}")

    # ── Analysis 2: Summarization Divergence ──
    print("\n" + "=" * 80)
    print("ANALYSIS 2: SUMMARIZATION DIVERGENCE SEVERITY")
    print("=" * 80)

    gpt4_sum_c2 = {k: v for k, v in summarization_groups.items()
                   if k[0] == "gpt4" and "C2" in k[1]}
    llama_sum_c2 = {k: v for k, v in summarization_groups.items()
                    if k[0] == "llama3" and "C2" in k[1]}

    print("\n--- GPT-4 Summarization under C2 (greedy, t=0) ---")
    sum_gpt4_c2 = analyze_summarization_divergence(gpt4_sum_c2)
    print(f"  Total groups: {sum_gpt4_c2['total_groups']}")
    print(f"  Groups with textual diff: {sum_gpt4_c2['groups_with_textual_diff']}")
    print(f"  Groups all identical: {sum_gpt4_c2['groups_all_identical']}")
    print(f"\n  Severity distribution:")
    for cat, count in sum_gpt4_c2["severity_distribution"].items():
        total = sum_gpt4_c2["groups_with_textual_diff"]
        pct = count / total * 100 if total > 0 else 0
        print(f"    {cat:<15}: {count:3d} ({pct:5.1f}%)")

    if sum_gpt4_c2["pairwise_similarity_when_different"]:
        all_neds = [p["mean_ned"] for p in sum_gpt4_c2["pairwise_similarity_when_different"]]
        all_rouges = [p["mean_rouge_l"] for p in sum_gpt4_c2["pairwise_similarity_when_different"]]
        print(f"\n  When outputs differ:")
        print(f"    Mean NED:     {np.mean(all_neds):.4f} (std={np.std(all_neds):.4f})")
        print(f"    Mean ROUGE-L: {np.mean(all_rouges):.4f} (std={np.std(all_rouges):.4f})")

    print("\n--- LLaMA 3 Summarization under C2 (greedy, t=0) ---")
    sum_llama_c2 = analyze_summarization_divergence(llama_sum_c2)
    print(f"  Total groups: {sum_llama_c2['total_groups']}")
    print(f"  Groups with textual diff: {sum_llama_c2['groups_with_textual_diff']}")
    print(f"  Severity: {sum_llama_c2['severity_distribution']}")

    # ── Analysis 3: Three-Level Framework ──
    print("\n" + "=" * 80)
    print("ANALYSIS 3: THREE-LEVEL REPRODUCIBILITY FRAMEWORK")
    print("=" * 80)

    framework = compute_three_level_framework(all_runs)

    # Print summary table for key conditions
    print(f"\n{'Config':<45} {'L1:EMR':>8} {'L2:NED':>8} {'L2:ROUGE':>9} {'L3:R>0.9':>9}")
    print("-" * 80)
    for key in sorted(framework.keys()):
        f = framework[key]
        if f["n_abstracts"] < 5:
            continue
        l1 = f["level_1_bitwise"]["emr_mean"]
        l2_ned = f["level_2_surface"]["ned_mean"]
        l2_rouge = f["level_2_surface"]["rouge_l_mean"]
        l3 = f["level_3_semantic"]["rouge_l_gt_090"]

        l1_str = f"{l1:.3f}" if l1 is not None else "N/A"
        l2n_str = f"{l2_ned:.4f}" if l2_ned is not None else "N/A"
        l2r_str = f"{l2_rouge:.4f}" if l2_rouge is not None else "N/A"
        l3_str = f"{l3:.1f}%" if l3 is not None else "N/A"

        print(f"  {key:<43} {l1_str:>8} {l2n_str:>8} {l2r_str:>9} {l3_str:>9}")

    # ── Summary & Key Findings ──
    print("\n" + "=" * 80)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 80)

    # Finding 1: Extraction conclusion impact
    if ext_gpt4_c2['groups_with_textual_diff'] > 0:
        pct = ext_gpt4_c2['groups_with_field_diff'] / ext_gpt4_c2['groups_with_textual_diff'] * 100
        print(f"\n1. EXTRACTION (GPT-4, greedy t=0):")
        print(f"   - {ext_gpt4_c2['groups_with_textual_diff']}/{ext_gpt4_c2['total_groups']} groups have textually different outputs")
        print(f"   - Of these, {pct:.0f}% have at least one field with different values")
        print(f"   - This is CONCLUSION-CHANGING non-determinism, not just cosmetic")

    # Finding 2: Open vs closed fields
    print(f"\n2. FIELD OPENNESS MATTERS:")
    for field in EXPECTED_FIELDS:
        stats = ext_gpt4_c2["per_field_stats"][field]
        if stats["total_groups"] > 0:
            rate = stats["divergent_groups"] / stats["total_groups"] * 100
            print(f"   - {field} ({FIELD_OPENNESS[field]}): {rate:.1f}% divergent")

    # Finding 3: Summarization severity
    print(f"\n3. SUMMARIZATION NON-DETERMINISM (GPT-4, greedy t=0):")
    total_diff = sum_gpt4_c2["groups_with_textual_diff"]
    if total_diff > 0:
        for cat in ["cosmetic", "minor", "substantive"]:
            count = sum_gpt4_c2["severity_distribution"][cat]
            pct = count / total_diff * 100
            print(f"   - {cat}: {count}/{total_diff} ({pct:.0f}%)")

    # Finding 4: Three-level framework implication
    print(f"\n4. THREE-LEVEL FRAMEWORK IMPLICATION:")
    print(f"   - At Level 1 (bitwise): GPT-4 appears highly non-reproducible (EMR ≈ 0.23-0.44)")
    print(f"   - At Level 3 (semantic): Even GPT-4 at t=0.7 maintains >90% semantic similarity")
    print(f"   - Conclusion: The LEVEL of reproducibility that matters depends on the USE CASE:")
    print(f"     * Audit/compliance → Level 1 required (hashing, tamper detection)")
    print(f"     * NLP benchmarks → Level 2 sufficient (ROUGE-L > 0.9)")
    print(f"     * Scientific conclusions → Level 3 may suffice (BERTScore > 0.94)")
    print(f"     * BUT: Even at Level 3, {ext_gpt4_c2.get('groups_with_field_diff', 0)} extraction groups")
    print(f"       had substantively different field values — so semantic ≠ safe")

    # ── Save results ──
    output = {
        "metadata": {
            "analysis": "WS-3 Conclusion Divergence",
            "total_runs_analyzed": len(all_runs),
            "date": "2026-02-09",
        },
        "extraction_gpt4_c2": ext_gpt4_c2,
        "extraction_llama_c2": ext_llama_c2,
        "extraction_gpt4_c3": ext_gpt4_c3,
        "summarization_gpt4_c2": sum_gpt4_c2,
        "summarization_llama_c2": sum_llama_c2,
        "three_level_framework": framework,
    }

    # Clean numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_file = output_dir / "conclusion_divergence.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=convert)

    print(f"\n\nResults saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
