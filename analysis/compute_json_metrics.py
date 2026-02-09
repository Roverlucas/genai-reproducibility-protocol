#!/usr/bin/env python3
"""
Compute JSON extraction quality metrics for all extraction runs.
Uses the updated validation.py module to assess JSON validity, schema compliance,
and field-level accuracy across different models and conditions.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.validation import (
    json_validity_rate,
    schema_compliance_rate,
    field_level_accuracy
)


def extract_condition_from_run_id(run_id: str) -> str:
    """
    Extract condition label from run_id.
    Examples:
    - llama3_8b_extraction_abs_001_C1_fixed_seed_rep0 -> C1_fixed_seed
    - gpt4_extraction_abs_001_C2_var_seed_rep0 -> C2_var_seed
    - llama3_8b_extraction_abs_001_C3_temp0.0_rep0 -> C3_temp0.0
    """
    parts = run_id.split("_")

    # Find the C1/C2/C3 part
    for i, part in enumerate(parts):
        if part.startswith("C1") or part.startswith("C2") or part.startswith("C3"):
            # Collect condition parts (e.g., C1, fixed, seed or C3, temp0.0)
            condition_parts = [part]

            # Add following parts until we hit "rep"
            for j in range(i + 1, len(parts)):
                if parts[j].startswith("rep"):
                    break
                condition_parts.append(parts[j])

            return "_".join(condition_parts)

    return "unknown"


def extract_abstract_id(run_id: str) -> str:
    """
    Extract abstract ID from run_id.
    Example: llama3_8b_extraction_abs_001_C1_fixed_seed_rep0 -> abs_001
    """
    parts = run_id.split("_")
    for i, part in enumerate(parts):
        if part == "abs" and i + 1 < len(parts):
            return f"abs_{parts[i + 1]}"
    return "unknown"


def determine_model_group(run_id: str, model_name: str) -> str:
    """
    Determine model group label.
    - llama3:8b -> llama3
    - gpt-4 -> gpt4
    - chat control runs -> llama3_chat
    """
    if "chat" in run_id.lower():
        return "llama3_chat"
    elif "llama" in model_name.lower():
        return "llama3"
    elif "gpt" in model_name.lower():
        return "gpt4"
    else:
        return "unknown"


def load_extraction_runs(runs_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all extraction runs and group by (model, condition).
    Returns: {(model, condition): [run_data, ...]}
    """
    grouped_runs = defaultdict(list)

    # Find all JSON files
    json_files = list(runs_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {runs_dir}")

    extraction_count = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                run_data = json.load(f)

            # Filter for extraction runs only
            task_id = run_data.get("task_id", "")
            if "extraction" not in task_id.lower():
                continue

            extraction_count += 1
            run_id = run_data.get("run_id", "")
            model_name = run_data.get("model_name", "")

            # Determine grouping
            model_group = determine_model_group(run_id, model_name)
            condition = extract_condition_from_run_id(run_id)
            abstract_id = extract_abstract_id(run_id)

            # Store run data with metadata
            run_data["_model_group"] = model_group
            run_data["_condition"] = condition
            run_data["_abstract_id"] = abstract_id

            key = (model_group, condition)
            grouped_runs[key].append(run_data)

        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    print(f"Loaded {extraction_count} extraction runs")
    print(f"Found {len(grouped_runs)} unique (model, condition) groups")

    return grouped_runs


def compute_field_presence_rates(outputs: List[str]) -> Dict[str, Any]:
    """
    Compute field presence and non-empty rates for expected fields.
    """
    from metrics.validation import EXPECTED_FIELDS, _try_parse_json

    parsed_outputs = []
    for out in outputs:
        parsed, _, _ = _try_parse_json(out)
        if parsed is not None:
            parsed_outputs.append(parsed)

    if not parsed_outputs:
        return {
            "present_rate": 0.0,
            "non_empty_rate": 0.0,
            "per_field_present_rate": {f: 0.0 for f in EXPECTED_FIELDS},
            "per_field_non_empty_rate": {f: 0.0 for f in EXPECTED_FIELDS}
        }

    n = len(parsed_outputs)
    per_field_present = {f: 0 for f in EXPECTED_FIELDS}
    per_field_non_empty = {f: 0 for f in EXPECTED_FIELDS}

    for parsed in parsed_outputs:
        for field in EXPECTED_FIELDS:
            if field in parsed:
                per_field_present[field] += 1
                val = (parsed.get(field) or "").strip()
                if val:
                    per_field_non_empty[field] += 1

    per_field_present_rate = {f: count / n for f, count in per_field_present.items()}
    per_field_non_empty_rate = {f: count / n for f, count in per_field_non_empty.items()}

    overall_present = sum(per_field_present.values()) / (len(EXPECTED_FIELDS) * n)
    overall_non_empty = sum(per_field_non_empty.values()) / (len(EXPECTED_FIELDS) * n)

    return {
        "present_rate": overall_present,
        "non_empty_rate": overall_non_empty,
        "per_field_present_rate": per_field_present_rate,
        "per_field_non_empty_rate": per_field_non_empty_rate
    }


def compute_metrics_for_group(runs: List[Dict[str, Any]],
                               model_group: str,
                               condition: str) -> Dict[str, Any]:
    """
    Compute JSON extraction metrics for a group of runs.
    """
    print(f"\nProcessing {model_group} / {condition} ({len(runs)} runs)")

    # Group by abstract_id to get per-abstract output lists
    abstract_groups = defaultdict(list)
    for run in runs:
        abstract_id = run["_abstract_id"]
        output_text = run.get("output_text", "")
        abstract_groups[abstract_id].append(output_text)

    print(f"  - {len(abstract_groups)} unique abstracts")

    # Collect all outputs for overall metrics
    all_outputs = [run.get("output_text", "") for run in runs]

    # Compute validity rate (using updated validation.py)
    validity_result = json_validity_rate(all_outputs)

    # Compute schema compliance rate
    compliance_result = schema_compliance_rate(all_outputs)

    # Compute field-level accuracy (pairwise exact match)
    field_accuracy_result = field_level_accuracy(all_outputs)

    # Compute field presence rates
    field_presence_result = compute_field_presence_rates(all_outputs)

    # Compute per-abstract metrics
    per_abstract_validity = {}
    per_abstract_compliance = {}
    per_abstract_field_accuracy = {}
    for abstract_id, outputs in abstract_groups.items():
        per_abstract_validity[abstract_id] = json_validity_rate(outputs)
        per_abstract_compliance[abstract_id] = schema_compliance_rate(outputs)
        per_abstract_field_accuracy[abstract_id] = field_level_accuracy(outputs)

    return {
        "model": model_group,
        "condition": condition,
        "n_runs": len(runs),
        "n_abstracts": len(abstract_groups),
        "overall_validity": validity_result,
        "overall_compliance": compliance_result,
        "field_accuracy": field_accuracy_result,
        "field_presence": field_presence_result,
        "per_abstract_validity": per_abstract_validity,
        "per_abstract_compliance": per_abstract_compliance,
        "per_abstract_field_accuracy": per_abstract_field_accuracy
    }


def print_summary_table(results: List[Dict[str, Any]]):
    """
    Print formatted summary table of results.
    """
    print("\n" + "=" * 100)
    print("JSON EXTRACTION QUALITY METRICS SUMMARY")
    print("=" * 100)

    # Sort by model, then condition
    sorted_results = sorted(results, key=lambda x: (x["model"], x["condition"]))

    # Table 1: Validity and Compliance
    print(f"\n{'Model':<15} {'Condition':<20} {'N Runs':<8} {'N Abs':<8} "
          f"{'Raw %':<10} {'Extract %':<12} {'Compliant %':<12}")
    print("-" * 100)

    for r in sorted_results:
        model = r["model"]
        condition = r["condition"]
        n_runs = r["n_runs"]
        n_abs = r["n_abstracts"]

        validity = r["overall_validity"]
        raw_pct = validity.get("json_validity_rate_raw", 0) * 100
        extracted_pct = validity.get("json_validity_rate_extracted", 0) * 100

        compliance = r["overall_compliance"]
        compliant_pct = compliance.get("schema_compliance_rate", 0) * 100

        print(f"{model:<15} {condition:<20} {n_runs:<8} {n_abs:<8} "
              f"{raw_pct:>9.1f}% {extracted_pct:>11.1f}% {compliant_pct:>11.1f}%")

    # Table 2: Field-Level Exact Match Rates (EMR)
    print("\n" + "=" * 100)
    print("\nField-Level Exact Match Rates (Pairwise EMR):")
    print("-" * 100)

    for r in sorted_results:
        print(f"\n{r['model']} / {r['condition']}:")
        field_acc = r["field_accuracy"]

        overall_emr = field_acc.get("overall_field_emr")
        if overall_emr is not None:
            print(f"  Overall Field EMR: {overall_emr:.4f}")

            per_field = field_acc.get("field_accuracy", {})
            if per_field:
                print(f"  Per-Field EMR:")
                for field, rate in sorted(per_field.items()):
                    if rate is not None:
                        print(f"    {field:<25}: {rate:.4f}")
                    else:
                        print(f"    {field:<25}: N/A (insufficient data)")
        else:
            print(f"  N/A (insufficient data for pairwise comparison)")

    # Table 3: Field Presence and Completeness
    print("\n" + "=" * 100)
    print("\nField Presence and Completeness Rates:")
    print("-" * 100)

    for r in sorted_results:
        print(f"\n{r['model']} / {r['condition']}:")
        field_pres = r["field_presence"]

        print(f"  Overall Present Rate: {field_pres['present_rate']:.4f}")
        print(f"  Overall Non-Empty Rate: {field_pres['non_empty_rate']:.4f}")
        print(f"  Per-Field Rates:")

        for field in sorted(field_pres["per_field_present_rate"].keys()):
            present = field_pres["per_field_present_rate"][field]
            non_empty = field_pres["per_field_non_empty_rate"][field]
            print(f"    {field:<25}: present={present:.4f}, non-empty={non_empty:.4f}")


def main():
    """Main execution."""
    # Paths
    project_root = Path(__file__).parent.parent
    runs_dir = project_root / "outputs" / "runs"
    output_file = project_root / "analysis" / "json_extraction_metrics.json"

    print("=" * 100)
    print("JSON EXTRACTION QUALITY METRICS COMPUTATION")
    print("=" * 100)
    print(f"Runs directory: {runs_dir}")
    print(f"Output file: {output_file}")

    # Load and group runs
    grouped_runs = load_extraction_runs(runs_dir)

    if not grouped_runs:
        print("\nERROR: No extraction runs found!")
        return

    # Compute metrics for each group
    results = []
    for (model_group, condition), runs in sorted(grouped_runs.items()):
        metrics = compute_metrics_for_group(runs, model_group, condition)
        results.append(metrics)

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Print summary table
    print_summary_table(results)

    print("\n" + "=" * 100)
    print("COMPUTATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
