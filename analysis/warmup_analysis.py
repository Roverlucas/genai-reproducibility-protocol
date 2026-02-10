#!/usr/bin/env python3
"""
Warmup Analysis: Investigate whether rep0 (first repetition) is disproportionately
the outlier in non-unanimous groups for local models under C1_fixed_seed.

This checks whether a "cold-start" / warmup effect exists in Ollama inference,
where the first run for a given (model, task, abstract) group might produce
a different output than subsequent runs.
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

RUNS_DIR = Path("/Users/lucasrover/paper-experiment/outputs/runs")
OUTPUT_FILE = Path("/Users/lucasrover/paper-experiment/analysis/warmup_results.json")

# Models and tasks to analyze
MODEL_PREFIXES = ["llama3_8b_chat", "mistral_7b"]
TASKS = ["extraction", "summarization"]
CONDITION = "C1_fixed_seed"

# Friendly names for output
MODEL_DISPLAY = {
    "llama3_8b_chat": "LLaMA 3 8B",
    "mistral_7b": "Mistral 7B",
}


def parse_filename(fname: str):
    """
    Parse a run filename like:
      llama3_8b_chat_extraction_abs_001_C1_fixed_seed_rep0.json
    Returns dict with model_prefix, task, abs_num, condition, rep_num or None.
    """
    # Remove .json
    stem = fname.replace(".json", "")

    # Find '_abs_' to split model+task from the rest
    abs_idx = stem.find("_abs_")
    if abs_idx == -1:
        return None

    prefix_part = stem[:abs_idx]       # e.g. "llama3_8b_chat_extraction"
    rest = stem[abs_idx + 5:]          # e.g. "001_C1_fixed_seed_rep0"

    # Determine model prefix and task
    model_prefix = None
    task = None
    for mp in MODEL_PREFIXES:
        for t in TASKS:
            candidate = f"{mp}_{t}"
            if prefix_part == candidate:
                model_prefix = mp
                task = t
                break
        if model_prefix:
            break

    if model_prefix is None:
        return None

    # Parse rest: "001_C1_fixed_seed_rep0"
    # abs_num is first 3 digits
    m = re.match(r"(\d+)_(.+)_rep(\d+)$", rest)
    if not m:
        return None

    abs_num = int(m.group(1))
    condition = m.group(2)
    rep_num = int(m.group(3))

    return {
        "model_prefix": model_prefix,
        "task": task,
        "abs_num": abs_num,
        "condition": condition,
        "rep_num": rep_num,
    }


def main():
    # Collect data: groups[(model, task, abs_num)] -> {rep_num: output_hash}
    groups = defaultdict(dict)

    for fname in sorted(os.listdir(RUNS_DIR)):
        if not fname.endswith(".json"):
            continue

        parsed = parse_filename(fname)
        if parsed is None:
            continue
        if parsed["condition"] != CONDITION:
            continue
        if parsed["model_prefix"] not in MODEL_PREFIXES:
            continue
        if parsed["task"] not in TASKS:
            continue

        fpath = RUNS_DIR / fname
        with open(fpath) as f:
            data = json.load(f)

        output_hash = data.get("output_hash")
        if output_hash is None:
            print(f"  WARNING: no output_hash in {fname}")
            continue

        key = (parsed["model_prefix"], parsed["task"], parsed["abs_num"])
        groups[key][parsed["rep_num"]] = output_hash

    print(f"Total groups loaded: {len(groups)}")
    print()

    # Analyze each group
    # For each (model, task): track non-unanimous groups and which rep is the outlier
    results_by_model_task = defaultdict(lambda: {
        "total_groups": 0,
        "unanimous_groups": 0,
        "non_unanimous_groups": 0,
        "outlier_rep_counts": Counter(),
        "details": [],
    })

    for (model, task, abs_num), rep_hashes in sorted(groups.items()):
        mt_key = (model, task)
        info = results_by_model_task[mt_key]
        info["total_groups"] += 1

        hashes = list(rep_hashes.values())
        unique_hashes = set(hashes)

        if len(unique_hashes) == 1:
            info["unanimous_groups"] += 1
            continue

        # Non-unanimous group
        info["non_unanimous_groups"] += 1

        # Find the modal (majority) hash
        hash_counts = Counter(hashes)
        modal_hash = hash_counts.most_common(1)[0][0]
        modal_count = hash_counts[modal_hash]

        # Identify outlier reps (those that differ from the majority)
        outlier_reps = []
        for rep_num, h in sorted(rep_hashes.items()):
            if h != modal_hash:
                outlier_reps.append(rep_num)
                info["outlier_rep_counts"][rep_num] += 1

        detail = {
            "abs_num": abs_num,
            "num_reps": len(rep_hashes),
            "num_unique_hashes": len(unique_hashes),
            "modal_hash_short": modal_hash[:12],
            "modal_count": modal_count,
            "outlier_reps": outlier_reps,
            "all_hashes_short": {r: h[:12] for r, h in sorted(rep_hashes.items())},
        }
        info["details"].append(detail)

    # Print results
    print("=" * 80)
    print("WARMUP ANALYSIS: Is rep0 disproportionately the outlier?")
    print("Condition: C1_fixed_seed | Models: LLaMA 3 8B, Mistral 7B")
    print("Tasks: extraction, summarization | Reps: 0-4")
    print("=" * 80)

    all_results = {}

    for (model, task), info in sorted(results_by_model_task.items()):
        display_name = MODEL_DISPLAY.get(model, model)
        print(f"\n{'─' * 70}")
        print(f"  {display_name} / {task}")
        print(f"{'─' * 70}")
        print(f"  Total groups (abstracts):   {info['total_groups']}")
        print(f"  Unanimous (EMR=1.0):        {info['unanimous_groups']}")
        print(f"  Non-unanimous (EMR<1.0):    {info['non_unanimous_groups']}")

        n_non = info["non_unanimous_groups"]

        if n_non == 0:
            print(f"  --> All groups are unanimous. No outliers to analyze.")
        else:
            print(f"\n  Outlier distribution across reps (in {n_non} non-unanimous groups):")
            print(f"  {'Rep':<8} {'Times as outlier':<20} {'Fraction':<12}")
            print(f"  {'---':<8} {'---':<20} {'---':<12}")

            # Determine which reps exist
            all_reps = set()
            for detail in info["details"]:
                all_reps.update(detail["all_hashes_short"].keys())

            for rep in sorted(all_reps):
                count = info["outlier_rep_counts"].get(rep, 0)
                frac = count / n_non if n_non > 0 else 0
                marker = " <-- FIRST REP" if rep == 0 else ""
                print(f"  rep{rep:<4} {count:<20} {frac:.3f}{marker}")

            print(f"\n  Detailed non-unanimous groups:")
            for detail in info["details"]:
                print(f"    abs_{detail['abs_num']:03d}: "
                      f"{detail['num_unique_hashes']} unique hashes, "
                      f"modal count={detail['modal_count']}/{detail['num_reps']}, "
                      f"outlier rep(s)={detail['outlier_reps']}")
                for r, h in sorted(detail["all_hashes_short"].items()):
                    is_outlier = "  <-- OUTLIER" if r in detail["outlier_reps"] else ""
                    print(f"      rep{r}: {h}...{is_outlier}")

        # Build JSON-serializable result
        mt_result = {
            "model": display_name,
            "task": task,
            "total_groups": info["total_groups"],
            "unanimous_groups": info["unanimous_groups"],
            "non_unanimous_groups": n_non,
            "outlier_rep_counts": {f"rep{k}": v for k, v in sorted(info["outlier_rep_counts"].items())},
            "details": info["details"],
        }
        all_results[f"{model}_{task}"] = mt_result

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"{'Model':<20} {'Task':<18} {'Non-unan.':<12} {'rep0 outlier':<14} {'rep0 frac':<12} {'rep1':<8} {'rep2':<8} {'rep3':<8} {'rep4':<8}")
    print(f"{'─' * 20} {'─' * 18} {'─' * 12} {'─' * 14} {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    total_non_unan = 0
    total_rep0_outlier = 0

    for (model, task), info in sorted(results_by_model_task.items()):
        display_name = MODEL_DISPLAY.get(model, model)
        n_non = info["non_unanimous_groups"]
        rep0_count = info["outlier_rep_counts"].get(0, 0)
        rep0_frac = rep0_count / n_non if n_non > 0 else 0.0
        rep1 = info["outlier_rep_counts"].get(1, 0)
        rep2 = info["outlier_rep_counts"].get(2, 0)
        rep3 = info["outlier_rep_counts"].get(3, 0)
        rep4 = info["outlier_rep_counts"].get(4, 0)

        total_non_unan += n_non
        total_rep0_outlier += rep0_count

        print(f"{display_name:<20} {task:<18} {n_non:<12} {rep0_count:<14} {rep0_frac:<12.3f} {rep1:<8} {rep2:<8} {rep3:<8} {rep4:<8}")

    print(f"{'─' * 20} {'─' * 18} {'─' * 12} {'─' * 14} {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    overall_frac = total_rep0_outlier / total_non_unan if total_non_unan > 0 else 0.0
    print(f"{'TOTAL':<20} {'all':<18} {total_non_unan:<12} {total_rep0_outlier:<14} {overall_frac:<12.3f}")

    print(f"\n{'=' * 80}")
    if total_non_unan > 0:
        print(f"CONCLUSION: In {total_rep0_outlier} out of {total_non_unan} non-unanimous groups "
              f"({overall_frac:.1%}), rep0 was the outlier.")
        if overall_frac > 0.5:
            print("  --> STRONG warmup effect: rep0 is disproportionately the outlier.")
        elif overall_frac > 0.3:
            print("  --> MODERATE warmup effect: rep0 is somewhat more likely to be the outlier.")
        else:
            print("  --> NO clear warmup effect: rep0 is not disproportionately the outlier.")
    else:
        print("CONCLUSION: No non-unanimous groups found. All groups are perfectly reproducible.")
    print(f"{'=' * 80}")

    # Save results
    all_results["summary"] = {
        "total_non_unanimous_groups": total_non_unan,
        "total_rep0_outlier": total_rep0_outlier,
        "rep0_outlier_fraction": overall_frac,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
