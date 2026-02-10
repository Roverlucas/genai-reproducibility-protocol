#!/usr/bin/env python3
"""Bootstrap 95% CIs for EMR and balanced subsample analysis.

Reads run files from outputs/runs/, computes per-abstract EMR via pairwise
comparison of output_hash values, then bootstraps 95% CIs.

Also computes a balanced 10-abstract subsample analysis to verify findings
hold under equal sample sizes across models.
"""

import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations
import numpy as np

RUNS_DIR = "/Users/lucasrover/paper-experiment/outputs/runs"
OUTPUT_DIR = "/Users/lucasrover/paper-experiment/analysis"

# Mapping from filename prefix to canonical model name
MODEL_FILE_PREFIX = {
    "llama3_8b": "llama3_8b",
    "mistral_7b": "mistral_7b",
    "gemma2_9b": "gemma2_9b",
    "gpt-4": "gpt4",
    "sonnet-4-5": "claude_sonnet",
    "deepseek-chat": "deepseek_chat",
    "sonar": "perplexity_sonar",
}

# For Table 3 (greedy decoding EMR): which condition to use per model
TABLE3_CONDITIONS = {
    "llama3_8b":         "C1_fixed_seed",
    "mistral_7b":        "C1_fixed_seed",
    "gemma2_9b":         "C1_fixed_seed",
    "gpt4":              "C2_same_params",
    "claude_sonnet":     "C1_fixed_seed",
    "deepseek_chat":     "C1_fixed_seed",
    "perplexity_sonar":  "C1_fixed_seed",
}

TABLE3_TASKS = ["extraction", "summarization"]

# Table 5: multiturn and RAG, 4 models (no GPT-4)
TABLE5_MODELS = ["llama3_8b", "mistral_7b", "gemma2_9b", "claude_sonnet"]
TABLE5_TASKS = ["multiturn_refinement", "rag_extraction"]
TABLE5_CONDITION = "C1_fixed_seed"

BOOTSTRAP_N = 10000
RANDOM_SEED = 42


def parse_filename(fname):
    """Parse a run filename into (canonical_model, task, abs_num, condition, rep)."""
    if not fname.endswith('.json'):
        return None
    base = fname[:-5]
    abs_idx = base.find('_abs_')
    if abs_idx == -1:
        return None
    model_task_str = base[:abs_idx]
    rest = base[abs_idx + 5:]
    abs_num = int(rest[:3])
    rep_match = re.search(r'_rep(\d+)$', rest)
    if not rep_match:
        return None
    rep = int(rep_match.group(1))
    condition = rest[4:rep_match.start()]

    file_model_prefix = None
    task = None
    for prefix in sorted(MODEL_FILE_PREFIX.keys(), key=len, reverse=True):
        if model_task_str.startswith(prefix + "_"):
            file_model_prefix = prefix
            task = model_task_str[len(prefix) + 1:]
            break
    if file_model_prefix is None:
        return None
    # Skip chat-control runs
    if task.startswith("chat_"):
        return None
    canonical_model = MODEL_FILE_PREFIX[file_model_prefix]
    return (canonical_model, task, abs_num, condition, rep)


def compute_per_abstract_emr(hashes):
    """EMR = fraction of C(n,2) pairs with identical output_hash."""
    n = len(hashes)
    if n < 2:
        return None
    pairs = list(combinations(range(n), 2))
    matching = sum(1 for i, j in pairs if hashes[i] == hashes[j])
    return matching / len(pairs)


def bootstrap_ci(per_abstract_emrs, n_boot=BOOTSTRAP_N, seed=RANDOM_SEED):
    """Bootstrap 95% CI for the mean of per-abstract EMR values."""
    rng = np.random.RandomState(seed)
    arr = np.array(per_abstract_emrs)
    n = len(arr)
    if n == 0:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "n_abstracts": 0}
    point_est = float(np.mean(arr))
    if n == 1:
        return {"mean": round(point_est, 4), "ci_lower": round(point_est, 4),
                "ci_upper": round(point_est, 4), "n_abstracts": n, "std": 0.0}
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    std = float(np.std(arr, ddof=1))
    return {
        "mean": round(point_est, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n_abstracts": n,
        "std": round(std, 4),
    }


def main():
    print("=" * 70)
    print("BOOTSTRAP CI AND BALANCED SUBSAMPLE ANALYSIS")
    print("=" * 70)

    # Step 1: Load all run files
    print("\n[1/4] Loading run files...")
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    files = os.listdir(RUNS_DIR)
    loaded = 0
    skipped = 0
    for fname in sorted(files):
        parsed = parse_filename(fname)
        if parsed is None:
            skipped += 1
            continue
        model, task, abs_num, condition, rep = parsed
        fpath = os.path.join(RUNS_DIR, fname)
        try:
            with open(fpath) as f:
                run_data = json.load(f)
            output_hash = run_data.get("output_hash")
            if output_hash is None:
                skipped += 1
                continue
            data[model][task][condition][abs_num].append(output_hash)
            loaded += 1
        except (json.JSONDecodeError, KeyError):
            skipped += 1
    print(f"  Loaded {loaded} run files, skipped {skipped}")

    # Step 2: Compute per-abstract EMR
    print("\n[2/4] Computing per-abstract EMR values...")
    per_abstract_emrs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for model in sorted(data.keys()):
        for task in sorted(data[model].keys()):
            for condition in sorted(data[model][task].keys()):
                abs_dict = data[model][task][condition]
                for abs_num in sorted(abs_dict.keys()):
                    hashes = abs_dict[abs_num]
                    emr = compute_per_abstract_emr(hashes)
                    if emr is not None:
                        per_abstract_emrs[model][task][condition][abs_num] = emr

    # Step 3: Bootstrap CIs
    print(f"\n[3/4] Bootstrapping 95% CIs (n_boot={BOOTSTRAP_N})...")
    bootstrap_results = {}

    # Table 3
    print("\n  --- Table 3: EMR Greedy (7 models x extraction, summarization) ---")
    table3 = {}
    for model in ["llama3_8b", "mistral_7b", "gemma2_9b", "gpt4", "claude_sonnet", "deepseek_chat", "perplexity_sonar"]:
        cond = TABLE3_CONDITIONS[model]
        table3[model] = {}
        for task in TABLE3_TASKS:
            emr_dict = per_abstract_emrs.get(model, {}).get(task, {}).get(cond, {})
            emr_values = list(emr_dict.values())
            ci = bootstrap_ci(emr_values)
            table3[model][task] = ci
            if ci['mean'] is not None:
                print(f"    {model:20s} | {task:20s} | {cond:20s} | EMR={ci['mean']:.3f} 95%CI [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]  (n={ci['n_abstracts']})")
            else:
                print(f"    {model:20s} | {task:20s} | {cond:20s} | NO DATA")
    bootstrap_results["table3_emr_greedy"] = table3

    # Table 5
    print("\n  --- Table 5: Multiturn & RAG (4 models x 2 tasks, C1_fixed_seed) ---")
    table5 = {}
    for model in TABLE5_MODELS:
        table5[model] = {}
        for task in TABLE5_TASKS:
            emr_dict = per_abstract_emrs.get(model, {}).get(task, {}).get(TABLE5_CONDITION, {})
            emr_values = list(emr_dict.values())
            ci = bootstrap_ci(emr_values)
            table5[model][task] = ci
            if ci['mean'] is not None:
                print(f"    {model:20s} | {task:25s} | EMR={ci['mean']:.3f} 95%CI [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]  (n={ci['n_abstracts']})")
            else:
                print(f"    {model:20s} | {task:25s} | NO DATA")
    bootstrap_results["table5_multiturn_rag"] = table5

    # All conditions
    print("\n  Computing CIs for all model/task/condition combos...")
    all_cis = {}
    for model in sorted(per_abstract_emrs.keys()):
        all_cis[model] = {}
        for task in sorted(per_abstract_emrs[model].keys()):
            all_cis[model][task] = {}
            for condition in sorted(per_abstract_emrs[model][task].keys()):
                emr_dict = per_abstract_emrs[model][task][condition]
                emr_values = list(emr_dict.values())
                ci = bootstrap_ci(emr_values)
                all_cis[model][task][condition] = ci
    bootstrap_results["all_conditions"] = all_cis

    out_path = os.path.join(OUTPUT_DIR, "bootstrap_cis.json")
    with open(out_path, "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"\n  Saved bootstrap CIs to {out_path}")

    # Step 4: Balanced subsample
    print("\n[4/4] Balanced subsample analysis (first 10 abstracts only)...")
    SUBSAMPLE_ABSTRACTS = set(range(1, 11))
    balanced_results = {}

    print("\n  --- Balanced Table 3 (first 10 abstracts) ---")
    balanced_table3 = {}
    for model in ["llama3_8b", "mistral_7b", "gemma2_9b", "gpt4", "claude_sonnet", "deepseek_chat", "perplexity_sonar"]:
        cond = TABLE3_CONDITIONS[model]
        balanced_table3[model] = {}
        for task in TABLE3_TASKS:
            emr_dict = per_abstract_emrs.get(model, {}).get(task, {}).get(cond, {})
            emr_values_sub = [v for k, v in emr_dict.items() if k in SUBSAMPLE_ABSTRACTS]
            emr_values_full = list(emr_dict.values())
            ci_sub = bootstrap_ci(emr_values_sub)
            ci_full = bootstrap_ci(emr_values_full)
            delta = None
            if ci_sub["mean"] is not None and ci_full["mean"] is not None:
                delta = round(ci_sub["mean"] - ci_full["mean"], 4)
            balanced_table3[model][task] = {
                "subsample_10": ci_sub, "full": ci_full, "delta": delta,
            }
            sub_str = f"{ci_sub['mean']:.3f} [{ci_sub['ci_lower']:.3f}, {ci_sub['ci_upper']:.3f}]" if ci_sub['mean'] is not None else "N/A"
            full_str = f"{ci_full['mean']:.3f}" if ci_full['mean'] is not None else "N/A"
            delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
            print(f"    {model:20s} | {task:15s} | sub10: {sub_str} | full: {full_str} | delta: {delta_str}")
    balanced_results["table3_balanced"] = balanced_table3

    print("\n  --- Local vs API averages (10-abstract subsample) ---")
    local_models = ["llama3_8b", "mistral_7b", "gemma2_9b"]
    api_models = ["gpt4", "claude_sonnet", "deepseek_chat", "perplexity_sonar"]
    for label, model_list in [("Local", local_models), ("API", api_models)]:
        all_emrs_sub = []
        all_emrs_full = []
        for model in model_list:
            cond = TABLE3_CONDITIONS[model]
            for task in TABLE3_TASKS:
                emr_dict = per_abstract_emrs.get(model, {}).get(task, {}).get(cond, {})
                sub_vals = [v for k, v in emr_dict.items() if k in SUBSAMPLE_ABSTRACTS]
                full_vals = list(emr_dict.values())
                all_emrs_sub.extend(sub_vals)
                all_emrs_full.extend(full_vals)
        avg_sub = float(np.mean(all_emrs_sub)) if all_emrs_sub else None
        avg_full = float(np.mean(all_emrs_full)) if all_emrs_full else None
        balanced_results[f"{label.lower()}_avg_emr_sub10"] = round(avg_sub, 4) if avg_sub is not None else None
        balanced_results[f"{label.lower()}_avg_emr_full"] = round(avg_full, 4) if avg_full is not None else None
        gap = (avg_sub - avg_full) if (avg_sub is not None and avg_full is not None) else None
        sub_str = f"{avg_sub:.4f}" if avg_sub is not None else "N/A"
        full_str = f"{avg_full:.4f}" if avg_full is not None else "N/A"
        gap_str = f"{gap:+.4f}" if gap is not None else "N/A"
        print(f"    {label:6s} avg EMR | subsample_10: {sub_str} | full: {full_str} | delta: {gap_str}")

    local_sub = balanced_results.get("local_avg_emr_sub10")
    api_sub = balanced_results.get("api_avg_emr_sub10")
    if local_sub and api_sub and api_sub > 0:
        gap_ratio = local_sub / api_sub
        balanced_results["local_api_gap_ratio_sub10"] = round(gap_ratio, 1)
        print(f"\n    Local/API ratio (sub10): {gap_ratio:.1f}x")
    local_full = balanced_results.get("local_avg_emr_full")
    api_full = balanced_results.get("api_avg_emr_full")
    if local_full and api_full and api_full > 0:
        gap_ratio_full = local_full / api_full
        balanced_results["local_api_gap_ratio_full"] = round(gap_ratio_full, 1)
        print(f"    Local/API ratio (full):  {gap_ratio_full:.1f}x")

    print("\n  --- LLaMA 3 8B: 10 vs 30 abstracts ---")
    llama_comparison = {}
    for task in TABLE3_TASKS:
        emr_dict = per_abstract_emrs.get("llama3_8b", {}).get(task, {}).get("C1_fixed_seed", {})
        sub_vals = [v for k, v in emr_dict.items() if k in SUBSAMPLE_ABSTRACTS]
        full_vals = list(emr_dict.values())
        ci_sub = bootstrap_ci(sub_vals)
        ci_full = bootstrap_ci(full_vals)
        llama_comparison[task] = {"sub10": ci_sub, "full_30": ci_full}
        print(f"    LLaMA 3 8B | {task:15s} | 10-abs EMR: {ci_sub['mean']:.3f} [{ci_sub['ci_lower']:.3f}, {ci_sub['ci_upper']:.3f}] | 30-abs EMR: {ci_full['mean']:.3f} [{ci_full['ci_lower']:.3f}, {ci_full['ci_upper']:.3f}]")
    balanced_results["llama3_10_vs_30"] = llama_comparison

    out_path2 = os.path.join(OUTPUT_DIR, "balanced_subsample.json")
    with open(out_path2, "w") as f:
        json.dump(balanced_results, f, indent=2)
    print(f"\n  Saved balanced subsample results to {out_path2}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nTable 3 -- EMR under Greedy Decoding (with 95% Bootstrap CIs):")
    print(f"  {'Model':<20s} | {'Extraction':>30s} | {'Summarization':>30s}")
    print("  " + "-" * 85)
    for model in ["llama3_8b", "mistral_7b", "gemma2_9b", "gpt4", "claude_sonnet", "deepseek_chat", "perplexity_sonar"]:
        ext = table3[model]["extraction"]
        summ = table3[model]["summarization"]
        def fmt(ci):
            if ci['mean'] is None: return "N/A"
            return f"{ci['mean']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"
        print(f"  {model:<20s} | {fmt(ext):>30s} | {fmt(summ):>30s}")

    print("\nTable 5 -- Multiturn & RAG EMR (with 95% Bootstrap CIs):")
    print(f"  {'Model':<20s} | {'Multiturn':>30s} | {'RAG':>30s}")
    print("  " + "-" * 85)
    for model in TABLE5_MODELS:
        mt = table5[model]["multiturn_refinement"]
        rag = table5[model]["rag_extraction"]
        def fmt(ci):
            if ci['mean'] is None: return "N/A"
            return f"{ci['mean']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"
        print(f"  {model:<20s} | {fmt(mt):>30s} | {fmt(rag):>30s}")

    print("\nBalanced Subsample (10 abstracts) -- Key Finding:")
    print(f"  Local avg EMR (sub10): {balanced_results.get('local_avg_emr_sub10', 'N/A')}")
    print(f"  API   avg EMR (sub10): {balanced_results.get('api_avg_emr_sub10', 'N/A')}")
    print(f"  Local/API gap (sub10): {balanced_results.get('local_api_gap_ratio_sub10', 'N/A')}x")
    print(f"  Local avg EMR (full):  {balanced_results.get('local_avg_emr_full', 'N/A')}")
    print(f"  API   avg EMR (full):  {balanced_results.get('api_avg_emr_full', 'N/A')}")
    print(f"  Local/API gap (full):  {balanced_results.get('local_api_gap_ratio_full', 'N/A')}x")
    print("\nDone.")


if __name__ == "__main__":
    main()
