#!/usr/bin/env python3
"""Phase 2 Statistical Enhancements for JAIR Manuscript.

Computes directly from raw run files in outputs/runs/:
1. Cliff's delta for all local-vs-API model pairs
2. Bootstrap CI for the local/API EMR ratio
3. Spearman correlation: EMR vs output length per abstract
4. Summarizes BCa CIs from existing data
"""

import json
import os
import re
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy import stats
from pathlib import Path

SEED = 42
N_BOOTSTRAP = 10000
np.random.seed(SEED)

BASE = Path("/Users/lucasrover/paper-experiment")
ANALYSIS = BASE / "analysis"
RUNS_DIR = BASE / "outputs" / "runs"

# Mapping from filename prefix to canonical model name (from bootstrap_analysis.py)
MODEL_FILE_PREFIX = {
    "llama3_8b": "llama3_8b",
    "mistral_7b": "mistral_7b",
    "gemma2_9b": "gemma2_9b",
    "gpt-4": "gpt4",
    "sonnet-4-5": "claude_sonnet",
    "gemini-2_5-pro": "gemini_pro",
    "deepseek-chat": "deepseek_chat",
    "sonar": "perplexity_sonar",
    "together_llama3_8b": "together_llama3_8b",
}

# Table 3 conditions per model (from bootstrap_analysis.py)
TABLE3_CONDITIONS = {
    "llama3_8b": "C1_fixed_seed",
    "mistral_7b": "C1_fixed_seed",
    "gemma2_9b": "C1_fixed_seed",
    "gpt4": "C2_same_params",
    "claude_sonnet": "C1_fixed_seed",
    "deepseek_chat": "C1_fixed_seed",
    "perplexity_sonar": "C1_fixed_seed",
    "together_llama3_8b": "C1_fixed_seed",
}

LOCAL_MODELS = ["llama3_8b", "mistral_7b", "gemma2_9b"]
API_MODELS = ["gpt4", "claude_sonnet", "deepseek_chat", "perplexity_sonar"]
SINGLE_TURN_TASKS = ["extraction", "summarization"]


def parse_filename(fname: str):
    """Parse a run filename into (canonical_model, task, abs_num, condition, rep).

    Copied from bootstrap_analysis.py for consistency.
    """
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


def load_run_data():
    """Load all run files and return grouped data structures.

    Returns:
        hash_data: {model -> task -> condition -> abs_num -> [hashes]}
        length_data: {model -> task -> condition -> abs_num -> [char_lengths]}
    """
    hash_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    length_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

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
                run = json.load(f)
            output_hash = run.get("output_hash")
            if output_hash is None:
                skipped += 1
                continue
            hash_data[model][task][condition][abs_num].append(output_hash)
            # Extract output length for correlation analysis
            output_text = run.get("output_text", "")
            length_data[model][task][condition][abs_num].append(len(output_text))
            loaded += 1
        except (json.JSONDecodeError, KeyError):
            skipped += 1

    print(f"  Loaded {loaded} run files, skipped {skipped}")
    return hash_data, length_data


def get_per_abstract_emrs(hash_data, model, task, condition):
    """Compute per-abstract EMR for a given model/task/condition."""
    abs_dict = hash_data.get(model, {}).get(task, {}).get(condition, {})
    emrs = {}
    for abs_num, hashes in sorted(abs_dict.items()):
        emr = compute_per_abstract_emr(hashes)
        if emr is not None:
            emrs[abs_num] = emr
    return emrs


def get_per_abstract_avg_length(length_data, model, task, condition):
    """Compute per-abstract average output length for a given model/task/condition."""
    abs_dict = length_data.get(model, {}).get(task, {}).get(condition, {})
    lengths = {}
    for abs_num, char_lengths in sorted(abs_dict.items()):
        if char_lengths:
            lengths[abs_num] = float(np.mean(char_lengths))
    return lengths


def cliffs_delta(x, y):
    """Compute Cliff's delta between two samples.

    delta = (# concordant - # discordant) / (n_x * n_y)
    Range: [-1, 1]. |delta| > 0.474 = large effect.
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return float("nan"), "undefined"

    concordant = 0
    discordant = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                concordant += 1
            elif xi < yj:
                discordant += 1

    delta = (concordant - discordant) / (n_x * n_y)

    # Effect size interpretation (Romano et al., 2006)
    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = "negligible"
    elif abs_d < 0.33:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return delta, magnitude


def bootstrap_ratio_ci(local_emrs, api_emrs, n_boot=N_BOOTSTRAP):
    """Bootstrap CI for the ratio mean(local) / mean(api)."""
    local_arr = np.array(local_emrs)
    api_arr = np.array(api_emrs)

    point_ratio = np.mean(local_arr) / np.mean(api_arr) if np.mean(api_arr) > 0 else float("inf")

    ratios = []
    for _ in range(n_boot):
        boot_local = np.random.choice(local_arr, size=len(local_arr), replace=True)
        boot_api = np.random.choice(api_arr, size=len(api_arr), replace=True)
        api_mean = np.mean(boot_api)
        if api_mean > 0:
            ratios.append(np.mean(boot_local) / api_mean)

    ratios = np.array(ratios)
    ci_lower = np.percentile(ratios, 2.5)
    ci_upper = np.percentile(ratios, 97.5)

    return {
        "point_estimate": round(float(point_ratio), 2),
        "ci_lower": round(float(ci_lower), 2),
        "ci_upper": round(float(ci_upper), 2),
        "n_bootstrap": n_boot,
        "n_local": len(local_arr),
        "n_api": len(api_arr),
    }


def main():
    print("=" * 60)
    print("PHASE 2: Statistical Enhancements")
    print("=" * 60)

    # Load raw run data
    print("\n[0] Loading raw run files...")
    hash_data, length_data = load_run_data()

    # Load BCa data from existing analysis
    with open(ANALYSIS / "enhanced_statistical_results.json") as f:
        enhanced = json.load(f)

    # =========================================================
    # 1. CLIFF'S DELTA for all local-vs-API pairs
    # =========================================================
    print("\n--- 1. Cliff's Delta (Local vs API) ---\n")

    cliff_results = {}

    for task in SINGLE_TURN_TASKS:
        print(f"\n  Task: {task}")
        print(f"  {'Pair':<45} {'delta':>8} {'magnitude':>12}")
        print(f"  {'-'*65}")

        all_local_emrs = []
        all_api_emrs = []

        for lmodel in LOCAL_MODELS:
            lcond = TABLE3_CONDITIONS[lmodel]
            local_emr_dict = get_per_abstract_emrs(hash_data, lmodel, task, lcond)
            local_emr_vals = list(local_emr_dict.values())
            all_local_emrs.extend(local_emr_vals)

            for amodel in API_MODELS:
                acond = TABLE3_CONDITIONS[amodel]
                api_emr_dict = get_per_abstract_emrs(hash_data, amodel, task, acond)
                api_emr_vals = list(api_emr_dict.values())

                if local_emr_vals and api_emr_vals:
                    delta, mag = cliffs_delta(local_emr_vals, api_emr_vals)
                    pair_key = f"{lmodel}_vs_{amodel}_{task}"
                    cliff_results[pair_key] = {
                        "delta": round(delta, 4),
                        "magnitude": mag,
                        "n_local": len(local_emr_vals),
                        "n_api": len(api_emr_vals),
                    }
                    print(f"  {lmodel} vs {amodel:<25} {delta:>8.4f} {mag:>12}")

        # Collect all API EMRs for aggregate
        for amodel in API_MODELS:
            acond = TABLE3_CONDITIONS[amodel]
            api_emr_dict = get_per_abstract_emrs(hash_data, amodel, task, acond)
            all_api_emrs.extend(list(api_emr_dict.values()))

        if all_local_emrs and all_api_emrs:
            delta_agg, mag_agg = cliffs_delta(all_local_emrs, all_api_emrs)
            agg_key = f"aggregate_local_vs_api_{task}"
            cliff_results[agg_key] = {
                "delta": round(delta_agg, 4),
                "magnitude": mag_agg,
                "n_local": len(all_local_emrs),
                "n_api": len(all_api_emrs),
            }
            print(f"  {'AGGREGATE local vs API':<45} {delta_agg:>8.4f} {mag_agg:>12}")

    # =========================================================
    # 2. BOOTSTRAP CI FOR LOCAL/API RATIO
    # =========================================================
    print("\n--- 2. Bootstrap CI for Local/API Ratio ---\n")

    all_local = []
    all_api = []

    for task in SINGLE_TURN_TASKS:
        for lmodel in LOCAL_MODELS:
            lcond = TABLE3_CONDITIONS[lmodel]
            emr_dict = get_per_abstract_emrs(hash_data, lmodel, task, lcond)
            all_local.extend(list(emr_dict.values()))

        for amodel in API_MODELS:
            acond = TABLE3_CONDITIONS[amodel]
            emr_dict = get_per_abstract_emrs(hash_data, amodel, task, acond)
            all_api.extend(list(emr_dict.values()))

    ratio_result = bootstrap_ratio_ci(all_local, all_api)
    print(f"  Local/API ratio (4 API models): {ratio_result['point_estimate']}x "
          f"[{ratio_result['ci_lower']}, {ratio_result['ci_upper']}] "
          f"(n_local={ratio_result['n_local']}, n_api={ratio_result['n_api']})")

    # 2-model API subset (GPT-4 + Claude, matching Table 4)
    all_api_2model = []
    for task in SINGLE_TURN_TASKS:
        for amodel in ["gpt4", "claude_sonnet"]:
            acond = TABLE3_CONDITIONS[amodel]
            emr_dict = get_per_abstract_emrs(hash_data, amodel, task, acond)
            all_api_2model.extend(list(emr_dict.values()))

    ratio_result_2model = bootstrap_ratio_ci(all_local, all_api_2model)
    print(f"  Local/API ratio (2 API models): {ratio_result_2model['point_estimate']}x "
          f"[{ratio_result_2model['ci_lower']}, {ratio_result_2model['ci_upper']}] "
          f"(n_local={ratio_result_2model['n_local']}, n_api={ratio_result_2model['n_api']})")

    # =========================================================
    # 3. SPEARMAN CORRELATION: EMR vs OUTPUT LENGTH
    # =========================================================
    print("\n--- 3. Spearman Correlation: EMR vs Output Length ---\n")

    correlation_results = {}
    all_models = LOCAL_MODELS + API_MODELS

    for model in all_models:
        cond = TABLE3_CONDITIONS[model]
        for task in SINGLE_TURN_TASKS:
            emr_dict = get_per_abstract_emrs(hash_data, model, task, cond)
            len_dict = get_per_abstract_avg_length(length_data, model, task, cond)

            common_abs = set(emr_dict.keys()) & set(len_dict.keys())
            if len(common_abs) >= 5:
                emrs = [emr_dict[a] for a in sorted(common_abs)]
                lengths = [len_dict[a] for a in sorted(common_abs)]

                # Skip if either input is constant (Spearman undefined)
                if len(set(emrs)) < 2 or len(set(lengths)) < 2:
                    key = f"{model}_{task}"
                    correlation_results[key] = {
                        "spearman_rho": None,
                        "p_value": None,
                        "n": len(emrs),
                        "significant": False,
                        "note": "constant input (EMR or length)",
                        "mean_length_chars": round(float(np.mean(lengths)), 1),
                        "emr_std": round(float(np.std(emrs)), 4),
                    }
                    print(f"  {model:<20} {task:<15} CONSTANT (emr_std={np.std(emrs):.4f}, len_std={np.std(lengths):.0f})  n={len(emrs)}")
                    continue

                rho, p_val = stats.spearmanr(emrs, lengths)
                key = f"{model}_{task}"
                correlation_results[key] = {
                    "spearman_rho": round(float(rho), 4),
                    "p_value": round(float(p_val), 6),
                    "n": len(emrs),
                    "significant": bool(p_val < 0.05),
                    "mean_length_chars": round(float(np.mean(lengths)), 1),
                }
                sig = "*" if p_val < 0.05 else ""
                print(f"  {model:<20} {task:<15} rho={rho:>7.4f}  p={p_val:.4f}{sig}  n={len(emrs)}")

    # =========================================================
    # 4. BCa CI SUMMARY
    # =========================================================
    print("\n--- 4. BCa vs Percentile CI Comparison ---\n")

    bca_data = enhanced.get("bca_bootstrap", {})
    bca_comparison = {}

    print(f"  {'Model+Task':<35} {'Percentile CI':>16} {'BCa CI':>16} {'Diff?':>6}")
    print(f"  {'-'*73}")

    for key, vals in sorted(bca_data.items()):
        pci = f"[{vals['ci_lower_percentile']:.2f}, {vals['ci_upper_percentile']:.2f}]"
        bci = f"[{vals['ci_lower_bca']:.2f}, {vals['ci_upper_bca']:.2f}]"
        diff = (abs(vals['ci_lower_bca'] - vals['ci_lower_percentile']) > 0.02 or
                abs(vals['ci_upper_bca'] - vals['ci_upper_percentile']) > 0.02)
        bca_comparison[key] = {
            "percentile": [vals['ci_lower_percentile'], vals['ci_upper_percentile']],
            "bca": [vals['ci_lower_bca'], vals['ci_upper_bca']],
            "notable_difference": diff,
        }
        print(f"  {key:<35} {pci:>16} {bci:>16} {'YES' if diff else 'no':>6}")

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    results = {
        "cliffs_delta": cliff_results,
        "ratio_bootstrap": {
            "all_4_api_models": ratio_result,
            "2_api_models_table4": ratio_result_2model,
        },
        "emr_vs_output_length": correlation_results,
        "bca_comparison": bca_comparison,
        "metadata": {
            "seed": SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "date": "2026-02-14",
            "script": "phase2_statistical_enhancements.py",
        },
    }

    outpath = ANALYSIS / "phase2_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {outpath}")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for task in SINGLE_TURN_TASKS:
        key = f"aggregate_local_vs_api_{task}"
        if key in cliff_results:
            d = cliff_results[key]
            print(f"  Cliff's delta (local vs API, {task}): {d['delta']} ({d['magnitude']})")

    print(f"  Local/API ratio (4 API models): {ratio_result['point_estimate']}x "
          f"95% CI [{ratio_result['ci_lower']}, {ratio_result['ci_upper']}]")
    print(f"  Local/API ratio (2 API models): {ratio_result_2model['point_estimate']}x "
          f"95% CI [{ratio_result_2model['ci_lower']}, {ratio_result_2model['ci_upper']}]")

    sig_corrs = {k: v for k, v in correlation_results.items() if v.get("significant")}
    if sig_corrs:
        print(f"  Significant EMR-length correlations: {len(sig_corrs)}")
        for k, v in sig_corrs.items():
            print(f"    {k}: rho={v['spearman_rho']}, p={v['p_value']}")
    else:
        print("  No significant EMR-length correlations found")

    notable = {k: v for k, v in bca_comparison.items() if v.get("notable_difference")}
    print(f"  BCa vs percentile notable differences: {len(notable)}/{len(bca_comparison)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
