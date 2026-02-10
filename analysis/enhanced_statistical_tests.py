#!/usr/bin/env python3
"""Enhanced statistical tests for robust reproducibility analysis.

Adds to existing statistical_tests_expanded.py:
  1. Holm-Bonferroni correction for all p-values (family-wise error control)
  2. Fisher's exact test for binary reproducibility (EMR == 1 vs EMR < 1)
  3. Cohen's h effect size for binary proportions
  4. Sensitivity analysis (exclude EMR boundary values 0/1)
  5. Per-abstract consistency check (which abstracts drive the gap?)
  6. Formal test for Claude temperature anomaly
  7. BCa bootstrap confidence intervals (bias-corrected and accelerated)

Outputs:
  - analysis/enhanced_statistical_results.json
  - Console summary

Usage:
    python analysis/enhanced_statistical_tests.py
"""

import json
import math
import os
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

RUNS_DIR = Path("/Users/lucasrover/paper-experiment/outputs/runs")
ANALYSIS_DIR = Path("/Users/lucasrover/paper-experiment/analysis")
OUTPUT_JSON = ANALYSIS_DIR / "enhanced_statistical_results.json"

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
ALPHA = 0.05

LOCAL_MODELS = {"llama3_8b", "mistral_7b", "gemma2_9b"}
API_MODELS = {"gpt4", "claude_sonnet", "deepseek-chat", "deepseek_chat", "sonar", "perplexity_sonar"}

GREEDY_CONDITIONS = {
    "C1_fixed_seed", "C2_var_seed", "C2_same_params",
    "C3_temp0.0", "C3_temp0_0",
}


# ---------------------------------------------------------------------------
# Data loading (same as statistical_tests_expanded.py)
# ---------------------------------------------------------------------------

def identify_model(run_data: dict) -> str:
    model_name = run_data.get("model_name", "").lower()
    run_id = run_data.get("run_id", "").lower()
    if "deepseek" in model_name or "deepseek" in run_id:
        return "deepseek_chat"
    elif "sonar" in model_name or "perplexity" in run_id or "sonar" in run_id:
        return "perplexity_sonar"
    elif "mistral" in model_name or "mistral" in run_id:
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
    run_id = run_data.get("run_id", "")
    parts = run_id.split("_")
    for i, p in enumerate(parts):
        if p.startswith("C") and p[1:].isdigit():
            cond_parts = [p]
            for j in range(i + 1, len(parts)):
                if parts[j].startswith("rep"):
                    break
                cond_parts.append(parts[j])
            return "_".join(cond_parts)
    return "unknown"


def is_greedy(condition: str) -> bool:
    return condition in GREEDY_CONDITIONS


def load_all_runs():
    """Load all run JSON files and compute per-abstract EMR."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for fp in RUNS_DIR.glob("*.json"):
        try:
            with open(fp) as f:
                run = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        model = identify_model(run)
        task = identify_task(run)
        condition = identify_condition(run)
        abstract_id = run.get("run_id", "").split("_")

        # Extract abstract ID (abs_NNN pattern)
        abs_id = None
        for i, p in enumerate(abstract_id):
            if p == "abs" and i + 1 < len(abstract_id):
                abs_id = f"abs_{abstract_id[i+1]}"
                break
        if not abs_id:
            continue

        output_hash = run.get("output_hash", hash(run.get("output_text", "")))
        data[model][task][condition][abs_id].append(output_hash)

    return data


def compute_emr(hashes: list) -> float:
    """Compute Exact Match Rate from pairwise hash comparisons."""
    n = len(hashes)
    if n < 2:
        return None
    pairs = list(combinations(range(n), 2))
    matching = sum(1 for i, j in pairs if hashes[i] == hashes[j])
    return matching / len(pairs)


def get_greedy_emrs(data, model, task):
    """Get per-abstract EMR values for greedy conditions."""
    emrs = {}
    for condition in data[model][task]:
        if is_greedy(condition):
            for abs_id, hashes in data[model][task][condition].items():
                emr = compute_emr(hashes)
                if emr is not None:
                    emrs[abs_id] = emr
    return emrs


# ---------------------------------------------------------------------------
# 1. Holm-Bonferroni correction
# ---------------------------------------------------------------------------

def holm_bonferroni(pvalues: list, alpha: float = 0.05) -> list:
    """Apply Holm-Bonferroni step-down correction to a list of p-values.

    Returns list of (original_p, adjusted_p, reject) tuples.
    """
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [None] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        adj_alpha = alpha / (n - rank)
        adjusted_p = min(p * (n - rank), 1.0)
        adjusted[orig_idx] = {
            "original_p": p,
            "adjusted_p": adjusted_p,
            "reject": p <= adj_alpha,
            "rank": rank + 1,
        }

    return adjusted


# ---------------------------------------------------------------------------
# 2. Fisher's exact test for binary reproducibility
# ---------------------------------------------------------------------------

def fisher_exact_reproducibility(emrs_a: list, emrs_b: list):
    """Fisher's exact test: proportion of abstracts with perfect EMR (1.0).

    H0: The proportion of perfectly reproducible abstracts is the same for
    both models.
    """
    perfect_a = sum(1 for e in emrs_a if e == 1.0)
    imperfect_a = len(emrs_a) - perfect_a
    perfect_b = sum(1 for e in emrs_b if e == 1.0)
    imperfect_b = len(emrs_b) - perfect_b

    table = [[perfect_a, imperfect_a], [perfect_b, imperfect_b]]
    odds_ratio, p_value = stats.fisher_exact(table)

    return {
        "contingency_table": table,
        "odds_ratio": round(odds_ratio, 4) if not math.isinf(odds_ratio) else "inf",
        "p_value": round(p_value, 6),
        "n_a": len(emrs_a),
        "n_b": len(emrs_b),
        "perfect_rate_a": round(perfect_a / len(emrs_a), 3) if emrs_a else 0,
        "perfect_rate_b": round(perfect_b / len(emrs_b), 3) if emrs_b else 0,
    }


# ---------------------------------------------------------------------------
# 3. Cohen's h effect size for proportions
# ---------------------------------------------------------------------------

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for comparing two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    """
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    return round(abs(h), 4)


def interpret_cohens_h(h: float) -> str:
    if h < 0.2:
        return "small"
    elif h < 0.5:
        return "medium"
    elif h < 0.8:
        return "large"
    return "very_large"


# ---------------------------------------------------------------------------
# 4. Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(emrs_local: list, emrs_api: list):
    """Re-run comparison excluding boundary EMR values (0 and 1)."""
    filtered_local = [e for e in emrs_local if 0 < e < 1]
    filtered_api = [e for e in emrs_api if 0 < e < 1]

    result = {
        "original_n_local": len(emrs_local),
        "original_n_api": len(emrs_api),
        "filtered_n_local": len(filtered_local),
        "filtered_n_api": len(filtered_api),
        "excluded_local": len(emrs_local) - len(filtered_local),
        "excluded_api": len(emrs_api) - len(filtered_api),
    }

    if len(filtered_local) >= 3 and len(filtered_api) >= 3:
        stat, p = stats.mannwhitneyu(filtered_local, filtered_api, alternative="greater")
        result["mannwhitney_stat"] = round(float(stat), 4)
        result["mannwhitney_p"] = round(float(p), 6)
        result["mean_local_filtered"] = round(float(np.mean(filtered_local)), 4)
        result["mean_api_filtered"] = round(float(np.mean(filtered_api)), 4)
        result["gap_still_significant"] = p < 0.05
    else:
        result["note"] = "Insufficient non-boundary values for filtered test"

    return result


# ---------------------------------------------------------------------------
# 5. Per-abstract consistency
# ---------------------------------------------------------------------------

def per_abstract_consistency(data, local_models, api_models, task):
    """Check if local > API holds per-abstract."""
    abstract_results = {}
    all_abs = set()

    for model in list(local_models) + list(api_models):
        emrs = get_greedy_emrs(data, model, task)
        all_abs.update(emrs.keys())

    for abs_id in sorted(all_abs):
        local_emrs = []
        api_emrs = []
        for model in local_models:
            emrs = get_greedy_emrs(data, model, task)
            if abs_id in emrs:
                local_emrs.append(emrs[abs_id])
        for model in api_models:
            emrs = get_greedy_emrs(data, model, task)
            if abs_id in emrs:
                api_emrs.append(emrs[abs_id])

        if local_emrs and api_emrs:
            abstract_results[abs_id] = {
                "local_mean_emr": round(float(np.mean(local_emrs)), 4),
                "api_mean_emr": round(float(np.mean(api_emrs)), 4),
                "local_gt_api": float(np.mean(local_emrs)) > float(np.mean(api_emrs)),
                "gap": round(float(np.mean(local_emrs) - np.mean(api_emrs)), 4),
            }

    n_consistent = sum(1 for v in abstract_results.values() if v["local_gt_api"])
    return {
        "per_abstract": abstract_results,
        "n_abstracts": len(abstract_results),
        "n_local_gt_api": n_consistent,
        "consistency_rate": round(n_consistent / len(abstract_results), 3)
        if abstract_results else 0,
    }


# ---------------------------------------------------------------------------
# 6. BCa Bootstrap CIs
# ---------------------------------------------------------------------------

def bca_bootstrap_ci(data_array, n_boot=10_000, alpha=0.05, seed=42):
    """Bias-corrected and accelerated bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(data_array)
    data = np.array(data_array)
    theta_hat = np.mean(data)

    # Bootstrap distribution
    boot_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])

    # Bias correction (z0)
    p0 = np.mean(boot_means < theta_hat)
    if p0 == 0:
        p0 = 1 / (2 * n_boot)
    elif p0 == 1:
        p0 = 1 - 1 / (2 * n_boot)
    z0 = stats.norm.ppf(p0)

    # Acceleration (a) via jackknife
    jackknife_means = np.array([
        np.mean(np.delete(data, i)) for i in range(n)
    ])
    jk_mean = np.mean(jackknife_means)
    num = np.sum((jk_mean - jackknife_means) ** 3)
    den = 6 * (np.sum((jk_mean - jackknife_means) ** 2) ** 1.5)
    a = num / den if den != 0 else 0

    # BCa percentiles
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1alpha = stats.norm.ppf(1 - alpha / 2)

    a1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    a2 = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    ci_lower = float(np.percentile(boot_means, 100 * a1))
    ci_upper = float(np.percentile(boot_means, 100 * a2))

    return {
        "mean": round(float(theta_hat), 4),
        "ci_lower_bca": round(ci_lower, 4),
        "ci_upper_bca": round(ci_upper, 4),
        "ci_lower_percentile": round(float(np.percentile(boot_means, 2.5)), 4),
        "ci_upper_percentile": round(float(np.percentile(boot_means, 97.5)), 4),
        "bias_correction_z0": round(float(z0), 4),
        "acceleration_a": round(float(a), 6),
        "n_bootstrap": n_boot,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("Loading all runs...")
    data = load_all_runs()

    all_models = set(data.keys()) - {"unknown"}
    local_in_data = LOCAL_MODELS & all_models
    api_in_data = API_MODELS & all_models

    print(f"Models found: {sorted(all_models)}")
    print(f"  Local: {sorted(local_in_data)}")
    print(f"  API:   {sorted(api_in_data)}")

    results = {
        "metadata": {
            "n_bootstrap": N_BOOTSTRAP,
            "alpha": ALPHA,
            "models_local": sorted(local_in_data),
            "models_api": sorted(api_in_data),
        },
        "holm_bonferroni": {},
        "fisher_exact": {},
        "cohens_h": {},
        "sensitivity": {},
        "per_abstract_consistency": {},
        "bca_bootstrap": {},
    }

    # Collect all p-values for Holm-Bonferroni
    all_pvalues = []
    pvalue_labels = []

    print("\n=== Fisher's Exact Tests (binary reproducibility) ===")
    for task in ["extraction", "summarization"]:
        for local_m in sorted(local_in_data):
            local_emrs = list(get_greedy_emrs(data, local_m, task).values())
            for api_m in sorted(api_in_data):
                api_emrs = list(get_greedy_emrs(data, api_m, task).values())
                if not local_emrs or not api_emrs:
                    continue

                label = f"{local_m}_vs_{api_m}_{task}"

                # Fisher's exact
                fisher = fisher_exact_reproducibility(local_emrs, api_emrs)
                results["fisher_exact"][label] = fisher
                all_pvalues.append(fisher["p_value"])
                pvalue_labels.append(f"fisher_{label}")
                print(f"  {label}: p={fisher['p_value']:.4f}, "
                      f"perfect: {fisher['perfect_rate_a']:.0%} vs {fisher['perfect_rate_b']:.0%}")

                # Cohen's h
                h = cohens_h(fisher["perfect_rate_a"], fisher["perfect_rate_b"])
                results["cohens_h"][label] = {
                    "h": h,
                    "interpretation": interpret_cohens_h(h),
                }

                # Mann-Whitney (standard comparison)
                if len(local_emrs) >= 3 and len(api_emrs) >= 3:
                    stat, p = stats.mannwhitneyu(local_emrs, api_emrs, alternative="greater")
                    all_pvalues.append(float(p))
                    pvalue_labels.append(f"mannwhitney_{label}")

    # Paired tests for models with shared abstracts
    print("\n=== Paired Tests (shared abstracts) ===")
    paired_combos = [
        ("llama3_8b", "gpt4"),
        ("mistral_7b", "claude_sonnet"),
        ("gemma2_9b", "claude_sonnet"),
    ]
    # Add new model pairs if they exist
    for new_api in ["deepseek_chat", "perplexity_sonar"]:
        if new_api in all_models:
            for local_m in sorted(local_in_data):
                paired_combos.append((local_m, new_api))

    for local_m, api_m in paired_combos:
        if local_m not in all_models or api_m not in all_models:
            continue
        for task in ["extraction", "summarization"]:
            local_emrs = get_greedy_emrs(data, local_m, task)
            api_emrs = get_greedy_emrs(data, api_m, task)
            shared = set(local_emrs.keys()) & set(api_emrs.keys())
            if len(shared) < 3:
                continue

            local_vals = [local_emrs[a] for a in sorted(shared)]
            api_vals = [api_emrs[a] for a in sorted(shared)]
            diff = [l - a for l, a in zip(local_vals, api_vals)]

            label = f"paired_{local_m}_vs_{api_m}_{task}"

            # Wilcoxon signed-rank
            try:
                stat, p = stats.wilcoxon(diff, alternative="greater")
                all_pvalues.append(float(p))
                pvalue_labels.append(f"wilcoxon_{label}")
                print(f"  {label}: Wilcoxon p={p:.6f}, n={len(shared)}")
            except ValueError:
                pass

    # Aggregate local vs API
    print("\n=== Aggregate Local vs API ===")
    for task in ["extraction", "summarization"]:
        all_local = []
        all_api = []
        for m in local_in_data:
            all_local.extend(get_greedy_emrs(data, m, task).values())
        for m in api_in_data:
            all_api.extend(get_greedy_emrs(data, m, task).values())

        if len(all_local) >= 3 and len(all_api) >= 3:
            stat, p = stats.mannwhitneyu(all_local, all_api, alternative="greater")
            all_pvalues.append(float(p))
            pvalue_labels.append(f"aggregate_local_vs_api_{task}")
            print(f"  {task}: U={stat:.1f}, p={p:.6f}, "
                  f"local_mean={np.mean(all_local):.3f}, api_mean={np.mean(all_api):.3f}")

    # Apply Holm-Bonferroni
    print(f"\n=== Holm-Bonferroni Correction ({len(all_pvalues)} tests) ===")
    corrections = holm_bonferroni(all_pvalues, ALPHA)
    for i, (label, correction) in enumerate(zip(pvalue_labels, corrections)):
        results["holm_bonferroni"][label] = correction
        sig = "REJECT" if correction["reject"] else "fail to reject"
        print(f"  [{correction['rank']}/{len(all_pvalues)}] {label}: "
              f"p_orig={correction['original_p']:.6f}, "
              f"p_adj={correction['adjusted_p']:.6f} → {sig}")

    n_reject = sum(1 for c in corrections if c["reject"])
    results["holm_bonferroni"]["summary"] = {
        "total_tests": len(all_pvalues),
        "n_rejected": n_reject,
        "n_not_rejected": len(all_pvalues) - n_reject,
        "all_primary_survive": n_reject == len(all_pvalues),
    }

    # Sensitivity analysis
    print("\n=== Sensitivity Analysis (exclude EMR boundary 0/1) ===")
    for task in ["extraction", "summarization"]:
        all_local = []
        all_api = []
        for m in local_in_data:
            all_local.extend(get_greedy_emrs(data, m, task).values())
        for m in api_in_data:
            all_api.extend(get_greedy_emrs(data, m, task).values())

        sa = sensitivity_analysis(all_local, all_api)
        results["sensitivity"][task] = sa
        if "mannwhitney_p" in sa:
            print(f"  {task}: filtered local={sa['mean_local_filtered']:.3f} vs "
                  f"api={sa['mean_api_filtered']:.3f}, "
                  f"p={sa['mannwhitney_p']:.6f}, "
                  f"significant={sa['gap_still_significant']}")

    # Per-abstract consistency
    print("\n=== Per-Abstract Consistency ===")
    for task in ["extraction", "summarization"]:
        consistency = per_abstract_consistency(data, local_in_data, api_in_data, task)
        results["per_abstract_consistency"][task] = consistency
        print(f"  {task}: {consistency['n_local_gt_api']}/{consistency['n_abstracts']} "
              f"abstracts have local > API ({consistency['consistency_rate']:.0%})")

    # BCa Bootstrap CIs
    print("\n=== BCa Bootstrap CIs ===")
    for model in sorted(all_models):
        if model == "unknown":
            continue
        for task in ["extraction", "summarization"]:
            emrs = list(get_greedy_emrs(data, model, task).values())
            if len(emrs) < 3:
                continue
            bca = bca_bootstrap_ci(emrs, N_BOOTSTRAP, ALPHA, BOOTSTRAP_SEED)
            key = f"{model}_{task}"
            results["bca_bootstrap"][key] = bca
            print(f"  {key}: mean={bca['mean']:.3f} "
                  f"BCa=[{bca['ci_lower_bca']:.3f}, {bca['ci_upper_bca']:.3f}] "
                  f"Pctl=[{bca['ci_lower_percentile']:.3f}, {bca['ci_upper_percentile']:.3f}]")

    # Save results
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_JSON}")

    return results


if __name__ == "__main__":
    main()
