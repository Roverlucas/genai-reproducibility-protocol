#!/usr/bin/env python3
"""Comprehensive statistical tests for the expanded 5-model reproducibility study.

Performs:
  1. Per-abstract EMR computation from raw run files (same grouping as analyze_expanded.py)
  2. Paired t-tests + Wilcoxon signed-rank for model pairs with shared abstracts
  3. Mann-Whitney U for model pairs without full overlap
  4. Cohen's d effect sizes
  5. Bootstrap 95% CIs (10,000 resamples) for each model/task/condition
  6. Local vs API aggregate comparisons
  7. Post-hoc power analysis

Model pairs tested:
  - LLaMA 3 8B vs GPT-4 (n=30 shared abstracts, extraction/summarization)
  - Mistral 7B vs Claude Sonnet 4.5 (n=10 shared abstracts)
  - Gemma 2 9B vs Claude Sonnet 4.5 (n=10 shared abstracts)
  - All local models combined vs all API models combined

Outputs:
  - Console results table
  - analysis/statistical_results_expanded.json
  - LaTeX snippet for inclusion in paper

Usage:
    python analysis/statistical_tests_expanded.py
"""

import json
import os
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUNS_DIR = Path("/Users/lucasrover/paper-experiment/outputs/runs")
ANALYSIS_DIR = Path("/Users/lucasrover/paper-experiment/analysis")
OUTPUT_JSON = ANALYSIS_DIR / "statistical_results_expanded.json"

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
ALPHA = 0.05

LOCAL_MODELS = {"llama3_8b", "mistral_7b", "gemma2_9b"}
API_MODELS = {"gpt4", "claude_sonnet"}

# Condition equivalences: GPT-4 uses C2_same_params, Claude uses temp0_0 etc.
# We normalize these for cross-model comparisons.
GREEDY_CONDITIONS = {
    "C1_fixed_seed", "C2_var_seed", "C2_same_params",
    "C3_temp0.0", "C3_temp0_0",
}


# ---------------------------------------------------------------------------
# Identification helpers (matching analyze_expanded.py logic exactly)
# ---------------------------------------------------------------------------

def identify_model(run_data: dict) -> str:
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
    run_id = run_data.get("run_id", "")
    parts = run_id.split("_")
    for i, part in enumerate(parts):
        if part == "abs" and i + 1 < len(parts):
            return f"abs_{parts[i + 1]}"
    return "unknown"


def is_greedy(condition: str) -> bool:
    """Check if this condition represents greedy/deterministic decoding."""
    return condition in GREEDY_CONDITIONS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_runs():
    """Load all run JSON files from the runs directory."""
    runs = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                runs.append(json.load(fp))
        except Exception as e:
            print(f"Warning: failed to load {f.name}: {e}")
    return runs


def group_runs(runs):
    """Group runs by (model, task, condition, abstract)."""
    groups = defaultdict(list)
    for run in runs:
        model = identify_model(run)
        task = identify_task(run)
        condition = identify_condition(run)
        abstract = identify_abstract(run)
        groups[(model, task, condition, abstract)].append(run)
    return groups


# ---------------------------------------------------------------------------
# EMR computation (per abstract, pairwise — same as analyze_expanded.py)
# ---------------------------------------------------------------------------

def compute_emr(runs):
    """Compute pairwise exact match rate for a list of runs."""
    outputs = [r.get("output_text", "") for r in runs]
    outputs = [o for o in outputs if o.strip()]
    n = len(outputs)
    if n < 2:
        return None
    pairs = list(combinations(range(n), 2))
    match_count = sum(1 for i, j in pairs if outputs[i] == outputs[j])
    return match_count / len(pairs)


def build_per_abstract_emr(groups):
    """Build per-abstract EMR for each (model, task, condition).

    Returns dict: (model, task, condition) -> {abstract_id: emr_value, ...}
    """
    mtc_emr = defaultdict(dict)
    for (model, task, condition, abstract), runs in groups.items():
        emr = compute_emr(runs)
        if emr is not None:
            mtc_emr[(model, task, condition)][abstract] = emr
    return mtc_emr


# ---------------------------------------------------------------------------
# Statistical test helpers
# ---------------------------------------------------------------------------

def cohens_d_paired(x, y):
    """Cohen's d for paired samples (standardized mean difference)."""
    diff = np.array(x) - np.array(y)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return float("inf") if mean_diff != 0 else 0.0
    return mean_diff / std_diff


def cohens_d_independent(x, y):
    """Cohen's d for independent samples (pooled SD)."""
    nx, ny = len(x), len(y)
    mean_diff = np.mean(x) - np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_sd == 0:
        return float("inf") if mean_diff != 0 else 0.0
    return mean_diff / pooled_sd


def bootstrap_ci(data, n_boot=N_BOOTSTRAP, alpha=ALPHA, seed=BOOTSTRAP_SEED):
    """Compute bootstrap 95% CI for the mean of data."""
    rng = np.random.RandomState(seed)
    data = np.array(data)
    n = len(data)
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        return (float(data[0]), float(data[0]))
    boot_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (ci_low, ci_high)


def post_hoc_power(cohens_d_val, n, alpha=ALPHA):
    """Approximate post-hoc power for a paired t-test using normal approx."""
    if abs(cohens_d_val) == float("inf"):
        return 1.0
    t_crit = stats.t.ppf(1 - alpha / 2, df=max(n - 1, 1))
    ncp = abs(cohens_d_val) * np.sqrt(n)
    power = 1 - stats.norm.cdf(t_crit - ncp) + stats.norm.cdf(-t_crit - ncp)
    return float(power)


def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# Paired model comparison
# ---------------------------------------------------------------------------

def paired_comparison(emr_a, emr_b, name_a, name_b, shared_abstracts):
    """Run full paired statistical comparison between two models.

    emr_a, emr_b: dicts {abstract_id: emr}
    shared_abstracts: sorted list of shared abstract IDs
    Returns results dict.
    """
    vals_a = np.array([emr_a[ab] for ab in shared_abstracts])
    vals_b = np.array([emr_b[ab] for ab in shared_abstracts])
    n = len(shared_abstracts)

    result = {
        "comparison": f"{name_a} vs {name_b}",
        "n_abstracts": n,
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "std_a": float(np.std(vals_a, ddof=1)),
        "std_b": float(np.std(vals_b, ddof=1)),
    }

    diff = vals_a - vals_b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    result["mean_diff"] = mean_diff
    result["std_diff"] = std_diff

    # --- Paired t-test ---
    if std_diff > 0 and n >= 2:
        t_stat, t_p = stats.ttest_rel(vals_a, vals_b)
        result["ttest_t"] = float(t_stat)
        result["ttest_p"] = float(t_p)
        result["ttest_sig"] = significance_stars(t_p)
    else:
        # All differences are zero or only 1 sample
        result["ttest_t"] = float("nan")
        result["ttest_p"] = float("nan") if n < 2 else 1.0
        result["ttest_sig"] = "n.s." if n < 2 else ("n/a (zero variance)" if std_diff == 0 else "n.s.")

    # --- Wilcoxon signed-rank ---
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) >= 1:
        try:
            w_stat, w_p = stats.wilcoxon(vals_a, vals_b, alternative="two-sided")
            result["wilcoxon_W"] = float(w_stat)
            result["wilcoxon_p"] = float(w_p)
            result["wilcoxon_sig"] = significance_stars(w_p)
        except ValueError:
            # All differences are zero
            result["wilcoxon_W"] = float("nan")
            result["wilcoxon_p"] = 1.0
            result["wilcoxon_sig"] = "n/a (all tied)"
    else:
        result["wilcoxon_W"] = float("nan")
        result["wilcoxon_p"] = 1.0
        result["wilcoxon_sig"] = "n/a (all tied)"

    # --- Cohen's d (paired) ---
    d = cohens_d_paired(vals_a, vals_b)
    result["cohens_d"] = float(d) if abs(d) != float("inf") else "inf"

    # --- 95% CI for the paired difference (parametric) ---
    if std_diff > 0 and n >= 2:
        se = std_diff / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        result["ci_diff_low"] = float(mean_diff - t_crit * se)
        result["ci_diff_high"] = float(mean_diff + t_crit * se)
    else:
        result["ci_diff_low"] = mean_diff
        result["ci_diff_high"] = mean_diff

    # --- Bootstrap CI for the difference ---
    ci = bootstrap_ci(diff)
    result["bootstrap_ci_diff"] = [ci[0], ci[1]]

    # --- Bootstrap CIs for each model ---
    ci_a = bootstrap_ci(vals_a)
    ci_b = bootstrap_ci(vals_b)
    result["bootstrap_ci_a"] = [ci_a[0], ci_a[1]]
    result["bootstrap_ci_b"] = [ci_b[0], ci_b[1]]

    # --- Post-hoc power ---
    if isinstance(d, float) and abs(d) != float("inf") and n >= 2:
        result["power"] = post_hoc_power(d, n)
    else:
        result["power"] = 1.0

    # --- Shapiro-Wilk normality of differences ---
    if n >= 3 and std_diff > 0:
        sw_stat, sw_p = stats.shapiro(diff)
        result["shapiro_W"] = float(sw_stat)
        result["shapiro_p"] = float(sw_p)
        result["normality"] = "met" if sw_p > 0.05 else "violated"
    else:
        result["shapiro_W"] = float("nan")
        result["shapiro_p"] = float("nan")
        result["normality"] = "n/a"

    return result


# ---------------------------------------------------------------------------
# Independent model comparison (Mann-Whitney U)
# ---------------------------------------------------------------------------

def independent_comparison(vals_a, vals_b, name_a, name_b):
    """Run independent-sample tests when abstracts do not overlap."""
    vals_a = np.array(vals_a)
    vals_b = np.array(vals_b)
    na, nb = len(vals_a), len(vals_b)

    result = {
        "comparison": f"{name_a} vs {name_b}",
        "n_a": na,
        "n_b": nb,
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "std_a": float(np.std(vals_a, ddof=1)) if na > 1 else 0.0,
        "std_b": float(np.std(vals_b, ddof=1)) if nb > 1 else 0.0,
    }

    # Independent t-test (Welch's)
    if na >= 2 and nb >= 2:
        t_stat, t_p = stats.ttest_ind(vals_a, vals_b, equal_var=False)
        result["ttest_t"] = float(t_stat)
        result["ttest_p"] = float(t_p)
        result["ttest_sig"] = significance_stars(t_p)
    else:
        result["ttest_t"] = float("nan")
        result["ttest_p"] = float("nan")
        result["ttest_sig"] = "n/a"

    # Mann-Whitney U
    if na >= 1 and nb >= 1:
        try:
            u_stat, u_p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            result["mannwhitney_U"] = float(u_stat)
            result["mannwhitney_p"] = float(u_p)
            result["mannwhitney_sig"] = significance_stars(u_p)
        except ValueError:
            result["mannwhitney_U"] = float("nan")
            result["mannwhitney_p"] = 1.0
            result["mannwhitney_sig"] = "n/a"
    else:
        result["mannwhitney_U"] = float("nan")
        result["mannwhitney_p"] = float("nan")
        result["mannwhitney_sig"] = "n/a"

    # Cohen's d (independent)
    d = cohens_d_independent(vals_a, vals_b)
    result["cohens_d"] = float(d) if abs(d) != float("inf") else "inf"

    # Bootstrap CIs
    result["bootstrap_ci_a"] = list(bootstrap_ci(vals_a))
    result["bootstrap_ci_b"] = list(bootstrap_ci(vals_b))
    ci_diff = bootstrap_ci(
        np.concatenate([vals_a, -vals_b]) if na == nb
        else vals_a,  # fallback: just report CI of model A
    )
    # For unequal sizes, use permutation-style bootstrap for difference
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    boot_diffs = []
    for _ in range(N_BOOTSTRAP):
        sample_a = rng.choice(vals_a, size=na, replace=True)
        sample_b = rng.choice(vals_b, size=nb, replace=True)
        boot_diffs.append(np.mean(sample_a) - np.mean(sample_b))
    boot_diffs = np.array(boot_diffs)
    result["bootstrap_ci_diff"] = [
        float(np.percentile(boot_diffs, 100 * ALPHA / 2)),
        float(np.percentile(boot_diffs, 100 * (1 - ALPHA / 2))),
    ]

    return result


# ---------------------------------------------------------------------------
# Normalize conditions for cross-model comparisons
# ---------------------------------------------------------------------------

def normalize_condition(cond):
    """Normalize condition names for cross-model matching.

    GPT-4 uses C2_same_params (equiv to C2_var_seed for local).
    Claude uses C3_temp0_0 (equiv to C3_temp0.0).
    """
    mappings = {
        "C2_same_params": "C2_greedy",
        "C2_var_seed": "C2_greedy",
        "C1_fixed_seed": "C1_greedy",
        "C3_temp0.0": "C3_temp0.0",
        "C3_temp0_0": "C3_temp0.0",
        "C3_temp0.3": "C3_temp0.3",
        "C3_temp0_3": "C3_temp0.3",
        "C3_temp0.7": "C3_temp0.7",
        "C3_temp0_7": "C3_temp0.7",
    }
    return mappings.get(cond, cond)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("COMPREHENSIVE STATISTICAL TESTS — EXPANDED 5-MODEL STUDY")
    print("=" * 90)

    # Load and group runs
    print("\nLoading runs...")
    runs = load_all_runs()
    print(f"  Loaded {len(runs)} total runs")

    groups = group_runs(runs)
    print(f"  {len(groups)} unique (model, task, condition, abstract) groups")

    # Build per-abstract EMR
    per_abstract_emr = build_per_abstract_emr(groups)
    print(f"  {len(per_abstract_emr)} unique (model, task, condition) entries")

    # Print summary of available data
    model_counts = defaultdict(int)
    for r in runs:
        model_counts[identify_model(r)] += 1
    for model, count in sorted(model_counts.items()):
        print(f"    {model}: {count} runs")

    # ===================================================================
    # SECTION 1: Bootstrap CIs for each model x task x condition
    # ===================================================================
    print("\n" + "=" * 90)
    print("SECTION 1: BOOTSTRAP 95% CIs FOR EACH MODEL x TASK x CONDITION")
    print("=" * 90)

    bootstrap_results = []
    print(f"\n{'Model':<16} {'Task':<24} {'Condition':<18} {'N':>4} "
          f"{'EMR Mean':>9} {'95% CI':>22}")
    print("-" * 95)

    for (model, task, condition), ab_emrs in sorted(per_abstract_emr.items()):
        emr_values = list(ab_emrs.values())
        n = len(emr_values)
        mean_emr = float(np.mean(emr_values))
        std_emr = float(np.std(emr_values, ddof=1)) if n > 1 else 0.0
        ci = bootstrap_ci(emr_values)

        entry = {
            "model": model,
            "task": task,
            "condition": condition,
            "n_abstracts": n,
            "emr_mean": mean_emr,
            "emr_std": std_emr,
            "bootstrap_ci_low": ci[0],
            "bootstrap_ci_high": ci[1],
        }
        bootstrap_results.append(entry)

        print(f"{model:<16} {task:<24} {condition:<18} {n:>4} "
              f"{mean_emr:>9.4f} [{ci[0]:>8.4f}, {ci[1]:>8.4f}]")

    # ===================================================================
    # SECTION 2: Paired comparisons — LLaMA vs GPT-4 (n=30 shared)
    # ===================================================================
    print("\n" + "=" * 90)
    print("SECTION 2: PAIRED COMPARISONS — LLaMA 3 8B vs GPT-4")
    print("=" * 90)

    paired_results = []

    # LLaMA and GPT-4 share abstracts 001-030 for extraction and summarization
    # But conditions differ: LLaMA uses C1/C2_var_seed/C3, GPT-4 uses C2_same_params/C3
    # We match: C1_fixed_seed (LLaMA) — may not exist for GPT-4 extraction
    #           C2_var_seed (LLaMA) <-> C2_same_params (GPT-4) as "greedy C2"
    #           C3_temp0.X (same)
    llama_gpt4_pairs = [
        # (llama_condition, gpt4_condition, label)
        ("C2_var_seed", "C2_same_params", "C2 (greedy)"),
        ("C3_temp0.0", "C3_temp0.0", "C3 temp=0.0"),
        ("C3_temp0.3", "C3_temp0.3", "C3 temp=0.3"),
        ("C3_temp0.7", "C3_temp0.7", "C3 temp=0.7"),
    ]

    for task in ["extraction", "summarization"]:
        print(f"\n--- {task.upper()} ---")
        for llama_cond, gpt4_cond, label in llama_gpt4_pairs:
            key_l = ("llama3_8b", task, llama_cond)
            key_g = ("gpt4", task, gpt4_cond)

            if key_l not in per_abstract_emr or key_g not in per_abstract_emr:
                print(f"  {label}: SKIPPED (data not available)")
                continue

            emr_l = per_abstract_emr[key_l]
            emr_g = per_abstract_emr[key_g]
            shared = sorted(set(emr_l.keys()) & set(emr_g.keys()))

            if len(shared) < 2:
                print(f"  {label}: SKIPPED (only {len(shared)} shared abstracts)")
                continue

            res = paired_comparison(emr_l, emr_g, "LLaMA 3 8B", "GPT-4", shared)
            res["task"] = task
            res["condition_label"] = label
            res["condition_a"] = llama_cond
            res["condition_b"] = gpt4_cond
            paired_results.append(res)

            print(f"\n  {label} (n={res['n_abstracts']} shared abstracts):")
            print(f"    LLaMA: {res['mean_a']:.4f} +/- {res['std_a']:.4f}  "
                  f"95% boot CI [{res['bootstrap_ci_a'][0]:.4f}, {res['bootstrap_ci_a'][1]:.4f}]")
            print(f"    GPT-4: {res['mean_b']:.4f} +/- {res['std_b']:.4f}  "
                  f"95% boot CI [{res['bootstrap_ci_b'][0]:.4f}, {res['bootstrap_ci_b'][1]:.4f}]")
            print(f"    Diff:  {res['mean_diff']:+.4f} [{res['ci_diff_low']:+.4f}, {res['ci_diff_high']:+.4f}]")
            print(f"    Paired t({res['n_abstracts']-1}): t={res['ttest_t']:.3f}, "
                  f"p={res['ttest_p']:.2e} {res['ttest_sig']}")
            print(f"    Wilcoxon: W={res['wilcoxon_W']:.1f}, "
                  f"p={res['wilcoxon_p']:.2e} {res['wilcoxon_sig']}"
                  if not np.isnan(res.get('wilcoxon_W', float('nan')))
                  else f"    Wilcoxon: {res['wilcoxon_sig']}")
            d_str = f"{res['cohens_d']:.3f}" if isinstance(res['cohens_d'], float) else res['cohens_d']
            print(f"    Cohen's d: {d_str}")
            print(f"    Power: {res['power']:.3f}")
            print(f"    Normality (Shapiro-Wilk): {res['normality']} (p={res['shapiro_p']:.4f})"
                  if not np.isnan(res.get('shapiro_p', float('nan')))
                  else f"    Normality: {res['normality']}")

    # ===================================================================
    # SECTION 3: Paired comparisons — Mistral vs Claude (n=10 shared)
    # ===================================================================
    print("\n" + "=" * 90)
    print("SECTION 3: PAIRED COMPARISONS — MISTRAL 7B vs CLAUDE SONNET 4.5")
    print("=" * 90)

    # Mistral: C1/C2_var_seed/C3_temp0.X, Claude: C1/C2_var_seed/C3_temp0_X
    mistral_claude_pairs = [
        ("C1_fixed_seed", "C1_fixed_seed", "C1 (fixed seed)"),
        ("C2_var_seed", "C2_var_seed", "C2 (variable seed)"),
        ("C3_temp0.0", "C3_temp0_0", "C3 temp=0.0"),
        ("C3_temp0.3", "C3_temp0_3", "C3 temp=0.3"),
        ("C3_temp0.7", "C3_temp0_7", "C3 temp=0.7"),
    ]

    for task in ["extraction", "summarization"]:
        print(f"\n--- {task.upper()} ---")
        for mistral_cond, claude_cond, label in mistral_claude_pairs:
            key_m = ("mistral_7b", task, mistral_cond)
            key_c = ("claude_sonnet", task, claude_cond)

            if key_m not in per_abstract_emr or key_c not in per_abstract_emr:
                print(f"  {label}: SKIPPED (data not available)")
                continue

            emr_m = per_abstract_emr[key_m]
            emr_c = per_abstract_emr[key_c]
            shared = sorted(set(emr_m.keys()) & set(emr_c.keys()))

            if len(shared) < 2:
                print(f"  {label}: SKIPPED (only {len(shared)} shared abstracts)")
                continue

            res = paired_comparison(emr_m, emr_c, "Mistral 7B", "Claude Sonnet 4.5", shared)
            res["task"] = task
            res["condition_label"] = label
            res["condition_a"] = mistral_cond
            res["condition_b"] = claude_cond
            paired_results.append(res)

            print(f"\n  {label} (n={res['n_abstracts']} shared abstracts):")
            print(f"    Mistral: {res['mean_a']:.4f} +/- {res['std_a']:.4f}  "
                  f"95% boot CI [{res['bootstrap_ci_a'][0]:.4f}, {res['bootstrap_ci_a'][1]:.4f}]")
            print(f"    Claude:  {res['mean_b']:.4f} +/- {res['std_b']:.4f}  "
                  f"95% boot CI [{res['bootstrap_ci_b'][0]:.4f}, {res['bootstrap_ci_b'][1]:.4f}]")
            print(f"    Diff:  {res['mean_diff']:+.4f} [{res['ci_diff_low']:+.4f}, {res['ci_diff_high']:+.4f}]")
            print(f"    Paired t({res['n_abstracts']-1}): t={res['ttest_t']:.3f}, "
                  f"p={res['ttest_p']:.2e} {res['ttest_sig']}")
            if not np.isnan(res.get('wilcoxon_W', float('nan'))):
                print(f"    Wilcoxon: W={res['wilcoxon_W']:.1f}, "
                      f"p={res['wilcoxon_p']:.2e} {res['wilcoxon_sig']}")
            else:
                print(f"    Wilcoxon: {res['wilcoxon_sig']}")
            d_str = f"{res['cohens_d']:.3f}" if isinstance(res['cohens_d'], float) else res['cohens_d']
            print(f"    Cohen's d: {d_str}")
            print(f"    Power: {res['power']:.3f}")
            if not np.isnan(res.get('shapiro_p', float('nan'))):
                print(f"    Normality (Shapiro-Wilk): {res['normality']} (p={res['shapiro_p']:.4f})")
            else:
                print(f"    Normality: {res['normality']}")

    # ===================================================================
    # SECTION 4: Paired comparisons — Gemma vs Claude (n=10 shared)
    # ===================================================================
    print("\n" + "=" * 90)
    print("SECTION 4: PAIRED COMPARISONS — GEMMA 2 9B vs CLAUDE SONNET 4.5")
    print("=" * 90)

    gemma_claude_pairs = [
        ("C1_fixed_seed", "C1_fixed_seed", "C1 (fixed seed)"),
        ("C2_var_seed", "C2_var_seed", "C2 (variable seed)"),
        ("C3_temp0.0", "C3_temp0_0", "C3 temp=0.0"),
        ("C3_temp0.3", "C3_temp0_3", "C3 temp=0.3"),
        ("C3_temp0.7", "C3_temp0_7", "C3 temp=0.7"),
    ]

    for task in ["extraction", "summarization"]:
        print(f"\n--- {task.upper()} ---")
        for gemma_cond, claude_cond, label in gemma_claude_pairs:
            key_g = ("gemma2_9b", task, gemma_cond)
            key_c = ("claude_sonnet", task, claude_cond)

            if key_g not in per_abstract_emr or key_c not in per_abstract_emr:
                print(f"  {label}: SKIPPED (data not available)")
                continue

            emr_g = per_abstract_emr[key_g]
            emr_c = per_abstract_emr[key_c]
            shared = sorted(set(emr_g.keys()) & set(emr_c.keys()))

            if len(shared) < 2:
                print(f"  {label}: SKIPPED (only {len(shared)} shared abstracts)")
                continue

            res = paired_comparison(emr_g, emr_c, "Gemma 2 9B", "Claude Sonnet 4.5", shared)
            res["task"] = task
            res["condition_label"] = label
            res["condition_a"] = gemma_cond
            res["condition_b"] = claude_cond
            paired_results.append(res)

            print(f"\n  {label} (n={res['n_abstracts']} shared abstracts):")
            print(f"    Gemma:  {res['mean_a']:.4f} +/- {res['std_a']:.4f}  "
                  f"95% boot CI [{res['bootstrap_ci_a'][0]:.4f}, {res['bootstrap_ci_a'][1]:.4f}]")
            print(f"    Claude: {res['mean_b']:.4f} +/- {res['std_b']:.4f}  "
                  f"95% boot CI [{res['bootstrap_ci_b'][0]:.4f}, {res['bootstrap_ci_b'][1]:.4f}]")
            print(f"    Diff:  {res['mean_diff']:+.4f} [{res['ci_diff_low']:+.4f}, {res['ci_diff_high']:+.4f}]")
            print(f"    Paired t({res['n_abstracts']-1}): t={res['ttest_t']:.3f}, "
                  f"p={res['ttest_p']:.2e} {res['ttest_sig']}")
            if not np.isnan(res.get('wilcoxon_W', float('nan'))):
                print(f"    Wilcoxon: W={res['wilcoxon_W']:.1f}, "
                      f"p={res['wilcoxon_p']:.2e} {res['wilcoxon_sig']}")
            else:
                print(f"    Wilcoxon: {res['wilcoxon_sig']}")
            d_str = f"{res['cohens_d']:.3f}" if isinstance(res['cohens_d'], float) else res['cohens_d']
            print(f"    Cohen's d: {d_str}")
            print(f"    Power: {res['power']:.3f}")
            if not np.isnan(res.get('shapiro_p', float('nan'))):
                print(f"    Normality (Shapiro-Wilk): {res['normality']} (p={res['shapiro_p']:.4f})")
            else:
                print(f"    Normality: {res['normality']}")

    # ===================================================================
    # SECTION 5: Local avg vs API avg — aggregate comparison
    # ===================================================================
    print("\n" + "=" * 90)
    print("SECTION 5: AGGREGATE — ALL LOCAL MODELS vs ALL API MODELS")
    print("=" * 90)

    aggregate_results = []

    # Collect all greedy-condition EMRs per model category
    # "Greedy" = C1, C2, or C3_temp0.0 (temperature=0 conditions)
    for task in ["extraction", "summarization"]:
        local_emrs = []
        api_emrs = []

        for (model, t, condition), ab_emrs in per_abstract_emr.items():
            if t != task:
                continue
            norm_cond = normalize_condition(condition)
            # Only include greedy/deterministic conditions for fair comparison
            if norm_cond not in ("C1_greedy", "C2_greedy", "C3_temp0.0"):
                continue

            emr_vals = list(ab_emrs.values())
            if model in LOCAL_MODELS:
                local_emrs.extend(emr_vals)
            elif model in API_MODELS:
                api_emrs.extend(emr_vals)

        if local_emrs and api_emrs:
            res = independent_comparison(local_emrs, api_emrs,
                                         "Local (LLaMA+Mistral+Gemma)",
                                         "API (GPT-4+Claude)")
            res["task"] = task
            res["condition_label"] = "greedy conditions (C1/C2/C3_t0)"
            aggregate_results.append(res)

            print(f"\n--- {task.upper()} (greedy conditions) ---")
            print(f"  Local models: n={res['n_a']}, mean EMR={res['mean_a']:.4f} +/- {res['std_a']:.4f}")
            print(f"    95% boot CI [{res['bootstrap_ci_a'][0]:.4f}, {res['bootstrap_ci_a'][1]:.4f}]")
            print(f"  API models:   n={res['n_b']}, mean EMR={res['mean_b']:.4f} +/- {res['std_b']:.4f}")
            print(f"    95% boot CI [{res['bootstrap_ci_b'][0]:.4f}, {res['bootstrap_ci_b'][1]:.4f}]")
            d_str = f"{res['cohens_d']:.3f}" if isinstance(res['cohens_d'], float) else res['cohens_d']
            print(f"  Welch's t: t={res['ttest_t']:.3f}, p={res['ttest_p']:.2e} {res['ttest_sig']}")
            print(f"  Mann-Whitney U: U={res['mannwhitney_U']:.1f}, "
                  f"p={res['mannwhitney_p']:.2e} {res['mannwhitney_sig']}")
            print(f"  Cohen's d: {d_str}")
            print(f"  Bootstrap CI for diff: [{res['bootstrap_ci_diff'][0]:.4f}, "
                  f"{res['bootstrap_ci_diff'][1]:.4f}]")

    # Also do an overall aggregate (both tasks combined)
    local_all = []
    api_all = []
    for (model, t, condition), ab_emrs in per_abstract_emr.items():
        norm_cond = normalize_condition(condition)
        if norm_cond not in ("C1_greedy", "C2_greedy", "C3_temp0.0"):
            continue
        if t not in ("extraction", "summarization"):
            continue
        emr_vals = list(ab_emrs.values())
        if model in LOCAL_MODELS:
            local_all.extend(emr_vals)
        elif model in API_MODELS:
            api_all.extend(emr_vals)

    if local_all and api_all:
        res = independent_comparison(local_all, api_all,
                                     "Local (all)", "API (all)")
        res["task"] = "all (extraction + summarization)"
        res["condition_label"] = "greedy conditions (C1/C2/C3_t0)"
        aggregate_results.append(res)

        print(f"\n--- OVERALL (extraction + summarization, greedy) ---")
        print(f"  Local: n={res['n_a']}, mean EMR={res['mean_a']:.4f} +/- {res['std_a']:.4f}")
        print(f"    95% boot CI [{res['bootstrap_ci_a'][0]:.4f}, {res['bootstrap_ci_a'][1]:.4f}]")
        print(f"  API:   n={res['n_b']}, mean EMR={res['mean_b']:.4f} +/- {res['std_b']:.4f}")
        print(f"    95% boot CI [{res['bootstrap_ci_b'][0]:.4f}, {res['bootstrap_ci_b'][1]:.4f}]")
        d_str = f"{res['cohens_d']:.3f}" if isinstance(res['cohens_d'], float) else res['cohens_d']
        print(f"  Welch's t: t={res['ttest_t']:.3f}, p={res['ttest_p']:.2e} {res['ttest_sig']}")
        print(f"  Mann-Whitney U: U={res['mannwhitney_U']:.1f}, "
              f"p={res['mannwhitney_p']:.2e} {res['mannwhitney_sig']}")
        print(f"  Cohen's d: {d_str}")
        print(f"  Bootstrap CI for diff: [{res['bootstrap_ci_diff'][0]:.4f}, "
              f"{res['bootstrap_ci_diff'][1]:.4f}]")

    # ===================================================================
    # SECTION 6: Per-model aggregate bootstrap CIs (greedy only)
    # ===================================================================
    print("\n" + "=" * 90)
    print("SECTION 6: PER-MODEL AGGREGATE EMR (GREEDY CONDITIONS)")
    print("=" * 90)

    per_model_aggregate = {}
    for (model, task, condition), ab_emrs in per_abstract_emr.items():
        norm_cond = normalize_condition(condition)
        if norm_cond not in ("C1_greedy", "C2_greedy", "C3_temp0.0"):
            continue
        if task not in ("extraction", "summarization"):
            continue
        per_model_aggregate.setdefault(model, []).extend(ab_emrs.values())

    print(f"\n{'Model':<18} {'N':>5} {'Mean EMR':>10} {'Std':>8} {'95% Bootstrap CI':>24}")
    print("-" * 70)
    model_agg_results = {}
    for model in sorted(per_model_aggregate.keys()):
        vals = per_model_aggregate[model]
        n = len(vals)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        ci = bootstrap_ci(vals)
        model_agg_results[model] = {
            "n": n, "mean": mean, "std": std,
            "ci_low": ci[0], "ci_high": ci[1],
        }
        print(f"  {model:<16} {n:>5} {mean:>10.4f} {std:>8.4f} "
              f"[{ci[0]:>8.4f}, {ci[1]:>8.4f}]")

    # ===================================================================
    # SECTION 7: Summary table — effect sizes across all comparisons
    # ===================================================================
    print("\n" + "=" * 90)
    print("SECTION 7: SUMMARY OF ALL PAIRED COMPARISONS")
    print("=" * 90)
    print(f"\n{'Comparison':<35} {'Task':<15} {'Condition':<18} "
          f"{'n':>4} {'d':>8} {'t-p':>10} {'W-p':>10} {'Pwr':>6}")
    print("-" * 110)
    for res in paired_results:
        d_str = f"{res['cohens_d']:.3f}" if isinstance(res['cohens_d'], float) else res['cohens_d']
        t_p_str = f"{res['ttest_p']:.2e}" if not np.isnan(res.get('ttest_p', float('nan'))) else "n/a"
        w_p_str = f"{res['wilcoxon_p']:.2e}" if not np.isnan(res.get('wilcoxon_p', float('nan'))) else "n/a"
        pwr_str = f"{res['power']:.3f}"
        print(f"  {res['comparison']:<33} {res['task']:<15} {res['condition_label']:<18} "
              f"{res['n_abstracts']:>4} {d_str:>8} {t_p_str:>10} {w_p_str:>10} {pwr_str:>6}")

    # ===================================================================
    # SECTION 8: LaTeX snippet
    # ===================================================================
    print("\n" + "=" * 90)
    print("SECTION 8: LaTeX SNIPPET FOR PAPER")
    print("=" * 90)

    latex = generate_latex_snippet(paired_results, aggregate_results, model_agg_results)
    print(latex)

    # ===================================================================
    # Save all results
    # ===================================================================
    output = {
        "description": "Statistical tests for 5-model reproducibility study",
        "n_bootstrap": N_BOOTSTRAP,
        "alpha": ALPHA,
        "bootstrap_cis_per_condition": bootstrap_results,
        "paired_comparisons": sanitize_for_json(paired_results),
        "aggregate_comparisons": sanitize_for_json(aggregate_results),
        "per_model_aggregate_greedy": model_agg_results,
        "latex_snippet": latex,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {OUTPUT_JSON}")
    print("Done.")


def sanitize_for_json(data):
    """Replace NaN/inf with string representations for JSON serialization."""
    if isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, float):
        if np.isnan(data):
            return "NaN"
        elif np.isinf(data):
            return "Inf" if data > 0 else "-Inf"
        return data
    return data


def generate_latex_snippet(paired_results, aggregate_results, model_agg_results):
    """Generate LaTeX table snippet for the paper."""
    lines = []
    lines.append("% --- Statistical tests LaTeX snippet (auto-generated) ---")
    lines.append("% Include in paper with \\input{statistical_tests_table.tex}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Statistical comparisons of output reproducibility (EMR) "
                  "between local and API-hosted models under greedy decoding conditions. "
                  "Paired $t$-tests and Wilcoxon signed-rank tests are reported with "
                  "Cohen's $d$ effect sizes and bootstrap 95\\% CIs.}")
    lines.append("\\label{tab:statistical-tests}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llccrrl}")
    lines.append("\\toprule")
    lines.append("Comparison & Task & $n$ & Cohen's $d$ & $t$-test $p$ & Wilcoxon $p$ & Sig. \\\\")
    lines.append("\\midrule")

    for res in paired_results:
        # Only include greedy conditions in the main table
        norm_cond = normalize_condition(res.get("condition_a", ""))
        if norm_cond not in ("C1_greedy", "C2_greedy", "C3_temp0.0"):
            continue

        comp = res["comparison"].replace("vs", "vs.")
        task = res["task"].replace("_", " ").title()
        n = res["n_abstracts"]
        d_val = res["cohens_d"]
        d_str = f"{d_val:.2f}" if isinstance(d_val, float) and abs(d_val) != float("inf") else "$\\infty$"
        t_p = res.get("ttest_p", float("nan"))
        t_p_str = f"{t_p:.1e}" if not np.isnan(t_p) else "---"
        w_p = res.get("wilcoxon_p", float("nan"))
        w_p_str = f"{w_p:.1e}" if not np.isnan(w_p) else "---"
        sig = res.get("ttest_sig", "n.s.")
        sig_map = {"***": "$^{***}$", "**": "$^{**}$", "*": "$^{*}$", "n.s.": "n.s."}
        sig_latex = sig_map.get(sig, sig)

        cond_short = res.get("condition_label", "")
        lines.append(f"{comp} & {task} ({cond_short}) & {n} & {d_str} & "
                      f"{t_p_str} & {w_p_str} & {sig_latex} \\\\")

    lines.append("\\midrule")

    # Aggregate rows
    for res in aggregate_results:
        comp = res["comparison"]
        task = res.get("task", "").replace("_", " ").title()
        na, nb = res.get("n_a", 0), res.get("n_b", 0)
        d_val = res["cohens_d"]
        d_str = f"{d_val:.2f}" if isinstance(d_val, float) and abs(d_val) != float("inf") else "$\\infty$"
        t_p = res.get("ttest_p", float("nan"))
        t_p_str = f"{t_p:.1e}" if not np.isnan(t_p) else "---"
        mw_p = res.get("mannwhitney_p", float("nan"))
        mw_p_str = f"{mw_p:.1e}" if not np.isnan(mw_p) else "---"
        sig = res.get("ttest_sig", "n.s.")
        sig_map = {"***": "$^{***}$", "**": "$^{**}$", "*": "$^{*}$", "n.s.": "n.s."}
        sig_latex = sig_map.get(sig, sig)

        lines.append(f"{comp} & {task} & {na}+{nb} & {d_str} & "
                      f"{t_p_str} & {mw_p_str}$^\\dagger$ & {sig_latex} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{2pt}")
    lines.append("\\raggedright\\footnotesize")
    lines.append("$^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$. "
                  "$^\\dagger$Mann--Whitney $U$ (independent samples).")
    lines.append("\\end{table}")

    # Also add a per-model summary table
    lines.append("")
    lines.append("% --- Per-model aggregate EMR with bootstrap CIs ---")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Per-model aggregate EMR under greedy decoding "
                  "(C1/C2/C3\\_t0) with bootstrap 95\\% confidence intervals.}")
    lines.append("\\label{tab:model-emr-ci}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccl}")
    lines.append("\\toprule")
    lines.append("Model & Type & $n$ & Mean EMR & 95\\% CI \\\\")
    lines.append("\\midrule")

    nice_names = {
        "llama3_8b": "LLaMA 3 8B",
        "mistral_7b": "Mistral 7B",
        "gemma2_9b": "Gemma 2 9B",
        "gpt4": "GPT-4",
        "claude_sonnet": "Claude Sonnet 4.5",
    }

    for model in ["gemma2_9b", "mistral_7b", "llama3_8b", "gpt4", "claude_sonnet"]:
        if model not in model_agg_results:
            continue
        info = model_agg_results[model]
        mtype = "Local" if model in LOCAL_MODELS else "API"
        name = nice_names.get(model, model)
        lines.append(f"{name} & {mtype} & {info['n']} & {info['mean']:.3f} & "
                      f"[{info['ci_low']:.3f}, {info['ci_high']:.3f}] \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
