#!/usr/bin/env python3
"""Generate matplotlib figures for the expanded reproducibility paper (v4: with bootstrap CIs).

Reads expanded_metrics.json and raw run files to produce publication-ready
PDF figures with bootstrap 95% confidence intervals and error bars.

Outputs (in analysis/figures/):
  - fig_emr_heatmap.pdf          -- EMR heatmap with +/- sigma annotations
  - fig_temp_effect.pdf           -- Temperature effect with shaded CI bands
  - fig_multiturn_comparison.pdf  -- Multi-turn grouped bars with CI error bars
  - fig_api_vs_local.pdf         -- API vs Local comparison with bootstrap CIs
  - fig_ned_comparison.pdf       -- NED grouped bars with error bars
  - fig_three_level_radar.pdf    -- Radar chart (5 axes, local solid / API dashed)

Usage:
    /Users/lucasrover/paper-experiment/.venv/bin/python analysis/regenerate_figures_v4.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ANALYSIS_DIR = Path(__file__).parent
FIGURES_DIR = ANALYSIS_DIR / "figures"
RUNS_DIR = ANALYSIS_DIR.parent / "outputs" / "runs"

# ──────────────────────────────────────────────────────────────────────────────
# Publication style
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
})

# ──────────────────────────────────────────────────────────────────────────────
# Model metadata
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAMES = {
    "gemma2_9b": "Gemma 2\n9B",
    "mistral_7b": "Mistral\n7B",
    "llama3_8b": "LLaMA 3\n8B",
    "gpt4": "GPT-4",
    "claude_sonnet": "Claude\nSonnet 4.5",
}

MODEL_NAMES_SHORT = {
    "gemma2_9b": "Gemma 2 9B",
    "mistral_7b": "Mistral 7B",
    "llama3_8b": "LLaMA 3 8B",
    "gpt4": "GPT-4",
    "claude_sonnet": "Claude Sonnet 4.5",
}

MODEL_ORDER = ["gemma2_9b", "mistral_7b", "llama3_8b", "gpt4", "claude_sonnet"]
LOCAL_MODELS = {"gemma2_9b", "mistral_7b", "llama3_8b"}
API_MODELS = {"gpt4", "claude_sonnet"}

MODEL_COLORS = {
    "gemma2_9b": "#2196F3",
    "mistral_7b": "#4CAF50",
    "llama3_8b": "#1565C0",
    "gpt4": "#E53935",
    "claude_sonnet": "#FF6F00",
}

LOCAL_COLOR = "#1976D2"
API_COLOR = "#D32F2F"

N_BOOTSTRAP = 10_000
SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_metrics():
    """Load expanded_metrics.json."""
    with open(ANALYSIS_DIR / "expanded_metrics.json") as f:
        return json.load(f)


def lookup(metrics, model, task, condition):
    for m in metrics:
        if m["model"] == model and m["task"] == task and m["condition"] == condition:
            return m
    return None


def lookup_greedy(metrics, model, task):
    for cond in ["C1_fixed_seed", "C2_var_seed", "C2_same_params"]:
        m = lookup(metrics, model, task, cond)
        if m is not None:
            return m
    return None


def lookup_temp(metrics, model, task, temp_label):
    variants = {
        "0.0": ["C3_temp0.0", "C3_temp0_0"],
        "0.3": ["C3_temp0.3", "C3_temp0_3"],
        "0.7": ["C3_temp0.7", "C3_temp0_7"],
    }
    for cond in variants.get(temp_label, []):
        m = lookup(metrics, model, task, cond)
        if m is not None:
            return m
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Raw run loading and per-abstract EMR computation
# ──────────────────────────────────────────────────────────────────────────────

def identify_model(run_data):
    model_name = run_data.get("model_name", "").lower()
    run_id = run_data.get("run_id", "").lower()
    if "mistral" in model_name or "mistral" in run_id:
        return "mistral_7b"
    elif "gemma" in model_name or "gemma" in run_id:
        return "gemma2_9b"
    elif "claude" in model_name or "sonnet" in run_id:
        return "claude_sonnet"
    elif "gpt" in model_name or "gpt" in run_id:
        return "gpt4"
    elif "llama" in model_name or "llama" in run_id:
        return "llama3_8b"
    return "unknown"


def identify_task(run_data):
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


def identify_condition(run_data):
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


def identify_abstract(run_data):
    run_id = run_data.get("run_id", "")
    parts = run_id.split("_")
    for i, part in enumerate(parts):
        if part == "abs" and i + 1 < len(parts):
            return f"abs_{parts[i + 1]}"
    return "unknown"


def normalize_condition(cond):
    """Normalize condition for matching between raw runs and expanded_metrics.json."""
    return cond


def _load_all_grouped_outputs():
    """Load all raw runs and group outputs by (model, task, condition, abstract).

    Returns dict: (model, task, condition, abstract) -> list of output strings.
    """
    groups = defaultdict(list)
    run_files = sorted(RUNS_DIR.glob("*.json"))
    print(f"  Found {len(run_files)} run files")

    for f in run_files:
        try:
            with open(f) as fp:
                run = json.load(fp)
        except Exception:
            continue

        model = identify_model(run)
        task = identify_task(run)
        condition = identify_condition(run)
        abstract = identify_abstract(run)
        output = run.get("output_text", "").strip()

        if not output:
            continue

        groups[(model, task, condition, abstract)].append(output)

    return groups


def load_per_abstract_emr():
    """Load all raw runs and compute per-abstract pairwise EMR values.

    Uses the same pairwise exact-match-rate as compute_all_metrics in variability.py:
    fraction of all unique pairs that are identical.

    Returns dict: (model, task, condition) -> list of per-abstract EMR values.
    """
    print("Loading raw run files for bootstrap analysis...")
    groups = _load_all_grouped_outputs()

    per_abstract_emr = defaultdict(list)
    for (model, task, condition, abstract), outputs in sorted(groups.items()):
        if len(outputs) < 2:
            continue
        # Pairwise exact match rate (same as exact_match_all_pairs in variability.py)
        n = len(outputs)
        total_pairs = 0
        match_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                if outputs[i] == outputs[j]:
                    match_count += 1
        emr = match_count / total_pairs if total_pairs > 0 else 1.0
        per_abstract_emr[(model, task, condition)].append(emr)

    print(f"  Computed per-abstract EMR for {len(per_abstract_emr)} (model, task, condition) groups")
    return per_abstract_emr


def load_per_abstract_ned():
    """Load all raw runs and compute per-abstract NED values.

    Uses the python-Levenshtein library (same as variability.py) for accurate
    pairwise normalized edit distance.

    Returns dict: (model, task, condition) -> list of per-abstract NED values.
    """
    try:
        import Levenshtein as lev
        use_lev = True
    except ImportError:
        use_lev = False
        print("  Warning: python-Levenshtein not available, using fallback")

    groups = _load_all_grouped_outputs()

    per_abstract_ned = defaultdict(list)
    for (model, task, condition, abstract), outputs in sorted(groups.items()):
        if len(outputs) < 2:
            continue
        # Compute mean pairwise NED (same method as edit_distance_stats)
        neds = []
        n = len(outputs)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = outputs[i], outputs[j]
                max_len = max(len(a), len(b), 1)
                if use_lev:
                    dist = lev.distance(a, b)
                else:
                    dist = _levenshtein(a, b)
                neds.append(dist / max_len)
        ned_val = float(np.mean(neds)) if neds else 0.0
        per_abstract_ned[(model, task, condition)].append(ned_val)

    return per_abstract_ned


def _levenshtein(s1, s2):
    """Compute Levenshtein distance between two strings (fallback).
    Uses optimized two-row approach for memory efficiency.
    """
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap helpers
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=0.95, seed=SEED):
    """Compute bootstrap confidence interval for the mean of `values`.

    Parameters
    ----------
    values : array-like
        Per-abstract metric values.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (0.95 = 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mean, ci_low, ci_high : floats
    """
    values = np.array(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return values[0], values[0], values[0]

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = values[rng.randint(0, n, size=n)]
        boot_means[i] = np.mean(sample)

    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_means, 100 * alpha)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha))
    return float(np.mean(values)), float(ci_low), float(ci_high)


def get_greedy_condition(model):
    """Return the greedy condition name for a model."""
    if model == "gpt4":
        return "C2_same_params"
    return "C1_fixed_seed"


def get_temp_conditions(model, temp_label):
    """Return possible condition names for a given temperature."""
    variants = {
        "0.0": ["C3_temp0.0", "C3_temp0_0"],
        "0.3": ["C3_temp0.3", "C3_temp0_3"],
        "0.7": ["C3_temp0.7", "C3_temp0_7"],
    }
    return variants.get(temp_label, [])


def find_per_abstract(per_abstract_data, model, task, condition_variants):
    """Find per-abstract data matching any of the condition variants."""
    if isinstance(condition_variants, str):
        condition_variants = [condition_variants]
    for cond in condition_variants:
        key = (model, task, cond)
        if key in per_abstract_data:
            return per_abstract_data[key]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: EMR Heatmap with +/- sigma
# ──────────────────────────────────────────────────────────────────────────────

def generate_emr_heatmap(metrics):
    """Heatmap of EMR under greedy decoding for 5 models x 2 tasks.
    Green=high, red=low, white line separating local/API.
    Add +/- sigma in smaller text below each value.
    """
    tasks_conds = [
        ("extraction", "Extraction\n(greedy)"),
        ("summarization", "Summarization\n(greedy)"),
    ]

    data = np.full((len(MODEL_ORDER), len(tasks_conds)), np.nan)
    stds = np.full((len(MODEL_ORDER), len(tasks_conds)), np.nan)

    for i, model in enumerate(MODEL_ORDER):
        for j, (task, _) in enumerate(tasks_conds):
            m = lookup_greedy(metrics, model, task)
            if m and m["emr_mean"] is not None:
                data[i, j] = m["emr_mean"]
                stds[i, j] = m.get("emr_std", 0) or 0

    fig, ax = plt.subplots(figsize=(6, 5.5))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(tasks_conds)))
    ax.set_xticklabels([label for _, label in tasks_conds], fontsize=10)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_NAMES[m] for m in MODEL_ORDER], fontsize=10)

    # Annotate cells with value and +/- sigma
    for i in range(len(MODEL_ORDER)):
        for j in range(len(tasks_conds)):
            val = data[i, j]
            std = stds[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                # Main EMR value
                ax.text(j, i - 0.12, f"{val:.3f}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)
                # Sigma below
                if not np.isnan(std):
                    ax.text(j, i + 0.22, f"\u00b1{std:.3f}", ha="center", va="center",
                            fontsize=8, color=color, alpha=0.8)

    # White line separating local (rows 0-2) from API (rows 3-4)
    ax.axhline(y=2.5, color="white", linewidth=3)

    # Add local / API labels
    ax.text(-0.8, 1.0, "Local", ha="center", va="center", fontsize=10,
            fontweight="bold", color=LOCAL_COLOR, rotation=90,
            transform=ax.transData)
    ax.text(-0.8, 3.5, "API", ha="center", va="center", fontsize=10,
            fontweight="bold", color=API_COLOR, rotation=90,
            transform=ax.transData)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Exact Match Rate (EMR)", fontsize=10)

    ax.set_title("Bitwise Reproducibility Under Greedy Decoding", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_emr_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [1/6] Generated: fig_emr_heatmap.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Temperature effect with shaded CI bands
# ──────────────────────────────────────────────────────────────────────────────

def generate_temp_effect(metrics, per_abstract_emr):
    """Two-panel line plot (extraction, summarization) showing EMR vs temperature.
    Solid lines for local, dashed for API.
    Shaded CI bands (bootstrap 95% CI).
    Claude line shows anomalous peak at t=0.3.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    temps = ["0.0", "0.3", "0.7"]
    x = [0.0, 0.3, 0.7]

    for idx, task in enumerate(["extraction", "summarization"]):
        ax = axes[idx]

        for model in MODEL_ORDER:
            emrs = []
            ci_lows = []
            ci_highs = []
            valid = False

            for temp in temps:
                # Look up from expanded_metrics.json
                m = lookup_temp(metrics, model, task, temp)
                if m and m["emr_mean"] is not None:
                    emr_val = m["emr_mean"]
                    valid = True
                else:
                    emr_val = np.nan

                emrs.append(emr_val)

                # Bootstrap CI from per-abstract data
                cond_variants = get_temp_conditions(model, temp)
                pa = find_per_abstract(per_abstract_emr, model, task, cond_variants)
                if pa is not None and len(pa) >= 2:
                    _, lo, hi = bootstrap_ci(pa)
                    ci_lows.append(lo)
                    ci_highs.append(hi)
                else:
                    ci_lows.append(emr_val if not np.isnan(emr_val) else 0)
                    ci_highs.append(emr_val if not np.isnan(emr_val) else 0)

            if not valid:
                continue

            is_local = model in LOCAL_MODELS
            marker = "o" if is_local else "s"
            linestyle = "-" if is_local else "--"
            linewidth = 2.0

            emrs_arr = np.array(emrs)
            ci_lows_arr = np.array(ci_lows)
            ci_highs_arr = np.array(ci_highs)

            # Plot line
            ax.plot(x, emrs_arr, marker=marker, linestyle=linestyle,
                    color=MODEL_COLORS[model], linewidth=linewidth, markersize=8,
                    label=MODEL_NAMES_SHORT[model], zorder=3)

            # Shaded CI band
            mask = ~np.isnan(emrs_arr)
            x_valid = np.array(x)[mask]
            lo_valid = ci_lows_arr[mask]
            hi_valid = ci_highs_arr[mask]
            ax.fill_between(x_valid, lo_valid, hi_valid,
                            color=MODEL_COLORS[model], alpha=0.12, zorder=1)

        ax.set_xlabel("Temperature", fontsize=11)
        title = "Extraction" if task == "extraction" else "Summarization"
        ax.set_title(f"({'a' if idx == 0 else 'b'}) {title}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xlim(-0.05, 0.75)
        ax.set_ylim(-0.05, 1.12)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Exact Match Rate (EMR)", fontsize=11)
    axes[1].legend(fontsize=9, loc="upper right", framealpha=0.9)

    fig.suptitle("Effect of Sampling Temperature on Reproducibility",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_temp_effect.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [2/6] Generated: fig_temp_effect.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Multi-turn comparison with bootstrap CI error bars
# ──────────────────────────────────────────────────────────────────────────────

def generate_multiturn_comparison(metrics, per_abstract_emr):
    """Grouped bar chart: EMR for 3 local models across 4 scenarios.
    Bootstrap CI error bars.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    local_models = ["gemma2_9b", "mistral_7b", "llama3_8b"]
    scenarios = [
        ("extraction", "C1_fixed_seed", "Single-turn\nExtraction"),
        ("summarization", "C1_fixed_seed", "Single-turn\nSummarization"),
        ("multiturn_refinement", "C1_fixed_seed", "Multi-turn\nRefinement"),
        ("rag_extraction", "C1_fixed_seed", "RAG\nExtraction"),
    ]

    x = np.arange(len(scenarios))
    width = 0.25
    offsets = [-width, 0, width]

    for i, model in enumerate(local_models):
        emrs = []
        err_lo = []
        err_hi = []

        for task_id, cond, _ in scenarios:
            m = lookup(metrics, model, task_id, cond)
            if m and m["emr_mean"] is not None:
                mean_val = m["emr_mean"]
            else:
                mean_val = 0.0

            # Bootstrap CI
            pa = find_per_abstract(per_abstract_emr, model, task_id, cond)
            if pa is not None and len(pa) >= 2:
                bmean, lo, hi = bootstrap_ci(pa)
                emrs.append(bmean)
                err_lo.append(bmean - lo)
                err_hi.append(hi - bmean)
            else:
                emrs.append(mean_val)
                err_lo.append(0)
                err_hi.append(0)

        bars = ax.bar(x + offsets[i], emrs, width,
                      yerr=[err_lo, err_hi],
                      label=MODEL_NAMES_SHORT[model], color=MODEL_COLORS[model],
                      alpha=0.85, edgecolor="white", capsize=3,
                      error_kw={"linewidth": 1.2, "capthick": 1.2})

        for bar, val in zip(bars, emrs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.04,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Exact Match Rate (EMR)", fontsize=11)
    ax.set_title("Reproducibility Across Interaction Regimes (C1, t=0)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in scenarios], fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_multiturn_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [3/6] Generated: fig_multiturn_comparison.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: API vs Local with bootstrap CI error bars
# ──────────────────────────────────────────────────────────────────────────────

def generate_api_vs_local(metrics, per_abstract_emr):
    """Side-by-side comparison of local vs API average metrics.
    Bootstrap CI error bars on each bar.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    metric_labels = ["EMR", "1-NED", "ROUGE-L", "BERTScore F1"]
    metric_keys = ["emr_mean", "ned_mean", "rouge_l_mean", "bertscore_f1_mean"]

    greedy_conds = ["C1_fixed_seed", "C2_var_seed", "C2_same_params"]

    # Collect per-model greedy values for each metric
    # For bootstrap on EMR, we have per-abstract data
    # For other metrics, we use the model-level values and bootstrap across models

    local_vals = []
    local_err_lo = []
    local_err_hi = []
    api_vals = []
    api_err_lo = []
    api_err_hi = []

    for mk_idx, metric_key in enumerate(metric_keys):
        for model_set, vals_list, err_lo_list, err_hi_list in [
            (LOCAL_MODELS, local_vals, local_err_lo, local_err_hi),
            (API_MODELS, api_vals, api_err_lo, api_err_hi),
        ]:
            if metric_key == "emr_mean":
                # Collect all per-abstract EMR values across models in this set
                all_abstract_emrs = []
                for model in model_set:
                    for cond in greedy_conds:
                        pa = find_per_abstract(per_abstract_emr, model, "extraction", cond)
                        if pa is not None:
                            all_abstract_emrs.extend(pa)
                        pa = find_per_abstract(per_abstract_emr, model, "summarization", cond)
                        if pa is not None:
                            all_abstract_emrs.extend(pa)

                if all_abstract_emrs:
                    bmean, lo, hi = bootstrap_ci(all_abstract_emrs)
                    vals_list.append(bmean)
                    err_lo_list.append(bmean - lo)
                    err_hi_list.append(hi - bmean)
                else:
                    vals_list.append(0)
                    err_lo_list.append(0)
                    err_hi_list.append(0)

            else:
                # For NED, ROUGE-L, BERTScore: collect per-model-task values
                collected = []
                for m in metrics:
                    if m["model"] in model_set and m["condition"] in greedy_conds:
                        val = m.get(metric_key)
                        if val is not None:
                            if metric_key == "ned_mean":
                                collected.append(1.0 - val)
                            else:
                                collected.append(val)

                if collected:
                    bmean, lo, hi = bootstrap_ci(collected)
                    vals_list.append(bmean)
                    err_lo_list.append(bmean - lo)
                    err_hi_list.append(hi - bmean)
                else:
                    vals_list.append(0)
                    err_lo_list.append(0)
                    err_hi_list.append(0)

    x = np.arange(len(metric_labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, local_vals, width,
                   yerr=[local_err_lo, local_err_hi],
                   label="Local (3 models)",
                   color=LOCAL_COLOR, alpha=0.85, edgecolor="white",
                   capsize=4, error_kw={"linewidth": 1.2, "capthick": 1.2})
    bars2 = ax.bar(x + width / 2, api_vals, width,
                   yerr=[api_err_lo, api_err_hi],
                   label="API (2 models)",
                   color=API_COLOR, alpha=0.85, edgecolor="white",
                   capsize=4, error_kw={"linewidth": 1.2, "capthick": 1.2})

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score (higher = more reproducible)", fontsize=11)
    ax.set_title("API vs Local Models: Reproducibility Under Greedy Decoding",
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_api_vs_local.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [4/6] Generated: fig_api_vs_local.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: NED comparison with error bars
# ──────────────────────────────────────────────────────────────────────────────

def generate_ned_comparison(metrics, per_abstract_ned):
    """Grouped bar chart of NED for 5 models x 2 tasks with bootstrap CI error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    tasks = ["extraction", "summarization"]
    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    task_colors = {"extraction": "#1976D2", "summarization": "#FF8F00"}

    for i, task in enumerate(tasks):
        neds = []
        err_lo = []
        err_hi = []

        for model in MODEL_ORDER:
            m = lookup_greedy(metrics, model, task)
            if m and m["ned_mean"] is not None:
                mean_ned = m["ned_mean"]
            else:
                mean_ned = 0.0

            # Bootstrap from per-abstract NED
            greedy_cond = get_greedy_condition(model)
            pa = find_per_abstract(per_abstract_ned, model, task, greedy_cond)
            if pa is None:
                # Try other greedy conditions
                for cond in ["C1_fixed_seed", "C2_var_seed", "C2_same_params"]:
                    pa = find_per_abstract(per_abstract_ned, model, task, cond)
                    if pa is not None:
                        break

            if pa is not None and len(pa) >= 2:
                bmean, lo, hi = bootstrap_ci(pa)
                neds.append(bmean)
                err_lo.append(bmean - lo)
                err_hi.append(hi - bmean)
            else:
                neds.append(mean_ned)
                err_lo.append(0)
                err_hi.append(0)

        color = task_colors[task]
        bars = ax.bar(x + (i - 0.5) * width, neds, width,
                      yerr=[err_lo, err_hi],
                      label="Extraction" if task == "extraction" else "Summarization",
                      color=color, alpha=0.85, edgecolor="white",
                      capsize=3, error_kw={"linewidth": 1.2, "capthick": 1.2})

        for bar, val in zip(bars, neds):
            if val > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Normalized Edit Distance (NED)", fontsize=11)
    ax.set_title("Surface-Level Variability Under Greedy Decoding", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES_SHORT[m] for m in MODEL_ORDER], fontsize=9)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ned_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [5/6] Generated: fig_ned_comparison.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6: Radar chart — 5 axes, local solid / API dashed
# ──────────────────────────────────────────────────────────────────────────────

def generate_radar_chart(metrics):
    """Radar chart with 5 axes: EMR ext, EMR sum, 1-NED, ROUGE-L, BERTScore.
    Solid lines for local, dashed for API.
    """
    categories = [
        "EMR\n(Extraction)",
        "EMR\n(Summarization)",
        "1-NED\n(Extraction)",
        "ROUGE-L\n(Extraction)",
        "BERTScore\n(Extraction)",
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in MODEL_ORDER:
        values = []

        # EMR extraction
        m_ext = lookup_greedy(metrics, model, "extraction")
        values.append(m_ext["emr_mean"] if m_ext and m_ext["emr_mean"] is not None else 0)

        # EMR summarization
        m_sum = lookup_greedy(metrics, model, "summarization")
        values.append(m_sum["emr_mean"] if m_sum and m_sum["emr_mean"] is not None else 0)

        # 1-NED extraction
        ned = m_ext["ned_mean"] if m_ext and m_ext["ned_mean"] is not None else 1
        values.append(1.0 - ned)

        # ROUGE-L extraction
        values.append(m_ext["rouge_l_mean"] if m_ext and m_ext["rouge_l_mean"] is not None else 0)

        # BERTScore extraction
        values.append(m_ext["bertscore_f1_mean"] if m_ext and m_ext["bertscore_f1_mean"] is not None else 0)

        values += values[:1]

        is_local = model in LOCAL_MODELS
        linestyle = "-" if is_local else "--"
        linewidth = 2.2 if is_local else 2.0
        ax.plot(angles, values, linewidth=linewidth, linestyle=linestyle,
                color=MODEL_COLORS[model], label=MODEL_NAMES_SHORT[model])
        ax.fill(angles, values, alpha=0.06, color=MODEL_COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Three-Level Reproducibility Profile\n(Greedy Decoding)",
                 fontsize=12, pad=20)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_three_level_radar.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [6/6] Generated: fig_three_level_radar.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load aggregate metrics
    metrics = load_metrics()
    print(f"Loaded {len(metrics)} metric entries from expanded_metrics.json")

    # Load per-abstract data from raw runs for bootstrap
    per_abstract_emr = load_per_abstract_emr()
    per_abstract_ned = load_per_abstract_ned()

    # Debug: print sample of per-abstract data
    print("\nPer-abstract EMR samples:")
    for key in sorted(per_abstract_emr.keys())[:5]:
        vals = per_abstract_emr[key]
        print(f"  {key}: n={len(vals)}, mean={np.mean(vals):.3f}, vals={vals[:5]}...")

    print(f"\nGenerating 6 figures in {FIGURES_DIR}/...\n")

    generate_emr_heatmap(metrics)
    generate_temp_effect(metrics, per_abstract_emr)
    generate_multiturn_comparison(metrics, per_abstract_emr)
    generate_api_vs_local(metrics, per_abstract_emr)
    generate_ned_comparison(metrics, per_abstract_ned)
    generate_radar_chart(metrics)

    print(f"\nAll 6 figures generated in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
