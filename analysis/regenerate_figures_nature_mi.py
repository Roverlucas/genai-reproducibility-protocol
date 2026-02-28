#!/usr/bin/env python3
"""Generate Nature Machine Intelligence-style figures for the reproducibility paper.

Follows Nature MI published figure standards:
  - Sans-serif font (Helvetica/Arial), 5-7pt labels
  - Max width 180mm (~7.09in), single column ~88mm (~3.46in)
  - Compact, information-dense, colorblind-friendly palette
  - Panel labels: bold lowercase a, b, c, d
  - Clean axes, minimal gridlines, high contrast
  - 300+ DPI output

Outputs (in article/figures/):
  - fig_emr_heatmap.pdf          -- EMR heatmap (2 panels: local vs API)
  - fig_temp_effect.pdf           -- Temperature effect (2 panels: a, b)
  - fig_multiturn_comparison.pdf  -- Multi-turn + API grouped bars (5 models)
  - fig_three_level_radar.pdf     -- Radar chart (5 axes)
  - fig_visual_abstract.pdf       -- Study overview pipeline

Usage:
    /Users/lucasrover/paper-experiment/.venv/bin/python analysis/regenerate_figures_nature_mi.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ANALYSIS_DIR = Path(__file__).parent
ARTICLE_FIGURES_DIR = ANALYSIS_DIR.parent / "article" / "figures"
RUNS_DIR = ANALYSIS_DIR.parent / "outputs" / "runs"

# ──────────────────────────────────────────────────────────────────────────────
# Nature MI publication style
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 6,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "text.usetex": False,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "patch.linewidth": 0.5,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.3,
})

# ──────────────────────────────────────────────────────────────────────────────
# Nature MI color palette (colorblind-friendly, inspired by published papers)
# ──────────────────────────────────────────────────────────────────────────────
# Using a palette similar to Okabe-Ito + Nature MI blue/orange accent
COLORS = {
    "gemma2_9b":     "#0072B2",  # strong blue
    "mistral_7b":    "#009E73",  # teal green
    "llama3_8b":     "#56B4E9",  # sky blue
    "gpt4":          "#D55E00",  # vermillion/red-orange
    "claude_sonnet": "#CC79A7",  # reddish purple
    "gemini":        "#E69F00",  # amber/orange
}

LOCAL_COLOR = "#0072B2"   # blue
API_COLOR = "#D55E00"     # vermillion

# Lighter versions for fill
LOCAL_FILL = "#56B4E9"
API_FILL = "#CC79A7"

# ──────────────────────────────────────────────────────────────────────────────
# Model metadata
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ORDER = ["gemma2_9b", "mistral_7b", "llama3_8b", "gpt4", "claude_sonnet"]
LOCAL_MODELS = {"gemma2_9b", "mistral_7b", "llama3_8b"}
API_MODELS = {"gpt4", "claude_sonnet"}

MODEL_LABELS = {
    "gemma2_9b": "Gemma 2 9B",
    "mistral_7b": "Mistral 7B",
    "llama3_8b": "LLaMA 3 8B",
    "gpt4": "GPT-4",
    "claude_sonnet": "Claude Sonnet 4.5",
    "gemini": "Gemini 2.5 Pro",
}

# Nature MI figure dimensions (in inches)
# Full width = 180mm = 7.09in, single column = 88mm = 3.46in
FULL_WIDTH = 7.09
SINGLE_COL = 3.46
DOUBLE_COL = 5.51  # 140mm, intermediate

N_BOOTSTRAP = 10_000
SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Data loading (reused from v4)
# ──────────────────────────────────────────────────────────────────────────────

def load_metrics():
    with open(ANALYSIS_DIR / "expanded_metrics.json") as f:
        return json.load(f)


def lookup(metrics, model, task, condition):
    for m in metrics:
        if m["model"] == model and m["task"] == task and m["condition"] == condition:
            return m
    return None


def lookup_greedy(metrics, model, task):
    if model == "gpt4":
        preferred = ["C2_same_params", "C1_fixed_seed"]
    else:
        preferred = ["C1_fixed_seed", "C2_var_seed", "C2_same_params"]
    for cond in preferred:
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


def _load_all_grouped_outputs():
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
    print("Loading raw run files for bootstrap analysis...")
    groups = _load_all_grouped_outputs()
    per_abstract_emr = defaultdict(list)
    for (model, task, condition, abstract), outputs in sorted(groups.items()):
        if len(outputs) < 2:
            continue
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
    print(f"  Computed per-abstract EMR for {len(per_abstract_emr)} groups")
    return per_abstract_emr


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=0.95, seed=SEED):
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
    if model == "gpt4":
        return "C2_same_params"
    return "C1_fixed_seed"


def get_temp_conditions(model, temp_label):
    variants = {
        "0.0": ["C3_temp0.0", "C3_temp0_0"],
        "0.3": ["C3_temp0.3", "C3_temp0_3"],
        "0.7": ["C3_temp0.7", "C3_temp0_7"],
    }
    return variants.get(temp_label, [])


def find_per_abstract(per_abstract_data, model, task, condition_variants):
    if isinstance(condition_variants, str):
        condition_variants = [condition_variants]
    for cond in condition_variants:
        key = (model, task, cond)
        if key in per_abstract_data:
            return per_abstract_data[key]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Helper: add panel label (Nature MI style: bold lowercase, top-left)
# ──────────────────────────────────────────────────────────────────────────────

def add_panel_label(ax, label, x=-0.08, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="right")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: clean axis style
# ──────────────────────────────────────────────────────────────────────────────

def clean_axes(ax, grid_y=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_y:
        ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.3)
        ax.set_axisbelow(True)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: EMR Heatmap — compact Nature MI style
# ──────────────────────────────────────────────────────────────────────────────

def generate_emr_heatmap(metrics):
    """Compact 2-panel heatmap: (a) Local, (b) API under greedy decoding."""

    with open(ANALYSIS_DIR / "bootstrap_cis.json") as f:
        bootstrap = json.load(f)

    tasks = [("extraction", "Extraction"), ("summarization", "Summarization")]
    local_order = ["gemma2_9b", "mistral_7b", "llama3_8b"]
    api_order = ["gpt4", "claude_sonnet"]
    ci_data = bootstrap.get("table3_emr_greedy", {})

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(2.2, 1.8),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.7}
    )

    cmap = plt.cm.RdYlBu

    for ax, model_list, label in [
        (ax_a, local_order, "a"),
        (ax_b, api_order, "b"),
    ]:
        data = np.full((len(model_list), len(tasks)), np.nan)
        for i, model in enumerate(model_list):
            model_ci = ci_data.get(model, {})
            for j, (task, _) in enumerate(tasks):
                task_ci = model_ci.get(task, {})
                if task_ci:
                    data[i, j] = task_ci["mean"]

        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([t[1] for t in tasks], fontsize=5)
        ax.set_yticks(range(len(model_list)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in model_list], fontsize=5)
        ax.tick_params(length=0)

        for i in range(len(model_list)):
            for j in range(len(tasks)):
                val = data[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.3 or val > 0.75 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=5.5, fontweight="bold", color=color)

        is_local = label == "a"
        title = "Local models" if is_local else "API models"
        add_panel_label(ax, label, x=-0.02, y=1.15)
        ax.set_title(title, fontsize=5.5, pad=3,
                     color=LOCAL_COLOR if is_local else API_COLOR,
                     fontweight="bold", loc="left")

    cbar = plt.colorbar(im, ax=[ax_a, ax_b], shrink=0.6, pad=0.1, aspect=18)
    cbar.ax.tick_params(labelsize=4, length=1.5, width=0.3)
    cbar.set_label("EMR", fontsize=5, labelpad=2)
    cbar.outline.set_linewidth(0.4)

    fig.savefig(ARTICLE_FIGURES_DIR / "fig_emr_heatmap.pdf")
    plt.close(fig)
    print("  [1/5] Generated: fig_emr_heatmap.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Multi-turn + RAG comparison (5 models, 4 scenarios)
# ──────────────────────────────────────────────────────────────────────────────

def generate_multiturn_comparison(metrics, per_abstract_emr):
    """Grouped bars: 5 models across 4 scenarios, with bootstrap CI error bars."""

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.4))

    all_models = ["gemma2_9b", "mistral_7b", "llama3_8b", "claude_sonnet", "gemini"]
    scenarios = [
        ("extraction", "C1_fixed_seed", "Single-turn\nextraction"),
        ("summarization", "C1_fixed_seed", "Single-turn\nsummarisation"),
        ("multiturn_refinement", "C1_fixed_seed", "Multi-turn\nrefinement"),
        ("rag_extraction", "C1_fixed_seed", "RAG\nextraction"),
    ]

    # Load Gemini multiturn/RAG from bootstrap_cis.json
    with open(ANALYSIS_DIR / "bootstrap_cis.json") as f:
        bootstrap = json.load(f)
    gemini_mt = bootstrap.get("table5_multiturn_rag", {}).get("gemini-2_5-pro", {})

    x = np.arange(len(scenarios))
    n_models = len(all_models)
    width = 0.15
    offsets = [(i - (n_models - 1) / 2) * width for i in range(n_models)]

    for i, model in enumerate(all_models):
        emrs = []
        err_lo = []
        err_hi = []

        for task_id, cond, _ in scenarios:
            if model == "gemini":
                # Gemini only has multi-turn and RAG
                if task_id == "multiturn_refinement":
                    gd = gemini_mt.get("multiturn_refinement", {})
                    emrs.append(gd.get("mean", 0))
                    lo = gd.get("ci_low", gd.get("mean", 0))
                    hi = gd.get("ci_high", gd.get("mean", 0))
                    emrs_val = emrs[-1]
                    err_lo.append(emrs_val - lo)
                    err_hi.append(hi - emrs_val)
                    continue
                elif task_id == "rag_extraction":
                    gd = gemini_mt.get("rag_extraction", {})
                    emrs.append(gd.get("mean", 0))
                    lo = gd.get("ci_low", gd.get("mean", 0))
                    hi = gd.get("ci_high", gd.get("mean", 0))
                    emrs_val = emrs[-1]
                    err_lo.append(emrs_val - lo)
                    err_hi.append(hi - emrs_val)
                    continue
                else:
                    emrs.append(0)
                    err_lo.append(0)
                    err_hi.append(0)
                    continue

            m = lookup(metrics, model, task_id, cond)
            mean_val = m["emr_mean"] if m and m["emr_mean"] is not None else 0.0

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

        color = COLORS.get(model, "#999999")
        label = MODEL_LABELS.get(model, model)

        # Don't plot zero-height bars for Gemini single-turn
        plot_emrs = []
        plot_err_lo = []
        plot_err_hi = []
        plot_x = []
        for j in range(len(scenarios)):
            if model == "gemini" and j < 2:
                continue
            plot_emrs.append(emrs[j])
            plot_err_lo.append(err_lo[j])
            plot_err_hi.append(err_hi[j])
            plot_x.append(x[j] + offsets[i])

        if plot_emrs:
            ax.bar(plot_x, plot_emrs, width * 0.9,
                   yerr=[plot_err_lo, plot_err_hi],
                   label=label, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.3,
                   capsize=2, error_kw={"linewidth": 0.6, "capthick": 0.6})

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Exact match rate (EMR)", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([s[2] for s in scenarios], fontsize=6)
    ax.legend(fontsize=5.5, loc="upper right", ncol=2,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.8)
    clean_axes(ax)

    # Add local/API bracket annotations
    ax.axvspan(-0.5, 1.5, alpha=0.04, color=LOCAL_COLOR)
    ax.axvspan(1.5, 3.5, alpha=0.04, color=API_COLOR)

    fig.tight_layout()
    fig.savefig(ARTICLE_FIGURES_DIR / "fig_multiturn_comparison.pdf")
    plt.close(fig)
    print("  [2/5] Generated: fig_multiturn_comparison.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Three-level radar — compact Nature MI
# ──────────────────────────────────────────────────────────────────────────────

def generate_radar_chart(metrics):
    """Radar chart: 5 axes, solid local / dashed API."""

    categories = ["EMR\n(extraction)", "EMR\n(summarisation)", "1 \u2212 NED",
                   "ROUGE-L", "BERTScore F1"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Compact size — small square to avoid dominating the page
    fig, ax = plt.subplots(figsize=(2.4, 2.4),
                            subplot_kw=dict(polar=True))

    for model in MODEL_ORDER:
        values = []
        m_ext = lookup_greedy(metrics, model, "extraction")
        values.append(m_ext["emr_mean"] if m_ext and m_ext["emr_mean"] is not None else 0)
        m_sum = lookup_greedy(metrics, model, "summarization")
        values.append(m_sum["emr_mean"] if m_sum and m_sum["emr_mean"] is not None else 0)
        ned = m_ext["ned_mean"] if m_ext and m_ext["ned_mean"] is not None else 1
        values.append(1.0 - ned)
        values.append(m_ext["rouge_l_mean"] if m_ext and m_ext["rouge_l_mean"] is not None else 0)
        values.append(m_ext["bertscore_f1_mean"] if m_ext and m_ext["bertscore_f1_mean"] is not None else 0)
        values += values[:1]

        is_local = model in LOCAL_MODELS
        linestyle = "-" if is_local else "--"
        linewidth = 1.0 if is_local else 0.8
        ax.plot(angles, values, linewidth=linewidth, linestyle=linestyle,
                color=COLORS[model], label=MODEL_LABELS[model])
        ax.fill(angles, values, alpha=0.04, color=COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=5, fontweight="bold")
    ax.tick_params(axis="x", pad=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=4.5)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0],
                  labels=["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=4.5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.05), fontsize=5,
              handlelength=1.2, handletextpad=0.3)

    fig.savefig(ARTICLE_FIGURES_DIR / "fig_three_level_radar.pdf",
                pad_inches=0.08)
    plt.close(fig)
    print("  [3/5] Generated: fig_three_level_radar.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: Temperature effect — compact 2-panel (a, b)
# ──────────────────────────────────────────────────────────────────────────────

def generate_temp_effect(metrics, per_abstract_emr):
    """Two-panel (a, b): EMR vs temperature for 5 models."""

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8), sharey=True)

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
                m = lookup_temp(metrics, model, task, temp)
                if m and m["emr_mean"] is not None:
                    emr_val = m["emr_mean"]
                    valid = True
                else:
                    emr_val = np.nan
                emrs.append(emr_val)

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

            emrs_arr = np.array(emrs)
            ci_lo_arr = np.array(ci_lows)
            ci_hi_arr = np.array(ci_highs)

            ax.plot(x, emrs_arr, marker=marker, linestyle=linestyle,
                    color=COLORS[model], markersize=4,
                    label=MODEL_LABELS[model], zorder=3)

            mask = ~np.isnan(emrs_arr)
            x_v = np.array(x)[mask]
            ax.fill_between(x_v, ci_lo_arr[mask], ci_hi_arr[mask],
                            color=COLORS[model], alpha=0.10, zorder=1)

        ax.set_xlabel("Temperature", fontsize=7)
        title = "Extraction" if task == "extraction" else "Summarisation"
        add_panel_label(ax, chr(97 + idx))  # a, b
        ax.set_title(title, fontsize=7, pad=4, loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xlim(-0.05, 0.75)
        ax.set_ylim(-0.05, 1.12)
        clean_axes(ax)

    axes[0].set_ylabel("Exact match rate (EMR)", fontsize=7)
    axes[1].legend(fontsize=5.5, loc="upper right",
                   handlelength=1.5, handletextpad=0.4)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(ARTICLE_FIGURES_DIR / "fig_temp_effect.pdf")
    plt.close(fig)
    print("  [4/5] Generated: fig_temp_effect.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Visual abstract — compact study overview
# ──────────────────────────────────────────────────────────────────────────────

def generate_visual_abstract():
    """Compact study overview showing protocol pipeline and key results."""

    fig = plt.figure(figsize=(FULL_WIDTH, 3.2))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 2, 1], wspace=0.3)

    # --- Panel a: Protocol pipeline (simplified) ---
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    add_panel_label(ax_a, "a", x=0.0, y=1.04)

    steps = [
        (0.5, 0.88, "Prompt Card", "#E8F5E9", "#2E7D32"),
        (0.5, 0.70, "LLM Inference", "#E3F2FD", "#1565C0"),
        (0.5, 0.52, "Run Card", "#FFF3E0", "#E65100"),
        (0.5, 0.34, "W3C PROV", "#F3E5F5", "#6A1B9A"),
        (0.5, 0.16, "Audit", "#FFEBEE", "#C62828"),
    ]
    box_w, box_h = 0.70, 0.11
    for cx, cy, text, bg, fg in steps:
        rect = FancyBboxPatch((cx - box_w/2, cy - box_h/2), box_w, box_h,
                               boxstyle="round,pad=0.015", facecolor=bg,
                               edgecolor=fg, linewidth=0.6)
        ax_a.add_patch(rect)
        ax_a.text(cx, cy, text, ha="center", va="center",
                  fontsize=5.5, fontweight="bold", color=fg)

    for i in range(len(steps) - 1):
        y1 = steps[i][1] - box_h/2
        y2 = steps[i+1][1] + box_h/2
        ax_a.annotate("", xy=(0.5, y2), xytext=(0.5, y1),
                      arrowprops=dict(arrowstyle="->", color="#666",
                                      lw=0.8, shrinkA=1, shrinkB=1))

    # --- Panel b: EMR bar chart (8 models) ---
    ax_b = fig.add_subplot(gs[1])
    add_panel_label(ax_b, "b", x=-0.04, y=1.04)

    models = ["Gemma 2\n9B", "LLaMA 3\n8B", "Mistral\n7B",
              "DeepSeek", "GPT-4", "Claude\nSonnet 4.5", "Perplexity\nSonar",
              "Gemini\n2.5 Pro"]
    emr_ext = [1.000, 0.987, 0.960, 0.800, 0.443, 0.190, 0.100, np.nan]
    emr_sum = [1.000, 0.947, 0.840, 0.760, 0.230, 0.020, 0.010, np.nan]

    x_pos = np.arange(len(models))
    width = 0.35
    colors_ext = [LOCAL_COLOR]*3 + [API_COLOR]*5
    colors_sum = [LOCAL_FILL]*3 + [API_FILL]*5

    for j in range(len(models)):
        if not np.isnan(emr_ext[j]):
            ax_b.bar(x_pos[j] - width/2, emr_ext[j], width, color=colors_ext[j],
                     alpha=0.85, edgecolor="white", linewidth=0.3)
        if not np.isnan(emr_sum[j]):
            ax_b.bar(x_pos[j] + width/2, emr_sum[j], width, color=colors_sum[j],
                     alpha=0.55, edgecolor="white", linewidth=0.3)

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(models, fontsize=5, rotation=30, ha="right")
    ax_b.set_ylabel("EMR (greedy, t = 0)", fontsize=6)
    ax_b.set_ylim(0, 1.15)

    # Divider line
    ax_b.axvline(x=2.5, color="#999", linestyle=":", linewidth=0.5)
    ax_b.text(1.0, 1.06, "Local", fontsize=5.5, fontweight="bold",
              color=LOCAL_COLOR, ha="center", transform=ax_b.get_xaxis_transform())
    ax_b.text(5.0, 1.06, "API", fontsize=5.5, fontweight="bold",
              color=API_COLOR, ha="center", transform=ax_b.get_xaxis_transform())

    # Averages
    local_avg = np.nanmean([1.000, 0.987, 0.960, 1.000, 0.947, 0.840]) / 1
    # simplify: extraction avg
    ax_b.axhline(y=0.956, xmin=0.02, xmax=0.35, color=LOCAL_COLOR,
                 linestyle="--", linewidth=0.6, alpha=0.7)
    ax_b.axhline(y=0.221, xmin=0.4, xmax=0.98, color=API_COLOR,
                 linestyle="--", linewidth=0.6, alpha=0.7)
    ax_b.text(-0.3, 0.956, "0.956", fontsize=4.5, color=LOCAL_COLOR,
              va="center", fontweight="bold")
    ax_b.text(7.3, 0.221, "0.221", fontsize=4.5, color=API_COLOR,
              va="center", fontweight="bold")

    clean_axes(ax_b)

    # --- Panel c: Key stats ---
    ax_c = fig.add_subplot(gs[2])
    ax_c.axis("off")
    add_panel_label(ax_c, "c", x=0.0, y=1.04)

    stats = [
        ("4,104", "experiments"),
        ("8", "models"),
        ("5", "API providers"),
        ("4", "tasks"),
        ("4.3\u00d7", "local/API gap"),
        ("<1%", "overhead"),
    ]

    for i, (num, desc) in enumerate(stats):
        y = 0.88 - i * 0.155
        ax_c.text(0.05, y, num, fontsize=10, fontweight="bold",
                  color=LOCAL_COLOR, va="center")
        ax_c.text(0.05, y - 0.06, desc, fontsize=5.5,
                  color="#444", va="center")

    fig.savefig(ARTICLE_FIGURES_DIR / "fig_visual_abstract.pdf")
    plt.close(fig)
    print("  [5/5] Generated: fig_visual_abstract.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ARTICLE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    print(f"Loaded {len(metrics)} metric entries")

    per_abstract_emr = load_per_abstract_emr()

    print(f"\nGenerating 5 Nature MI figures...\n")

    generate_emr_heatmap(metrics)
    generate_multiturn_comparison(metrics, per_abstract_emr)
    generate_radar_chart(metrics)
    generate_temp_effect(metrics, per_abstract_emr)
    generate_visual_abstract()

    print(f"\nAll 5 figures generated in {ARTICLE_FIGURES_DIR}/")


if __name__ == "__main__":
    main()
