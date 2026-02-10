#!/usr/bin/env python3
"""Generate matplotlib figures for the expanded reproducibility paper (v3: 5 models).

Reads expanded_metrics.json and produces publication-ready PDF figures.

Outputs (in analysis/figures/):
  - fig_emr_heatmap.pdf          — EMR heatmap: models x task/condition
  - fig_api_vs_local.pdf         — Grouped bar chart: API vs Local
  - fig_temp_effect.pdf          — Line plots: EMR vs temperature
  - fig_three_level_radar.pdf    — Radar chart: L1/L2/L3 per model
  - fig_multiturn_comparison.pdf — Multi-turn vs single-turn bars
  - fig_ned_comparison.pdf       — NED by model under greedy

Usage:
    python analysis/generate_figures_v3.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYSIS_DIR = Path(__file__).parent
FIGURES_DIR = ANALYSIS_DIR / "figures"

# Publication style
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
})

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

MODEL_COLORS = {
    "gemma2_9b": "#2196F3",
    "mistral_7b": "#4CAF50",
    "llama3_8b": "#1565C0",
    "gpt4": "#E53935",
    "claude_sonnet": "#FF6F00",
}

LOCAL_COLOR = "#1976D2"
API_COLOR = "#D32F2F"


def load_data():
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


# ─── Figure 1: EMR Heatmap ──────────────────────────────────────────────────

def generate_emr_heatmap(metrics):
    tasks_conds = [
        ("extraction", "Extraction\nC1 (greedy)"),
        ("summarization", "Summarization\nC1 (greedy)"),
    ]

    data = np.full((len(MODEL_ORDER), len(tasks_conds)), np.nan)
    for i, model in enumerate(MODEL_ORDER):
        for j, (task, _) in enumerate(tasks_conds):
            m = lookup_greedy(metrics, model, task)
            if m and m["emr_mean"] is not None:
                data[i, j] = m["emr_mean"]

    fig, ax = plt.subplots(figsize=(6, 5))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(tasks_conds)))
    ax.set_xticklabels([label for _, label in tasks_conds], fontsize=10)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_NAMES[m] for m in MODEL_ORDER], fontsize=10)

    for i in range(len(MODEL_ORDER)):
        for j in range(len(tasks_conds)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

    ax.axhline(y=2.5, color="white", linewidth=3)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Exact Match Rate (EMR)", fontsize=10)

    ax.set_title("Bitwise Reproducibility Under Greedy Decoding", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_emr_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Generated: fig_emr_heatmap.pdf")


# ─── Figure 2: API vs Local grouped bar chart ───────────────────────────────

def generate_api_vs_local(metrics):
    fig, ax = plt.subplots(figsize=(8, 5))

    metric_labels = ["EMR", "1-NED", "ROUGE-L", "BERTScore F1"]
    local_models = {"llama3_8b", "mistral_7b", "gemma2_9b"}
    api_models = {"gpt4", "claude_sonnet"}

    local_vals = []
    api_vals = []

    for metric_key in ["emr_mean", "ned_mean", "rouge_l_mean", "bertscore_f1_mean"]:
        for model_set, vals_list in [(local_models, local_vals), (api_models, api_vals)]:
            collected = []
            for m in metrics:
                if m["model"] in model_set and m["condition"] in ["C1_fixed_seed", "C2_var_seed", "C2_same_params"]:
                    val = m.get(metric_key)
                    if val is not None:
                        collected.append(val)
            if collected:
                avg = sum(collected) / len(collected)
                if metric_key == "ned_mean":
                    avg = 1.0 - avg
                vals_list.append(avg)
            else:
                vals_list.append(0)

    x = np.arange(len(metric_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, local_vals, width, label="Local (3 models)",
                   color=LOCAL_COLOR, alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, api_vals, width, label="API (2 models)",
                   color=API_COLOR, alpha=0.85, edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score (higher = more reproducible)", fontsize=11)
    ax.set_title("API vs Local Models: Reproducibility Under Greedy Decoding", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_api_vs_local.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Generated: fig_api_vs_local.pdf")


# ─── Figure 3: Temperature effect line plots ────────────────────────────────

def generate_temp_effect(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    temps = ["0.0", "0.3", "0.7"]
    x = [0.0, 0.3, 0.7]

    for idx, task in enumerate(["extraction", "summarization"]):
        ax = axes[idx]
        for model in MODEL_ORDER:
            emrs = []
            for temp in temps:
                m = lookup_temp(metrics, model, task, temp)
                if m and m["emr_mean"] is not None:
                    emrs.append(m["emr_mean"])
                else:
                    emrs.append(np.nan)

            if all(np.isnan(e) for e in emrs):
                continue

            marker = "o" if model in ["llama3_8b", "mistral_7b", "gemma2_9b"] else "s"
            linestyle = "-" if model in ["llama3_8b", "mistral_7b", "gemma2_9b"] else "--"
            ax.plot(x, emrs, marker=marker, linestyle=linestyle,
                    color=MODEL_COLORS[model], linewidth=2, markersize=8,
                    label=MODEL_NAMES_SHORT[model])

        ax.set_xlabel("Temperature", fontsize=11)
        title = "Extraction" if task == "extraction" else "Summarization"
        ax.set_title(f"(a) {title}" if idx == 0 else f"(b) {title}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xlim(-0.05, 0.75)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Exact Match Rate (EMR)", fontsize=11)
    axes[1].legend(fontsize=9, loc="upper right", framealpha=0.9)

    fig.suptitle("Effect of Sampling Temperature on Reproducibility", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_temp_effect.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Generated: fig_temp_effect.pdf")


# ─── Figure 4: Three-level radar chart ──────────────────────────────────────

def generate_radar_chart(metrics):
    categories = ["EMR\n(Extraction)", "EMR\n(Summarization)", "1-NED\n(Extraction)",
                  "ROUGE-L\n(Extraction)", "BERTScore\n(Extraction)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in MODEL_ORDER:
        values = []
        # EMR extraction
        m = lookup_greedy(metrics, model, "extraction")
        values.append(m["emr_mean"] if m and m["emr_mean"] is not None else 0)
        # EMR summarization
        m_s = lookup_greedy(metrics, model, "summarization")
        values.append(m_s["emr_mean"] if m_s and m_s["emr_mean"] is not None else 0)
        # 1-NED extraction
        ned = m["ned_mean"] if m and m["ned_mean"] is not None else 1
        values.append(1.0 - ned)
        # ROUGE-L extraction
        values.append(m["rouge_l_mean"] if m and m["rouge_l_mean"] is not None else 0)
        # BERTScore extraction
        values.append(m["bertscore_f1_mean"] if m and m["bertscore_f1_mean"] is not None else 0)

        values += values[:1]
        linestyle = "-" if model in ["llama3_8b", "mistral_7b", "gemma2_9b"] else "--"
        ax.plot(angles, values, linewidth=2, linestyle=linestyle,
                color=MODEL_COLORS[model], label=MODEL_NAMES_SHORT[model])
        ax.fill(angles, values, alpha=0.05, color=MODEL_COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Three-Level Reproducibility Profile\n(Greedy Decoding)", fontsize=12, pad=20)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_three_level_radar.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Generated: fig_three_level_radar.pdf")


# ─── Figure 5: Multi-turn vs single-turn ────────────────────────────────────

def generate_multiturn_comparison(metrics):
    fig, ax = plt.subplots(figsize=(10, 5))

    local_models = ["gemma2_9b", "mistral_7b", "llama3_8b"]
    scenarios = [
        ("extraction", "Single-turn\nExtraction"),
        ("summarization", "Single-turn\nSummarization"),
        ("multiturn_refinement", "Multi-turn\nRefinement"),
        ("rag_extraction", "RAG\nExtraction"),
    ]

    x = np.arange(len(scenarios))
    width = 0.25
    offsets = [-width, 0, width]

    for i, model in enumerate(local_models):
        emrs = []
        stds = []
        for task_id, _ in scenarios:
            m = lookup(metrics, model, task_id, "C1_fixed_seed")
            if m:
                emrs.append(m["emr_mean"] if m["emr_mean"] is not None else 0)
                stds.append(m.get("emr_std", 0) or 0)
            else:
                emrs.append(0)
                stds.append(0)

        bars = ax.bar(x + offsets[i], emrs, width, yerr=stds,
                      label=MODEL_NAMES_SHORT[model], color=MODEL_COLORS[model],
                      alpha=0.85, edgecolor="white", capsize=3)

        for bar, val in zip(bars, emrs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Exact Match Rate (EMR)", fontsize=11)
    ax.set_title("Reproducibility Across Interaction Regimes (C1, t=0)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in scenarios], fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_multiturn_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Generated: fig_multiturn_comparison.pdf")


# ─── Figure 6: NED comparison ───────────────────────────────────────────────

def generate_ned_comparison(metrics):
    fig, ax = plt.subplots(figsize=(8, 5))

    tasks = ["extraction", "summarization"]
    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    for i, task in enumerate(tasks):
        neds = []
        for model in MODEL_ORDER:
            m = lookup_greedy(metrics, model, task)
            if m and m["ned_mean"] is not None:
                neds.append(m["ned_mean"])
            else:
                neds.append(0)

        color = "#1976D2" if task == "extraction" else "#FF8F00"
        bars = ax.bar(x + (i - 0.5) * width, neds, width,
                      label="Extraction" if task == "extraction" else "Summarization",
                      color=color, alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, neds):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
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
    print("  Generated: fig_ned_comparison.pdf")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_data()
    print(f"Loaded {len(metrics)} metric entries")
    print(f"Generating figures in {FIGURES_DIR}/...\n")

    generate_emr_heatmap(metrics)
    generate_api_vs_local(metrics)
    generate_temp_effect(metrics)
    generate_radar_chart(metrics)
    generate_multiturn_comparison(metrics)
    generate_ned_comparison(metrics)

    print(f"\nAll 6 figures generated in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
