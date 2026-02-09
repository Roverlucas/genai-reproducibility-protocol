#!/usr/bin/env python3
"""Generate publication figures for JAIR paper (v2: LLaMA + GPT-4)."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

ANALYSIS_DIR = Path(__file__).parent
FIGURES_DIR = ANALYSIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

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


def load_analysis():
    with open(ANALYSIS_DIR / "full_analysis.json") as f:
        return json.load(f)


def load_all_runs():
    with open(Path(__file__).parent.parent / "outputs" / "all_runs.json") as f:
        return json.load(f)


def fig_temperature_both_models(analysis):
    """Line plot: temperature effect on ROUGE-L for both models."""
    agg = analysis["variability_aggregated"]

    temps = [0.0, 0.3, 0.7]
    temp_conds = ["C3_t0.0", "C3_t0.3", "C3_t0.7"]

    configs = [
        ("llama3_8b", "summarization", "LLaMA 3 - Sum.", "#2196F3", "o", "-"),
        ("llama3_8b", "extraction", "LLaMA 3 - Ext.", "#2196F3", "s", "--"),
        ("gpt4", "summarization", "GPT-4 - Sum.", "#F44336", "o", "-"),
        ("gpt4", "extraction", "GPT-4 - Ext.", "#F44336", "s", "--"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ROUGE-L
    for model, task, label, color, marker, ls in configs:
        values = []
        for cond in temp_conds:
            key = f"{model}_{task}_{cond}"
            if key in agg:
                values.append(agg[key]["rouge_l"]["mean"])
            else:
                values.append(None)
        ax1.plot(temps, values, marker=marker, color=color, label=label,
                linewidth=2, markersize=7, linestyle=ls)

    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Mean ROUGE-L F1")
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)
    ax1.set_title("(a) ROUGE-L vs Temperature")

    # EMR
    for model, task, label, color, marker, ls in configs:
        values = []
        for cond in temp_conds:
            key = f"{model}_{task}_{cond}"
            if key in agg:
                values.append(agg[key]["exact_match_rate"]["mean"])
            else:
                values.append(None)
        ax2.plot(temps, values, marker=marker, color=color, label=label,
                linewidth=2, markersize=7, linestyle=ls)

    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Mean Exact Match Rate")
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.set_title("(b) EMR vs Temperature")

    plt.tight_layout()
    path = FIGURES_DIR / "fig_temperature_both_models.pdf"
    plt.savefig(path)
    plt.savefig(FIGURES_DIR / "fig_temperature_both_models.png")
    plt.close()
    print(f"  [OK] {path}")


def fig_model_comparison_bar(analysis):
    """Grouped bar chart: LLaMA vs GPT-4 under greedy decoding (t=0)."""
    agg = analysis["variability_aggregated"]

    # Compare C2 condition (both models have it)
    tasks = ["summarization", "extraction"]
    task_labels = ["Summarization", "Extraction"]
    metrics_keys = [
        ("exact_match_rate", "EMR"),
        ("edit_distance_normalized", "NED"),
        ("rouge_l", "ROUGE-L"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.arange(len(tasks))
    width = 0.35

    for ax_idx, (metric_key, ylabel) in enumerate(metrics_keys):
        ax = axes[ax_idx]

        llama_vals = []
        gpt4_vals = []
        for task in tasks:
            lk = f"llama3_8b_{task}_C2"
            gk = f"gpt4_{task}_C2"
            llama_vals.append(agg[lk][metric_key]["mean"] if lk in agg else 0)
            gpt4_vals.append(agg[gk][metric_key]["mean"] if gk in agg else 0)

        ax.bar(x - width/2, llama_vals, width, label="LLaMA 3 8B (local)",
               color="#2196F3", alpha=0.85, edgecolor="white")
        ax.bar(x + width/2, gpt4_vals, width, label="GPT-4 (API)",
               color="#F44336", alpha=0.85, edgecolor="white")

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels)
        ax.legend(loc="best", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        if metric_key != "edit_distance_normalized":
            ax.set_ylim(0, 1.15)

    fig.suptitle("Reproducibility under Greedy Decoding (t=0, same params)", fontsize=12, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "fig_model_comparison.pdf"
    plt.savefig(path)
    plt.savefig(FIGURES_DIR / "fig_model_comparison.png")
    plt.close()
    print(f"  [OK] {path}")


def fig_comprehensive_heatmap(analysis):
    """Heatmap: EMR for all (model, task, condition) combinations."""
    agg = analysis["variability_aggregated"]

    # Rows: conditions, Cols: model+task combos
    conditions = ["C1", "C2", "C3_t0.0", "C3_t0.3", "C3_t0.7"]
    cond_labels = ["C1 (fixed seed)", "C2 (same params)", "C3 (t=0.0)", "C3 (t=0.3)", "C3 (t=0.7)"]

    combos = [
        ("llama3_8b", "summarization", "LLaMA\nSum."),
        ("llama3_8b", "extraction", "LLaMA\nExt."),
        ("gpt4", "summarization", "GPT-4\nSum."),
        ("gpt4", "extraction", "GPT-4\nExt."),
    ]

    matrix = np.full((len(conditions), len(combos)), np.nan)
    for i, cond in enumerate(conditions):
        for j, (model, task, _) in enumerate(combos):
            key = f"{model}_{task}_{cond}"
            if key in agg:
                matrix[i, j] = agg[key]["exact_match_rate"]["mean"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([c[2] for c in combos], fontsize=9)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(cond_labels, fontsize=9)

    for i in range(len(conditions)):
        for j in range(len(combos)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray")
            else:
                color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=10, color=color, fontweight="bold")

    # Add vertical line separating models
    ax.axvline(x=1.5, color="black", linewidth=2)

    ax.set_title("Exact Match Rate: LLaMA 3 (local) vs GPT-4 (API)", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("EMR", fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_heatmap_both_models.pdf"
    plt.savefig(path)
    plt.savefig(FIGURES_DIR / "fig_heatmap_both_models.png")
    plt.close()
    print(f"  [OK] {path}")


def fig_overhead_both_models(all_runs):
    """Box plot: overhead comparison between models."""
    llama_overheads = [r["logging_overhead_ms"] for r in all_runs if "llama" in r.get("model_name", "").lower()]
    gpt4_overheads = [r["logging_overhead_ms"] for r in all_runs if "gpt" in r.get("model_name", "").lower()]

    llama_ratios = [r["logging_overhead_ms"] / r["execution_duration_ms"] * 100
                    for r in all_runs if "llama" in r.get("model_name", "").lower() and r["execution_duration_ms"] > 0]
    gpt4_ratios = [r["logging_overhead_ms"] / r["execution_duration_ms"] * 100
                   for r in all_runs if "gpt" in r.get("model_name", "").lower() and r["execution_duration_ms"] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    bp1 = ax1.boxplot([llama_overheads, gpt4_overheads], tick_labels=["LLaMA 3 8B", "GPT-4"],
                       patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    bp1["boxes"][0].set_facecolor("#2196F3")
    bp1["boxes"][0].set_alpha(0.7)
    bp1["boxes"][1].set_facecolor("#F44336")
    bp1["boxes"][1].set_alpha(0.7)
    ax1.set_ylabel("Logging Overhead (ms)")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    bp2 = ax2.boxplot([llama_ratios, gpt4_ratios], tick_labels=["LLaMA 3 8B", "GPT-4"],
                       patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    bp2["boxes"][0].set_facecolor("#2196F3")
    bp2["boxes"][0].set_alpha(0.7)
    bp2["boxes"][1].set_facecolor("#F44336")
    bp2["boxes"][1].set_alpha(0.7)
    ax2.set_ylabel("Overhead / Inference Time (%)")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_overhead_both_models.pdf"
    plt.savefig(path)
    plt.savefig(FIGURES_DIR / "fig_overhead_both_models.png")
    plt.close()
    print(f"  [OK] {path}")


def main():
    print("Generating figures v2 (LLaMA + GPT-4)...")
    analysis = load_analysis()
    all_runs = load_all_runs()

    fig_temperature_both_models(analysis)
    fig_model_comparison_bar(analysis)
    fig_comprehensive_heatmap(analysis)
    fig_overhead_both_models(all_runs)

    print(f"\n[OK] All v2 figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
