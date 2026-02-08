#!/usr/bin/env python3
"""Generate publication-quality figures for the JAIR paper.

Creates:
- Figure 1: Variability metrics comparison across conditions (grouped bar chart)
- Figure 2: Protocol overhead distribution (box plot)
- Figure 3: Temperature effect on ROUGE-L (line plot)
- Figure 4: Output length distribution per condition (violin/box)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

ANALYSIS_DIR = Path(__file__).parent
FIGURES_DIR = ANALYSIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

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

COLORS = {
    "C1": "#2196F3",
    "C2": "#4CAF50",
    "C3_t0.0": "#FF9800",
    "C3_t0.3": "#F44336",
    "C3_t0.7": "#9C27B0",
}


def load_analysis():
    with open(ANALYSIS_DIR / "full_analysis.json") as f:
        return json.load(f)


def load_all_runs():
    with open(Path(__file__).parent.parent / "outputs" / "all_runs.json") as f:
        return json.load(f)


def fig1_variability_comparison(analysis):
    """Grouped bar chart: EMR, NED, ROUGE-L per condition and task."""
    agg = analysis["variability_aggregated"]

    conditions = ["C1", "C2", "C3_t0.0", "C3_t0.3", "C3_t0.7"]
    cond_labels = ["C1\n(fixed seed)", "C2\n(var. seeds)", "C3\n(t=0.0)", "C3\n(t=0.3)", "C3\n(t=0.7)"]
    tasks = ["summarization", "extraction"]
    task_labels = ["Summarization", "Extraction"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    metrics = [
        ("exact_match_rate", "Exact Match Rate (EMR)", "mean"),
        ("edit_distance_normalized", "Normalized Edit Distance (NED)", "mean"),
        ("rouge_l", "ROUGE-L F1", "mean"),
    ]

    x = np.arange(len(conditions))
    width = 0.35

    for ax_idx, (metric_key, ylabel, stat_key) in enumerate(metrics):
        ax = axes[ax_idx]
        for t_idx, task in enumerate(tasks):
            values = []
            for cond in conditions:
                key = f"{task}_{cond}"
                if key in agg:
                    values.append(agg[key][metric_key][stat_key])
                else:
                    values.append(0)

            offset = -width / 2 + t_idx * width
            bars = ax.bar(x + offset, values, width, label=task_labels[t_idx],
                         color=["#2196F3", "#FF9800"][t_idx], alpha=0.85,
                         edgecolor="white", linewidth=0.5)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=8)
        if ax_idx == 1:
            ax.set_xlabel("Experimental Condition")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    # Adjust y-axis
    axes[0].set_ylim(0, 1.1)
    axes[2].set_ylim(0, 1.1)

    plt.tight_layout()
    path = FIGURES_DIR / "fig1_variability_comparison.pdf"
    plt.savefig(path)
    plt.savefig(FIGURES_DIR / "fig1_variability_comparison.png")
    plt.close()
    print(f"  [OK] {path}")


def fig2_overhead_distribution(all_runs):
    """Box plot of logging overhead and overhead ratio."""
    overheads = [r["logging_overhead_ms"] for r in all_runs]
    durations = [r["execution_duration_ms"] for r in all_runs]
    ratios = [r["logging_overhead_ms"] / r["execution_duration_ms"] * 100
              for r in all_runs if r["execution_duration_ms"] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Overhead in ms
    bp1 = ax1.boxplot(overheads, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#2196F3", alpha=0.7),
                       medianprops=dict(color="black", linewidth=1.5))
    ax1.set_ylabel("Logging Overhead (ms)")
    ax1.set_xticklabels(["All 190 runs"])
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    mean_oh = np.mean(overheads)
    ax1.axhline(y=mean_oh, color="red", linestyle="--", alpha=0.7, linewidth=0.8)
    ax1.annotate(f"mean={mean_oh:.1f}ms", xy=(1.15, mean_oh),
                fontsize=8, color="red", va="center")

    # Overhead ratio %
    bp2 = ax2.boxplot(ratios, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#FF9800", alpha=0.7),
                       medianprops=dict(color="black", linewidth=1.5))
    ax2.set_ylabel("Overhead / Inference Time (%)")
    ax2.set_xticklabels(["All 190 runs"])
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    mean_r = np.mean(ratios)
    ax2.axhline(y=mean_r, color="red", linestyle="--", alpha=0.7, linewidth=0.8)
    ax2.annotate(f"mean={mean_r:.3f}%", xy=(1.15, mean_r),
                fontsize=8, color="red", va="center")

    plt.tight_layout()
    path = FIGURES_DIR / "fig2_overhead_distribution.pdf"
    plt.savefig(path)
    plt.savefig(FIGURES_DIR / "fig2_overhead_distribution.png")
    plt.close()
    print(f"  [OK] {path}")


def fig3_temperature_effect(analysis):
    """Line plot: temperature effect on ROUGE-L and EMR."""
    agg = analysis["variability_aggregated"]

    temps = [0.0, 0.3, 0.7]
    temp_conds = ["C3_t0.0", "C3_t0.3", "C3_t0.7"]
    tasks = ["summarization", "extraction"]
    task_labels = ["Summarization", "Extraction"]
    task_colors = ["#2196F3", "#FF9800"]
    task_markers = ["o", "s"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # ROUGE-L vs temperature
    for t_idx, task in enumerate(tasks):
        values = []
        for cond in temp_conds:
            key = f"{task}_{cond}"
            values.append(agg[key]["rouge_l"]["mean"])
        ax1.plot(temps, values, marker=task_markers[t_idx], color=task_colors[t_idx],
                label=task_labels[t_idx], linewidth=2, markersize=7)

    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Mean ROUGE-L F1")
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # EMR vs temperature
    for t_idx, task in enumerate(tasks):
        values = []
        for cond in temp_conds:
            key = f"{task}_{cond}"
            values.append(agg[key]["exact_match_rate"]["mean"])
        ax2.plot(temps, values, marker=task_markers[t_idx], color=task_colors[t_idx],
                label=task_labels[t_idx], linewidth=2, markersize=7)

    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Mean Exact Match Rate")
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    path = FIGURES_DIR / "fig3_temperature_effect.pdf"
    plt.savefig(path)
    plt.savefig(FIGURES_DIR / "fig3_temperature_effect.png")
    plt.close()
    print(f"  [OK] {path}")


def fig4_output_length_distribution(all_runs):
    """Box plot: output length distribution per condition and task."""
    # Group outputs by (task, condition)
    groups = defaultdict(list)
    for run in all_runs:
        task = run["task_id"]
        run_id = run["run_id"]
        if "C1_fixed_seed" in run_id:
            cond = "C1"
        elif "C2_var_seed" in run_id:
            cond = "C2"
        elif "C3_temp0.0" in run_id:
            cond = "C3_t0.0"
        elif "C3_temp0.3" in run_id:
            cond = "C3_t0.3"
        elif "C3_temp0.7" in run_id:
            cond = "C3_t0.7"
        else:
            continue
        groups[(task, cond)].append(len(run["output_text"].split()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    conditions = ["C1", "C2", "C3_t0.0", "C3_t0.3", "C3_t0.7"]
    cond_labels = ["C1", "C2", "C3\nt=0.0", "C3\nt=0.3", "C3\nt=0.7"]

    for ax, task, title in [(ax1, "summarization", "Summarization"),
                             (ax2, "extraction", "Extraction")]:
        data = []
        for cond in conditions:
            data.append(groups.get((task, cond), []))

        bp = ax.boxplot(data, labels=cond_labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, cond in zip(bp["boxes"], conditions):
            patch.set_facecolor(COLORS[cond])
            patch.set_alpha(0.7)

        ax.set_title(title)
        ax.set_ylabel("Output Length (words)")
        ax.set_xlabel("Condition")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = FIGURES_DIR / "fig4_output_length.pdf"
    plt.savefig(path)
    plt.savefig(FIGURES_DIR / "fig4_output_length.png")
    plt.close()
    print(f"  [OK] {path}")


def fig5_heatmap_per_abstract(analysis):
    """Heatmap: EMR per abstract and condition for summarization."""
    var = analysis["variability_per_abstract"]

    conditions = ["C1", "C2", "C3_t0.0", "C3_t0.3", "C3_t0.7"]
    abstracts = ["abs_001", "abs_002", "abs_003", "abs_004", "abs_005"]
    abstract_labels = ["Vaswani\n(Transformer)", "Devlin\n(BERT)", "Brown\n(GPT-3)",
                       "Raffel\n(T5)", "Wei\n(CoT)"]
    cond_labels = ["C1 (fixed)", "C2 (var. seed)", "C3 (t=0.0)", "C3 (t=0.3)", "C3 (t=0.7)"]

    # Build matrix for summarization
    matrix = np.zeros((len(conditions), len(abstracts)))
    for i, cond in enumerate(conditions):
        for j, abs_id in enumerate(abstracts):
            key = f"summarization_{cond}_{abs_id}"
            if key in var:
                matrix[i, j] = var[key]["exact_match_rate"]

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(abstracts)))
    ax.set_xticklabels(abstract_labels, fontsize=8)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(cond_labels, fontsize=9)

    # Annotate cells
    for i in range(len(conditions)):
        for j in range(len(abstracts)):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   fontsize=9, color=color, fontweight="bold")

    ax.set_title("Exact Match Rate — Summarization Task", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("EMR", fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "fig5_heatmap_emr.pdf"
    plt.savefig(path)
    plt.savefig(FIGURES_DIR / "fig5_heatmap_emr.png")
    plt.close()
    print(f"  [OK] {path}")


def fig6_prov_diagram():
    """Generate a conceptual PROV diagram description (for manual LaTeX/TikZ)."""
    # This creates a simple placeholder diagram showing PROV entity relationships
    fig, ax = plt.subplots(figsize=(10, 5))

    # Entity boxes
    entities = {
        "Prompt": (1, 4),
        "Input Text": (1, 2),
        "Model\nVersion": (3, 4),
        "Inference\nParams": (3, 2),
        "Run\nActivity": (5, 3),
        "Output\nText": (7, 3),
        "PROV\nDocument": (9, 3),
    }

    agents = {
        "Researcher": (5, 5),
        "System\nExecutor": (5, 1),
    }

    # Draw entities
    for name, (x, y) in entities.items():
        rect = plt.Rectangle((x - 0.6, y - 0.4), 1.2, 0.8,
                             facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, name, ha="center", va="center", fontsize=8, fontweight="bold")

    # Draw agents
    for name, (x, y) in agents.items():
        rect = plt.Rectangle((x - 0.6, y - 0.4), 1.2, 0.8,
                             facecolor="#FFF3E0", edgecolor="#E65100", linewidth=1.5,
                             linestyle="--")
        ax.add_patch(rect)
        ax.text(x, y, name, ha="center", va="center", fontsize=8, fontweight="bold",
               color="#E65100")

    # Draw arrows (used, wasGeneratedBy, wasAssociatedWith)
    arrows = [
        ((1, 4), (5, 3), "used"),
        ((1, 2), (5, 3), "used"),
        ((3, 4), (5, 3), "used"),
        ((3, 2), (5, 3), "used"),
        ((5, 3), (7, 3), "wasGeneratedBy"),
        ((7, 3), (9, 3), "wasDerivedFrom"),
        ((5, 5), (5, 3.5), "wasAssoc.With"),
        ((5, 1), (5, 2.5), "wasAssoc.With"),
    ]

    for (x1, y1), (x2, y2), label in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.15, label, ha="center", va="bottom", fontsize=6,
               color="#666", fontstyle="italic")

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("W3C PROV Data Model for GenAI Experiment Runs", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = FIGURES_DIR / "fig6_prov_diagram.pdf"
    plt.savefig(path)
    plt.savefig(FIGURES_DIR / "fig6_prov_diagram.png")
    plt.close()
    print(f"  [OK] {path}")


def main():
    print("Generating figures for JAIR paper...")
    analysis = load_analysis()
    all_runs = load_all_runs()

    fig1_variability_comparison(analysis)
    fig2_overhead_distribution(all_runs)
    fig3_temperature_effect(analysis)
    fig4_output_length_distribution(all_runs)
    fig5_heatmap_per_abstract(analysis)
    fig6_prov_diagram()

    print(f"\n[OK] All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
