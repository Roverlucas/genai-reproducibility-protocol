#!/usr/bin/env python3
"""Generate visual abstract and conceptual pipeline figures.

Produces two publication-quality PDF figures:
  1. fig_visual_abstract.pdf   — single-page visual abstract for social media / talks
  2. fig_conceptual_pipeline.pdf — horizontal protocol pipeline diagram

Usage:
    python analysis/generate_visual_abstract.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTICLE_FIGURES = Path(__file__).resolve().parent.parent / "article" / "figures"
ARTICLE_FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
GREEN_DARK  = "#1b7837"
GREEN_MID   = "#4dac26"
GREEN_LIGHT = "#7fbc41"
RED_DARK    = "#c51b7d"
RED_MID     = "#e66101"
ORANGE      = "#f1a340"
ORANGE_LIGHT = "#fdb863"
GREY_BG     = "#f7f7f7"
GREY_LINE   = "#bdbdbd"
GREY_TEXT   = "#525252"
DARK_TEXT   = "#1a1a1a"
WHITE       = "#ffffff"
BLUE_ACCENT = "#2166ac"
BLUE_LIGHT  = "#d1e5f0"

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                        "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "text.color": DARK_TEXT,
    "axes.edgecolor": GREY_LINE,
    "axes.labelcolor": DARK_TEXT,
    "xtick.color": GREY_TEXT,
    "ytick.color": GREY_TEXT,
})


# ===================================================================
# FIGURE 1 — Visual Abstract
# ===================================================================
def draw_visual_abstract():
    fig = plt.figure(figsize=(16, 9), facecolor=WHITE)

    # ---- overall title ----
    fig.text(0.50, 0.96,
             "Hidden Non-Determinism in LLM APIs",
             ha="center", va="top", fontsize=24, fontweight="bold",
             color=DARK_TEXT)
    fig.text(0.50, 0.92,
             "A Lightweight Provenance Protocol for Reproducible Generative AI Research",
             ha="center", va="top", fontsize=13, color=GREY_TEXT, style="italic")

    # ---- bottom URL bar ----
    fig.text(0.50, 0.015,
             "github.com/Roverlucas/genai-reproducibility-protocol",
             ha="center", va="bottom", fontsize=11, color=BLUE_ACCENT,
             fontweight="bold", family="monospace")

    # ------------------------------------------------------------------
    # LEFT PANEL — Protocol diagram (axes coordinates 0.03–0.28)
    # ------------------------------------------------------------------
    ax_left = fig.add_axes([0.03, 0.08, 0.25, 0.78], frameon=False)
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 10)
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    ax_left.text(5, 9.8, "Protocol Pipeline", ha="center", va="top",
                 fontsize=14, fontweight="bold", color=DARK_TEXT)

    # Boxes: top-to-bottom flow
    boxes = [
        (5, 8.6, "Prompt\nCard",      BLUE_LIGHT,  BLUE_ACCENT),
        (5, 6.8, "Model\nExecution",  "#d9f0d3",   GREEN_DARK),
        (5, 5.0, "Run\nCard",         "#fee0b6",   RED_MID),
        (5, 3.2, "PROV\nGraph",       "#e0d4f5",   "#7b3294"),
    ]

    annotations_right = [
        (8.3, 8.6, "task, abstract,\nparams, seed",        GREY_TEXT),
        (8.3, 6.8, "temp=0, top_p=1,\nweights hash",      GREY_TEXT),
        (8.3, 5.0, "output hash,\ntimestamp, tokens",      GREY_TEXT),
        (8.3, 3.2, "wasGeneratedBy,\nwasDerivedFrom",      GREY_TEXT),
    ]

    for cx, cy, label, bg, ec in boxes:
        box = FancyBboxPatch((cx - 2.2, cy - 0.7), 4.4, 1.4,
                             boxstyle="round,pad=0.15",
                             facecolor=bg, edgecolor=ec, linewidth=1.8,
                             transform=ax_left.transData, zorder=3)
        ax_left.add_patch(box)
        ax_left.text(cx, cy, label, ha="center", va="center",
                     fontsize=10, fontweight="bold", color=ec, zorder=4)

    for tx, ty, label, col in annotations_right:
        ax_left.text(tx, ty, label, ha="left", va="center",
                     fontsize=7.5, color=col, style="italic")

    # Arrows between boxes
    arrow_kw = dict(arrowstyle="-|>", color=GREY_TEXT, lw=1.5,
                    mutation_scale=14, connectionstyle="arc3,rad=0")
    for i in range(len(boxes) - 1):
        y_start = boxes[i][1] - 0.75
        y_end   = boxes[i + 1][1] + 0.75
        ax_left.annotate("", xy=(5, y_end), xytext=(5, y_start),
                         arrowprops=arrow_kw, zorder=2)

    # Verification badge at bottom
    ax_left.text(5, 1.6, "Reproducibility\nVerification",
                 ha="center", va="center", fontsize=9,
                 fontweight="bold", color=GREEN_DARK,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#d9f0d3",
                           edgecolor=GREEN_DARK, linewidth=1.5))
    ax_left.annotate("", xy=(5, 2.15), xytext=(5, boxes[-1][1] - 0.75),
                     arrowprops=arrow_kw, zorder=2)

    # ------------------------------------------------------------------
    # CENTER PANEL — Bar chart (axes 0.33–0.68)
    # Leave room at bottom for group labels (lower y-origin)
    # ------------------------------------------------------------------
    ax_bar = fig.add_axes([0.33, 0.22, 0.35, 0.62])

    models = [
        "Gemma 2\n9B", "LLaMA 3\n8B", "Mistral\n7B",
        "DeepSeek\nR1 8B", "GPT-4", "Claude\nSonnet 4.5", "Perplexity\nSonar"
    ]
    emr_values = [1.000, 0.987, 0.960, 0.800, 0.443, 0.190, 0.100]
    colors = [GREEN_DARK, GREEN_MID, GREEN_LIGHT,
              ORANGE_LIGHT, ORANGE, RED_MID, RED_DARK]

    bars = ax_bar.bar(range(len(models)), emr_values, color=colors,
                      edgecolor=WHITE, linewidth=0.8, width=0.72, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, emr_values):
        ypos = bar.get_height() + 0.015
        ax_bar.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=DARK_TEXT)

    ax_bar.set_xticks(range(len(models)))
    ax_bar.set_xticklabels(models, fontsize=9)
    ax_bar.set_ylabel("Exact Match Rate (EMR)", fontsize=12, fontweight="bold")
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_xlim(-0.6, len(models) + 0.4)
    ax_bar.set_title("Greedy-Decoding Reproducibility (Extraction Task)",
                     fontsize=13, fontweight="bold", pad=10)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.yaxis.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax_bar.set_axisbelow(True)

    # --- Group brackets below x-axis (using data coords + axes fraction) ---
    # We use ax_bar.get_xaxis_transform() which maps x=data, y=axes fraction
    # y=0 is the x-axis line; negative values go below it.
    xax = ax_bar.get_xaxis_transform()

    # LOCAL bracket (bars 0,1,2)
    bracket_y = -0.14
    label_y   = -0.20
    avg_y     = -0.27
    # horizontal line
    ax_bar.plot([-0.35, 2.35], [bracket_y, bracket_y],
                color=GREEN_DARK, lw=2.5, transform=xax, clip_on=False)
    # end ticks
    for xp in [-0.35, 2.35]:
        ax_bar.plot([xp, xp], [bracket_y, bracket_y + 0.03],
                    color=GREEN_DARK, lw=2.5, transform=xax, clip_on=False)
    ax_bar.text(1.0, label_y, "LOCAL", ha="center", va="top",
                transform=xax, fontsize=11, fontweight="bold",
                color=GREEN_DARK, clip_on=False)
    ax_bar.text(1.0, avg_y, "avg EMR = 0.982", ha="center", va="top",
                transform=xax, fontsize=9, color=GREEN_DARK, clip_on=False)

    # API bracket (bars 3,4,5,6)
    ax_bar.plot([2.65, 6.35], [bracket_y, bracket_y],
                color=RED_MID, lw=2.5, transform=xax, clip_on=False)
    for xp in [2.65, 6.35]:
        ax_bar.plot([xp, xp], [bracket_y, bracket_y + 0.03],
                    color=RED_MID, lw=2.5, transform=xax, clip_on=False)
    ax_bar.text(4.5, label_y, "API", ha="center", va="top",
                transform=xax, fontsize=11, fontweight="bold",
                color=RED_MID, clip_on=False)
    ax_bar.text(4.5, avg_y, "avg EMR = 0.383", ha="center", va="top",
                transform=xax, fontsize=9, color=RED_MID, clip_on=False)

    # "3x gap" annotation — vertical double arrow to the right of the chart
    mid_local_y = np.mean([1.000, 0.987, 0.960])  # ~0.982
    mid_api_y   = np.mean([0.800, 0.443, 0.190, 0.100])  # ~0.383
    gap_x = 7.0  # to the right of the last bar, within extended xlim
    ax_bar.annotate("",
                    xy=(gap_x, mid_api_y), xytext=(gap_x, mid_local_y),
                    arrowprops=dict(arrowstyle="<->", color=DARK_TEXT,
                                    lw=2.5, shrinkA=0, shrinkB=0),
                    annotation_clip=False)
    ax_bar.text(gap_x + 0.15, (mid_local_y + mid_api_y) / 2,
                "3\u00d7\ngap", ha="left", va="center",
                fontsize=13, fontweight="bold", color=DARK_TEXT,
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=WHITE,
                          edgecolor=GREY_LINE, alpha=0.95))

    # ------------------------------------------------------------------
    # RIGHT PANEL — Key statistics (axes 0.73–0.97)
    # ------------------------------------------------------------------
    ax_right = fig.add_axes([0.73, 0.08, 0.25, 0.78], frameon=False)
    ax_right.set_xlim(0, 10)
    ax_right.set_ylim(0, 10)
    ax_right.set_xticks([])
    ax_right.set_yticks([])

    ax_right.text(5, 9.8, "Key Results", ha="center", va="top",
                  fontsize=14, fontweight="bold", color=DARK_TEXT)

    stats = [
        ("3,804",  "total runs"),
        ("7",      "LLM models"),
        ("4",      "tasks"),
        ("4",      "API providers"),
        ("<1%",    "overhead"),
    ]

    y_positions = [8.3, 7.0, 5.7, 4.4, 3.1]
    for (big, small), yp in zip(stats, y_positions):
        # Big number in a rounded box
        ax_right.text(5, yp + 0.15, big, ha="center", va="center",
                      fontsize=26, fontweight="bold", color=BLUE_ACCENT)
        ax_right.text(5, yp - 0.55, small, ha="center", va="center",
                      fontsize=11, color=GREY_TEXT)
        # Subtle divider
        if yp != y_positions[-1]:
            ax_right.plot([1.5, 8.5], [yp - 0.9, yp - 0.9],
                          color=GREY_LINE, lw=0.5, alpha=0.5)

    # Takeaway box at bottom
    takeaway_text = ("Local models reproduce\n"
                     "near-perfectly; API outputs\n"
                     "vary across identical calls.")
    ax_right.text(5, 1.3, takeaway_text, ha="center", va="center",
                  fontsize=10, color=DARK_TEXT, linespacing=1.4,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff7bc",
                            edgecolor="#d95f0e", linewidth=1.5))

    # ---- Save ----
    out = ARTICLE_FIGURES / "fig_visual_abstract.pdf"
    fig.savefig(out, facecolor=WHITE)
    plt.close(fig)
    print(f"Saved: {out}")


# ===================================================================
# FIGURE 2 — Conceptual Pipeline Diagram
# ===================================================================
def _rounded_box(ax, cx, cy, w, h, label, sublabel, fc, ec, fontsize=11):
    """Draw a rounded rectangle centred at (cx, cy) with label + sublabel."""
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                         boxstyle="round,pad=0.12",
                         facecolor=fc, edgecolor=ec, linewidth=2,
                         transform=ax.transData, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(cx, cy + 0.18, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=ec, zorder=4)
        ax.text(cx, cy - 0.28, sublabel, ha="center", va="center",
                fontsize=7.5, color=GREY_TEXT, style="italic", zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=ec, zorder=4)


def draw_conceptual_pipeline():
    fig = plt.figure(figsize=(16, 6), facecolor=WHITE)
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.82], frameon=False)
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-1.5, 4.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    fig.text(0.50, 0.95,
             "Reproducibility Protocol: Conceptual Pipeline",
             ha="center", va="top", fontsize=18, fontweight="bold",
             color=DARK_TEXT)

    # ---- Main pipeline boxes ----
    # x positions chosen for even spacing
    bw, bh = 2.2, 1.3  # box width, height
    row_y = 2.0

    main_boxes = [
        (1.2,  row_y, "Input",         "abstract text,\ntask type",       BLUE_LIGHT,  BLUE_ACCENT),
        (4.5,  row_y, "Prompt\nCard",  "seed, params,\nmodel ID",        "#d1e5f0",   BLUE_ACCENT),
        (7.8,  row_y, "Model",         "temp=0, top_p=1,\nweights hash", "#d9f0d3",   GREEN_DARK),
        (11.1, row_y, "Output",        "raw response,\ntoken count",     "#fee0b6",   RED_MID),
        (14.8, row_y, "Run\nCard",     "output hash,\ntimestamp, EMR",   "#fde0ef",   RED_DARK),
    ]

    for cx, cy, lab, sub, fc, ec in main_boxes:
        _rounded_box(ax, cx, cy, bw, bh, lab, sub, fc, ec)

    # ---- Arrows between main boxes ----
    arrow_kw = dict(arrowstyle="-|>", color=GREY_TEXT, lw=2,
                    mutation_scale=16, shrinkA=6, shrinkB=6)
    arrow_pairs = [(1.2, 4.5), (4.5, 7.8), (7.8, 11.1), (11.1, 14.8)]
    for x1, x2 in arrow_pairs:
        ax.annotate("", xy=(x2 - bw / 2, row_y),
                    xytext=(x1 + bw / 2, row_y),
                    arrowprops=arrow_kw, zorder=2)

    # ---- Protocol layer (below) ----
    proto_y = -0.2
    proto_boxes = [
        (4.5,  proto_y, "Hashing",   "SHA-256 of\nweights + output",  "#e0d4f5", "#7b3294"),
        (8.5,  proto_y, "Logging",   "JSON run record,\nauto-generated",  "#e0d4f5", "#7b3294"),
        (12.5, proto_y, "PROV\nGraph", "W3C PROV-DM,\nentity/activity",  "#e0d4f5", "#7b3294"),
    ]

    for cx, cy, lab, sub, fc, ec in proto_boxes:
        _rounded_box(ax, cx, cy, 2.4, 1.2, lab, sub, fc, ec, fontsize=10)

    # Arrows between protocol boxes
    proto_arrow_kw = dict(arrowstyle="-|>", color="#7b3294", lw=1.5,
                          mutation_scale=14, shrinkA=6, shrinkB=6)
    for x1, x2 in [(4.5, 8.5), (8.5, 12.5)]:
        ax.annotate("", xy=(x2 - 1.2, proto_y),
                    xytext=(x1 + 1.2, proto_y),
                    arrowprops=proto_arrow_kw, zorder=2)

    # ---- Vertical arrows connecting main row to protocol layer ----
    vert_kw = dict(arrowstyle="-|>", color="#7b3294", lw=1.2,
                   mutation_scale=12, shrinkA=4, shrinkB=4,
                   linestyle="dashed")
    # Model -> Hashing
    ax.annotate("", xy=(4.5, proto_y + 0.6),
                xytext=(7.8, row_y - bh / 2),
                arrowprops=vert_kw, zorder=2)
    # Output -> Logging
    ax.annotate("", xy=(8.5, proto_y + 0.6),
                xytext=(11.1, row_y - bh / 2),
                arrowprops=vert_kw, zorder=2)
    # PROV -> Run Card (upward)
    vert_up_kw = dict(arrowstyle="-|>", color="#7b3294", lw=1.2,
                      mutation_scale=12, shrinkA=4, shrinkB=4,
                      linestyle="dashed")
    ax.annotate("", xy=(14.8, row_y - bh / 2),
                xytext=(12.5, proto_y + 0.6),
                arrowprops=vert_up_kw, zorder=2)

    # ---- Protocol layer bracket ----
    bracket_y = proto_y - 1.0
    ax.plot([3.2, 13.8], [bracket_y, bracket_y],
            color="#7b3294", lw=2, clip_on=False)
    ax.plot([3.2, 3.2], [bracket_y, bracket_y + 0.15],
            color="#7b3294", lw=2, clip_on=False)
    ax.plot([13.8, 13.8], [bracket_y, bracket_y + 0.15],
            color="#7b3294", lw=2, clip_on=False)
    ax.text(8.5, bracket_y - 0.25,
            "Protocol Layer  (<1% runtime overhead)",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="#7b3294")

    # ---- Save ----
    out = ARTICLE_FIGURES / "fig_conceptual_pipeline.pdf"
    fig.savefig(out, facecolor=WHITE)
    plt.close(fig)
    print(f"Saved: {out}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    draw_visual_abstract()
    draw_conceptual_pipeline()
    print("Done — both figures generated.")
