#!/usr/bin/env python3
"""Generate LaTeX tables for the expanded reproducibility paper (v3: 5 models).

Reads expanded_metrics.json and cross_model_comparison.json to produce
publication-ready LaTeX tables for the JAIR paper.

Outputs (in analysis/tables/):
  - table_emr_greedy.tex       — EMR under greedy (C1/C2) for 5 models
  - table_three_level.tex      — L1/L2/L3 metrics under C1
  - table_temp_sweep.tex       — EMR across temperature settings
  - table_multiturn_rag.tex    — Multi-turn and RAG vs single-turn
  - table_overhead.tex         — Protocol overhead stats
  - table_api_vs_local.tex     — API vs local model comparison

Usage:
    python analysis/generate_tables_v3.py
"""

import json
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent
TABLES_DIR = ANALYSIS_DIR / "tables"

MODEL_NAMES = {
    "llama3_8b": "LLaMA 3 8B",
    "mistral_7b": "Mistral 7B",
    "gemma2_9b": "Gemma 2 9B",
    "gpt4": "GPT-4",
    "claude_sonnet": "Claude Sonnet 4.5",
}

MODEL_ORDER = ["gemma2_9b", "mistral_7b", "llama3_8b", "gpt4", "claude_sonnet"]

TASK_NAMES = {
    "extraction": "Extraction",
    "summarization": "Summarization",
    "multiturn_refinement": "Multi-turn",
    "rag_extraction": "RAG Extraction",
}


def load_data():
    with open(ANALYSIS_DIR / "expanded_metrics.json") as f:
        metrics = json.load(f)
    with open(ANALYSIS_DIR / "cross_model_comparison.json") as f:
        comparison = json.load(f)
    return metrics, comparison


def lookup(metrics, model, task, condition):
    for m in metrics:
        if m["model"] == model and m["task"] == task and m["condition"] == condition:
            return m
    return None


def lookup_greedy(metrics, model, task):
    """Find the best greedy-decoding entry for a model/task.

    Prefers the condition with the most abstracts (most representative).
    This handles GPT-4's incomplete C1 data (only 3 abstracts for
    summarization) by selecting C2 (30 abstracts) instead.
    """
    candidates = []
    for cond in ["C1_fixed_seed", "C2_var_seed", "C2_same_params"]:
        m = lookup(metrics, model, task, cond)
        if m is not None:
            candidates.append(m)
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.get("n_abstracts", 0))


def fmt(val, decimals=3):
    if val is None:
        return "---"
    return f"{val:.{decimals}f}"


def fmt_emr(val):
    if val is None:
        return "---"
    s = f"{val:.3f}"
    if val >= 0.95:
        return f"\\cellcolor{{green!20}}{s}"
    elif val >= 0.80:
        return f"\\cellcolor{{yellow!20}}{s}"
    elif val >= 0.40:
        return f"\\cellcolor{{orange!20}}{s}"
    else:
        return f"\\cellcolor{{red!20}}{s}"


# ─── Table 1: EMR under greedy decoding ──────────────────────────────────────

def generate_emr_greedy_table(metrics):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Exact Match Rate (EMR) under greedy decoding ($t{=}0$) across five models and two tasks. Higher is more reproducible. Local models achieve near-perfect bitwise reproducibility while API-served models exhibit substantial hidden non-determinism.}")
    lines.append(r"\label{tab:emr_greedy}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Source} & \textbf{Extraction} & \textbf{Summarization} & \textbf{$N$ Runs} & \textbf{$N$ Abstracts} \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        name = MODEL_NAMES[model]
        source = "Local" if model in ["llama3_8b", "mistral_7b", "gemma2_9b"] else "API"

        ext = lookup_greedy(metrics, model, "extraction")
        summ = lookup_greedy(metrics, model, "summarization")

        ext_emr = fmt_emr(ext["emr_mean"]) if ext else "---"
        summ_emr = fmt_emr(summ["emr_mean"]) if summ else "---"

        n_runs = (ext["n_runs"] if ext else 0) + (summ["n_runs"] if summ else 0)
        n_abs = max(ext["n_abstracts"] if ext else 0, summ["n_abstracts"] if summ else 0)

        lines.append(f"  {name} & {source} & {ext_emr} & {summ_emr} & {n_runs} & {n_abs} \\\\")

        if model == "llama3_8b":
            lines.append(r"  \midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ─── Table 2: Three-level reproducibility metrics ────────────────────────────

def generate_three_level_table(metrics):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Three-level reproducibility assessment under greedy decoding ($t{=}0$). L1: bitwise identity (EMR), L2: surface similarity (NED, ROUGE-L), L3: semantic equivalence (BERTScore F1). Values are means across abstracts.}")
    lines.append(r"\label{tab:three_level}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{2}{c}{\textbf{L1: Bitwise}} & \multicolumn{2}{c}{\textbf{L2: Surface}} & \multicolumn{1}{c}{\textbf{L3: Semantic}} \\")
    lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-7}")
    lines.append(r"\textbf{Model} & \textbf{Task} & \textbf{EMR} & \textbf{$\sigma$} & \textbf{NED}$\downarrow$ & \textbf{ROUGE-L}$\uparrow$ & \textbf{BERTScore F1}$\uparrow$ \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        name = MODEL_NAMES[model]
        first = True
        for task in ["extraction", "summarization"]:
            m = lookup_greedy(metrics, model, task)
            if m is None:
                continue
            mname = name if first else ""
            first = False
            tname = TASK_NAMES[task]
            emr = fmt(m["emr_mean"])
            emr_std = fmt(m.get("emr_std"), 3)
            ned = fmt(m["ned_mean"])
            rouge = fmt(m["rouge_l_mean"])
            bert = fmt(m["bertscore_f1_mean"], 4)
            lines.append(f"  {mname} & {tname} & {emr} & {emr_std} & {ned} & {rouge} & {bert} \\\\")

        if model == "llama3_8b":
            lines.append(r"  \midrule")
        elif model != "claude_sonnet":
            lines.append(r"  \addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ─── Table 3: Temperature sweep ──────────────────────────────────────────────

def generate_temp_sweep_table(metrics):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Effect of sampling temperature on Exact Match Rate (EMR). Temperature is the dominant user-controllable variability factor across all models. At $t{=}0.7$, all models achieve EMR${=}0$ for summarization.}")
    lines.append(r"\label{tab:temp_sweep}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Task} & \textbf{$t{=}0.0$} & \textbf{$t{=}0.3$} & \textbf{$t{=}0.7$} \\")
    lines.append(r"\midrule")

    temp_conds = {
        "0.0": ["C3_temp0.0", "C3_temp0_0"],
        "0.3": ["C3_temp0.3", "C3_temp0_3"],
        "0.7": ["C3_temp0.7", "C3_temp0_7"],
    }

    for model in MODEL_ORDER:
        name = MODEL_NAMES[model]
        first = True
        for task in ["extraction", "summarization"]:
            mname = name if first else ""
            first = False
            tname = TASK_NAMES[task]
            vals = []
            for temp in ["0.0", "0.3", "0.7"]:
                found = None
                for cond in temp_conds[temp]:
                    found = lookup(metrics, model, task, cond)
                    if found:
                        break
                if found:
                    vals.append(fmt_emr(found["emr_mean"]))
                else:
                    vals.append("---")
            lines.append(f"  {mname} & {tname} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

        if model == "llama3_8b":
            lines.append(r"  \midrule")
        elif model != "claude_sonnet":
            lines.append(r"  \addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ─── Table 4: Multi-turn and RAG ─────────────────────────────────────────────

def generate_multiturn_rag_table(metrics):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Reproducibility under complex interaction regimes (C1 fixed seed, $t{=}0$). Multi-turn refinement involves three successive prompt--response exchanges. RAG extraction augments the prompt with a retrieved context passage.}")
    lines.append(r"\label{tab:multiturn_rag}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Scenario} & \textbf{EMR} & \textbf{NED}$\downarrow$ & \textbf{ROUGE-L}$\uparrow$ & \textbf{BS-F1}$\uparrow$ \\")
    lines.append(r"\midrule")

    local_models = ["gemma2_9b", "mistral_7b", "llama3_8b"]
    tasks = [
        ("extraction", "Single-turn Extraction"),
        ("summarization", "Single-turn Summarization"),
        ("multiturn_refinement", "Multi-turn Refinement"),
        ("rag_extraction", "RAG Extraction"),
    ]

    for model in local_models:
        name = MODEL_NAMES[model]
        first = True
        for task_id, task_label in tasks:
            m = lookup(metrics, model, task_id, "C1_fixed_seed")
            if m is None:
                continue
            mname = name if first else ""
            first = False
            emr = fmt(m["emr_mean"])
            ned = fmt(m["ned_mean"])
            rouge = fmt(m["rouge_l_mean"])
            bert = fmt(m["bertscore_f1_mean"], 4)
            lines.append(f"  {mname} & {task_label} & {emr} & {ned} & {rouge} & {bert} \\\\")
        lines.append(r"  \addlinespace")

    # Remove trailing addlinespace
    if lines[-1].strip() == r"\addlinespace":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ─── Table 5: Protocol overhead ──────────────────────────────────────────────

def generate_overhead_table(metrics):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Provenance logging overhead across five models under greedy decoding (C1). The protocol adds negligible overhead (${<}1\%$) to inference latency across all models and deployment modes.}")
    lines.append(r"\label{tab:overhead}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Source} & \textbf{Mean Inference (ms)} & \textbf{Mean Overhead (ms)} & \textbf{Overhead (\%)} \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        name = MODEL_NAMES[model]
        source = "Local" if model in ["llama3_8b", "mistral_7b", "gemma2_9b"] else "API"

        durations = []
        overheads = []
        overhead_pcts = []
        for m in metrics:
            if m["model"] == model and m["condition"] in ["C1_fixed_seed", "C2_same_params"]:
                t = m.get("timing", {})
                if t.get("mean_duration_ms"):
                    durations.append(t["mean_duration_ms"])
                if t.get("mean_overhead_ms"):
                    overheads.append(t["mean_overhead_ms"])
                if t.get("overhead_pct"):
                    overhead_pcts.append(t["overhead_pct"])

        if durations:
            mean_dur = sum(durations) / len(durations)
            mean_oh = sum(overheads) / len(overheads)
            mean_pct = sum(overhead_pcts) / len(overhead_pcts)
            lines.append(f"  {name} & {source} & {mean_dur:,.1f} & {mean_oh:.1f} & {mean_pct:.3f} \\\\")
        else:
            lines.append(f"  {name} & {source} & --- & --- & --- \\\\")

        if model == "llama3_8b":
            lines.append(r"  \midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ─── Table 6: API vs Local comparison ────────────────────────────────────────

def generate_api_vs_local_table(metrics):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{API-served vs.\ locally deployed models under greedy decoding. Values are averaged across tasks and abstracts. Local models exhibit dramatically higher bitwise reproducibility, confirming that server-side non-determinism---not user-controllable parameters---is the primary source of variability in API-served models.}")
    lines.append(r"\label{tab:api_vs_local}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Deployment} & \textbf{EMR}$\uparrow$ & \textbf{NED}$\downarrow$ & \textbf{ROUGE-L}$\uparrow$ & \textbf{BS-F1}$\uparrow$ \\")
    lines.append(r"\midrule")

    local_models = {"llama3_8b", "mistral_7b", "gemma2_9b"}
    api_models = {"gpt4", "claude_sonnet"}

    for label, model_set in [("Local (3 models)", local_models), ("API (2 models)", api_models)]:
        emrs, neds, rouges, berts = [], [], [], []
        for m in metrics:
            if m["model"] in model_set and m["condition"] in ["C1_fixed_seed", "C2_var_seed", "C2_same_params"]:
                if m["emr_mean"] is not None:
                    emrs.append(m["emr_mean"])
                if m["ned_mean"] is not None:
                    neds.append(m["ned_mean"])
                if m["rouge_l_mean"] is not None:
                    rouges.append(m["rouge_l_mean"])
                if m["bertscore_f1_mean"] is not None:
                    berts.append(m["bertscore_f1_mean"])

        emr_avg = sum(emrs) / len(emrs) if emrs else None
        ned_avg = sum(neds) / len(neds) if neds else None
        rouge_avg = sum(rouges) / len(rouges) if rouges else None
        bert_avg = sum(berts) / len(berts) if berts else None

        lines.append(f"  {label} & {fmt(emr_avg)} & {fmt(ned_avg)} & {fmt(rouge_avg)} & {fmt(bert_avg, 4)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metrics, comparison = load_data()
    print(f"Loaded {len(metrics)} metric entries")

    tables = {
        "table_emr_greedy.tex": generate_emr_greedy_table(metrics),
        "table_three_level.tex": generate_three_level_table(metrics),
        "table_temp_sweep.tex": generate_temp_sweep_table(metrics),
        "table_multiturn_rag.tex": generate_multiturn_rag_table(metrics),
        "table_overhead.tex": generate_overhead_table(metrics),
        "table_api_vs_local.tex": generate_api_vs_local_table(metrics),
    }

    for filename, content in tables.items():
        path = TABLES_DIR / filename
        with open(path, "w") as f:
            f.write(content + "\n")
        print(f"  Written: {path}")

    # Also write combined file
    combined = ANALYSIS_DIR / "latex_tables_v3.tex"
    with open(combined, "w") as f:
        f.write("% Auto-generated LaTeX tables for JAIR paper (v3: 5 models, 3504 runs)\n")
        f.write("% Requires: \\usepackage[table]{xcolor} for cellcolor\n\n")
        for filename, content in tables.items():
            f.write(f"% === {filename} ===\n")
            f.write(content)
            f.write("\n\n")
    print(f"  Combined: {combined}")

    print(f"\nAll {len(tables)} tables generated.")


if __name__ == "__main__":
    main()
