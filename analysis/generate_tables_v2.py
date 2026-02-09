#!/usr/bin/env python3
"""Generate LaTeX tables from the full analysis (LLaMA + GPT-4) for JAIR paper."""

import json
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent


def load_analysis():
    with open(ANALYSIS_DIR / "full_analysis.json") as f:
        return json.load(f)


def table_main_results(analysis):
    """Table 2: Main variability results for both models."""
    agg = analysis["variability_aggregated"]

    condition_order_llama = ["C1", "C2", "C3_t0.0", "C3_t0.3", "C3_t0.7"]
    condition_order_gpt4 = ["C1", "C2", "C3_t0.0", "C3_t0.3", "C3_t0.7"]
    condition_labels = {
        "C1": r"C1 (fixed seed, $t{=}0$)",
        "C2": r"C2 (same params, $t{=}0$)",
        "C3_t0.0": r"C3 ($t{=}0.0$)",
        "C3_t0.3": r"C3 ($t{=}0.3$)",
        "C3_t0.7": r"C3 ($t{=}0.7$)",
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Output variability across experimental conditions for LLaMA~3 8B (local) and GPT-4 (API). Mean over 30~abstracts. EMR = Exact Match Rate, NED = Normalized Edit Distance, ROUGE-L = word-level LCS F1, BS-F1 = BERTScore F1.}")
    lines.append(r"\label{tab:variability-results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lllcccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Task} & \textbf{Condition} & \textbf{EMR}$\uparrow$ & \textbf{NED}$\downarrow$ & \textbf{ROUGE-L}$\uparrow$ & \textbf{BS-F1}$\uparrow$ \\")
    lines.append(r"\midrule")

    # LLaMA section
    prev_task = None
    for task in ["summarization", "extraction"]:
        for cond in condition_order_llama:
            key = f"llama3_8b_{task}_{cond}"
            if key not in agg:
                continue
            v = agg[key]
            em = v["exact_match_rate"]["mean"]
            ed = v["edit_distance_normalized"]["mean"]
            rl = v["rouge_l"]["mean"]
            bs = v.get("bertscore_f1", {}).get("mean")

            if prev_task is not None and task != prev_task:
                lines.append(r"\cmidrule{2-7}")
            model_label = r"\multirow{10}{*}{\rotatebox[origin=c]{90}{LLaMA~3 8B}}" if task == "summarization" and cond == "C1" else ""
            task_label = task.capitalize() if task != prev_task else ""
            cond_label = condition_labels[cond]

            em_str = r"\textbf{1.000}" if em == 1.0 else f"{em:.3f}"
            ed_str = r"\textbf{0.0000}" if ed == 0.0 else f"{ed:.4f}"
            rl_str = r"\textbf{1.0000}" if rl == 1.0 else f"{rl:.4f}"
            bs_str = f"{bs:.4f}" if bs is not None else "--"

            lines.append(f"  {model_label} & {task_label} & {cond_label} & {em_str} & {ed_str} & {rl_str} & {bs_str} \\\\")
            prev_task = task

    lines.append(r"\midrule")

    # GPT-4 section
    prev_task = None
    for task in ["summarization", "extraction"]:
        for cond in condition_order_gpt4:
            key = f"gpt4_{task}_{cond}"
            if key not in agg:
                continue
            v = agg[key]
            em = v["exact_match_rate"]["mean"]
            ed = v["edit_distance_normalized"]["mean"]
            rl = v["rouge_l"]["mean"]
            bs = v.get("bertscore_f1", {}).get("mean")

            if prev_task is not None and task != prev_task:
                lines.append(r"\cmidrule{2-7}")
            model_label = r"\multirow{10}{*}{\rotatebox[origin=c]{90}{GPT-4 (API)}}" if task == "summarization" and cond == "C1" else ""
            task_label = task.capitalize() if task != prev_task else ""
            cond_label = condition_labels[cond]

            em_str = f"{em:.3f}"
            ed_str = f"{ed:.4f}"
            rl_str = f"{rl:.4f}"
            bs_str = f"{bs:.4f}" if bs is not None else "--"

            lines.append(f"  {model_label} & {task_label} & {cond_label} & {em_str} & {ed_str} & {rl_str} & {bs_str} \\\\")
            prev_task = task

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def table_overhead(analysis):
    """Table 3: Protocol overhead metrics."""
    oh = analysis["overhead"]
    log_oh = oh["logging_overhead"]
    ratio_oh = oh["overhead_ratio"]
    dirs = oh["directory_sizes"]
    n_runs = analysis["n_total_runs"]
    n_llama = dirs["runs"]["file_count"]  # approximate
    n_prov = dirs["provenance"]["file_count"]

    return rf"""
\begin{{table}}[t]
\centering
\caption{{Protocol overhead: logging time and storage costs for {n_runs}~runs (1140 LLaMA~3 + 724 GPT-4).}}
\label{{tab:overhead}}
\small
\begin{{tabular}}{{@{{}}lrl@{{}}}}
\toprule
\textbf{{Metric}} & \textbf{{Value}} & \textbf{{Unit}} \\
\midrule
\multicolumn{{3}}{{l}}{{\textit{{Logging time overhead}}}} \\
\quad Mean per run & {log_oh['mean_ms']:.2f} $\pm$ {log_oh['std_ms']:.2f} & ms \\
\quad Min / Max & {log_oh['min_ms']:.2f} / {log_oh['max_ms']:.2f} & ms \\
\quad Total ({n_runs} runs) & {log_oh['total_ms']:.0f} & ms \\
\quad Mean overhead ratio & {ratio_oh['mean_percent']:.3f}\% & of inference time \\
\quad Max overhead ratio & {ratio_oh['max_percent']:.3f}\% & of inference time \\
\midrule
\multicolumn{{3}}{{l}}{{\textit{{Storage overhead}}}} \\
\quad Run logs ({dirs['runs']['file_count']} files) & {dirs['runs']['total_kb']:.0f} & KB \\
\quad PROV documents ({n_prov} files) & {dirs['provenance']['total_kb']:.0f} & KB \\
\quad Run Cards ({dirs['run_cards']['file_count']} files) & {dirs['run_cards']['total_kb']:.0f} & KB \\
\quad Total output & {dirs['total_output']['total_mb']:.2f} & MB \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_gpt4_vs_llama_summary(analysis):
    """Table comparing LLaMA vs GPT-4 key findings."""
    agg = analysis["variability_aggregated"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Reproducibility comparison: LLaMA~3 8B (local) vs.\ GPT-4 (API) under greedy decoding ($t{=}0$). GPT-4 shows significantly lower reproducibility due to server-side non-determinism.}")
    lines.append(r"\label{tab:model-comparison}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llcc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Task} & \textbf{Metric} & \textbf{LLaMA~3 8B} & \textbf{GPT-4} \\")
    lines.append(r"\midrule")

    for task in ["Summarization", "Extraction"]:
        task_key = task.lower()
        # LLaMA C2, GPT-4 C2 (both use t=0)
        llama_key = f"llama3_8b_{task_key}_C2"
        gpt4_key = f"gpt4_{task_key}_C2"

        if llama_key in agg and gpt4_key in agg:
            lv = agg[llama_key]
            gv = agg[gpt4_key]

            lem = lv["exact_match_rate"]["mean"]
            gem = gv["exact_match_rate"]["mean"]
            lines.append(f"  {task} & EMR & {lem:.3f} & {gem:.3f} \\\\")

            led = lv["edit_distance_normalized"]["mean"]
            ged = gv["edit_distance_normalized"]["mean"]
            lines.append(f"   & NED & {led:.4f} & {ged:.4f} \\\\")

            lrl = lv["rouge_l"]["mean"]
            grl = gv["rouge_l"]["mean"]
            lines.append(f"   & ROUGE-L & {lrl:.4f} & {grl:.4f} \\\\")

            if task == "Summarization":
                lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    analysis = load_analysis()

    tables = {
        "table2_main_results_v2": table_main_results(analysis),
        "table3_overhead_v2": table_overhead(analysis),
        "table_model_comparison": table_gpt4_vs_llama_summary(analysis),
    }

    output_path = ANALYSIS_DIR / "latex_tables_v2.tex"
    with open(output_path, "w") as f:
        f.write("% Auto-generated LaTeX tables for JAIR paper (v2: LLaMA + GPT-4)\n")
        f.write(f"% Generated from {analysis['n_total_runs']} experimental runs\n\n")
        for name, content in tables.items():
            f.write(f"% === {name} ===\n")
            f.write(content)
            f.write("\n\n")

    print(f"[OK] LaTeX tables v2 saved to: {output_path}")

    for name, content in tables.items():
        path = ANALYSIS_DIR / f"{name}.tex"
        with open(path, "w") as f:
            f.write(content)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
