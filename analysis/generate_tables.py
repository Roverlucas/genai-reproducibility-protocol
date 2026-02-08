#!/usr/bin/env python3
"""Generate LaTeX tables from the analysis results for the JAIR paper.

Creates publication-quality LaTeX tables for:
- Table 1: Experimental design overview
- Table 2: Variability metrics per condition (main results)
- Table 3: Protocol overhead metrics
- Table 4: Comparison with existing tools/frameworks
- Table 5: Per-abstract variability breakdown
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ANALYSIS_DIR = Path(__file__).parent


def load_analysis():
    with open(ANALYSIS_DIR / "full_analysis.json") as f:
        return json.load(f)


def table_experimental_design():
    """Table 1: Experimental design overview."""
    return r"""
\begin{table}[t]
\centering
\caption{Experimental design: conditions, parameters, and expected outcomes.}
\label{tab:experimental-design}
\small
\begin{tabular}{@{}llcccl@{}}
\toprule
\textbf{Cond.} & \textbf{Description} & \textbf{Temp.} & \textbf{Seed} & \textbf{Reps} & \textbf{Expected Outcome} \\
\midrule
C1 & Fixed seed, greedy & 0.0 & 42 (fixed) & 5 & Deterministic output \\
C2 & Variable seeds, greedy & 0.0 & 5 different & 5 & Near-deterministic \\
C3$_{t{=}0.0}$ & Temp.\ baseline & 0.0 & per-rep & 3 & Deterministic \\
C3$_{t{=}0.3}$ & Low temperature & 0.3 & per-rep & 3 & Low variability \\
C3$_{t{=}0.7}$ & High temperature & 0.7 & per-rep & 3 & High variability \\
\bottomrule
\end{tabular}
\vspace{4pt}
\raggedright\footnotesize
Experiments run on LLaMA~3 8B via Ollama (local, Apple M4, 24\,GB RAM). \\
Each condition applied to 5 abstracts $\times$ 2 tasks = 10 groups per condition. \\
Total: 190 logged runs.
\end{table}
"""


def table_main_results(analysis):
    """Table 2: Main variability results."""
    agg = analysis["variability_aggregated"]

    rows = []
    condition_order = ["C1", "C2", "C3_t0.0", "C3_t0.3", "C3_t0.7"]
    condition_labels = {
        "C1": r"C1 (fixed seed, $t{=}0$)",
        "C2": r"C2 (var.\ seeds, $t{=}0$)",
        "C3_t0.0": r"C3 ($t{=}0.0$)",
        "C3_t0.3": r"C3 ($t{=}0.3$)",
        "C3_t0.7": r"C3 ($t{=}0.7$)",
    }

    for task in ["summarization", "extraction"]:
        for cond in condition_order:
            key = f"{task}_{cond}"
            if key not in agg:
                continue
            v = agg[key]
            em = v["exact_match_rate"]["mean"]
            ed = v["edit_distance_normalized"]["mean"]
            rl = v["rouge_l"]["mean"]
            rows.append((task, cond, em, ed, rl))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Output variability across experimental conditions (mean over 5~abstracts). Exact Match Rate (EMR), Normalized Edit Distance (NED), and ROUGE-L F1 computed over all pairwise comparisons within each condition group.}")
    lines.append(r"\label{tab:variability-results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Task} & \textbf{Condition} & \textbf{EMR}$\uparrow$ & \textbf{NED}$\downarrow$ & \textbf{ROUGE-L}$\uparrow$ \\")
    lines.append(r"\midrule")

    prev_task = None
    for task, cond, em, ed, rl in rows:
        if prev_task is not None and task != prev_task:
            lines.append(r"\midrule")
        task_label = task.capitalize() if task != prev_task else ""
        cond_label = condition_labels[cond]

        # Bold the best values
        em_str = f"{em:.3f}"
        ed_str = f"{ed:.4f}"
        rl_str = f"{rl:.4f}"

        if em == 1.0:
            em_str = r"\textbf{1.000}"
        if ed == 0.0:
            ed_str = r"\textbf{0.0000}"
        if rl == 1.0:
            rl_str = r"\textbf{1.0000}"

        lines.append(f"  {task_label} & {cond_label} & {em_str} & {ed_str} & {rl_str} \\\\")
        prev_task = task

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def table_overhead(analysis):
    """Table 3: Protocol overhead metrics."""
    oh = analysis["overhead"]
    log_oh = oh["logging_overhead"]
    stor_oh = oh["storage_overhead"]
    ratio_oh = oh["overhead_ratio"]
    dirs = oh["directory_sizes"]

    return rf"""
\begin{{table}}[t]
\centering
\caption{{Protocol overhead: logging time and storage costs for 190~runs.}}
\label{{tab:overhead}}
\small
\begin{{tabular}}{{@{{}}lrl@{{}}}}
\toprule
\textbf{{Metric}} & \textbf{{Value}} & \textbf{{Unit}} \\
\midrule
\multicolumn{{3}}{{l}}{{\textit{{Logging time overhead}}}} \\
\quad Mean per run & {log_oh['mean_ms']:.2f} $\pm$ {log_oh['std_ms']:.2f} & ms \\
\quad Min / Max & {log_oh['min_ms']:.2f} / {log_oh['max_ms']:.2f} & ms \\
\quad Total (190 runs) & {log_oh['total_ms']:.1f} & ms \\
\quad Mean overhead ratio & {ratio_oh['mean_percent']:.3f}\% & of inference time \\
\quad Max overhead ratio & {ratio_oh['max_percent']:.3f}\% & of inference time \\
\midrule
\multicolumn{{3}}{{l}}{{\textit{{Storage overhead}}}} \\
\quad Mean per run record & {stor_oh['mean_kb']:.2f} $\pm$ {stor_oh['std_kb']:.2f} & KB \\
\quad Run logs (190~files) & {dirs['runs']['total_kb']:.1f} & KB \\
\quad PROV documents (191~files) & {dirs['provenance']['total_kb']:.1f} & KB \\
\quad Run Cards (190~files) & {dirs['run_cards']['total_kb']:.1f} & KB \\
\quad Total output & {dirs['total_output']['total_mb']:.2f} & MB \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_comparison():
    """Table 4: Comparison with existing frameworks."""
    return r"""
\begin{table*}[t]
\centering
\caption{Comparison of our protocol with existing reproducibility tools and frameworks for GenAI experiments. Checkmarks (\ding{51}) indicate full support; tildes ($\sim$) indicate partial support; dashes (--) indicate no support.}
\label{tab:comparison}
\small
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Feature} & \textbf{Ours} & \textbf{MLflow} & \textbf{W\&B} & \textbf{DVC} & \textbf{OpenAI Evals} & \textbf{LangSmith} \\
\midrule
Prompt versioning (Prompt Card) & \ding{51} & -- & $\sim$ & -- & $\sim$ & $\sim$ \\
Run-level provenance (W3C PROV) & \ding{51} & -- & -- & -- & -- & -- \\
Cryptographic output hashing & \ding{51} & -- & -- & \ding{51} & -- & -- \\
Seed \& param logging & \ding{51} & \ding{51} & \ding{51} & -- & \ding{51} & \ding{51} \\
Environment fingerprinting & \ding{51} & $\sim$ & $\sim$ & $\sim$ & -- & -- \\
Model weights hashing & \ding{51} & -- & $\sim$ & \ding{51} & -- & -- \\
Overhead $<$1\% of inference & \ding{51} & $\sim$ & $\sim$ & N/A & N/A & $\sim$ \\
Designed for GenAI text output & \ding{51} & -- & -- & -- & \ding{51} & \ding{51} \\
Open standard (PROV-JSON) & \ding{51} & -- & -- & -- & -- & -- \\
Local-first (no cloud dependency) & \ding{51} & \ding{51} & -- & \ding{51} & -- & -- \\
\bottomrule
\end{tabular}
\end{table*}
"""


def table_per_abstract(analysis):
    """Table 5: Per-abstract variability for key conditions."""
    var = analysis["variability_per_abstract"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-abstract Exact Match Rate for the summarization task across conditions. Each cell reports EMR over all pairwise output comparisons within that group.}")
    lines.append(r"\label{tab:per-abstract}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Condition} & \textbf{abs\_001} & \textbf{abs\_002} & \textbf{abs\_003} & \textbf{abs\_004} & \textbf{abs\_005} \\")
    lines.append(r"\midrule")

    condition_order = ["C1", "C2", "C3_t0.0", "C3_t0.3", "C3_t0.7"]
    condition_labels = {
        "C1": r"C1 (fixed seed)",
        "C2": r"C2 (var.\ seeds)",
        "C3_t0.0": r"C3 ($t{=}0.0$)",
        "C3_t0.3": r"C3 ($t{=}0.3$)",
        "C3_t0.7": r"C3 ($t{=}0.7$)",
    }

    for cond in condition_order:
        cells = [condition_labels[cond]]
        for abs_id in ["abs_001", "abs_002", "abs_003", "abs_004", "abs_005"]:
            key = f"summarization_{cond}_{abs_id}"
            if key in var:
                em = var[key]["exact_match_rate"]
                if em == 1.0:
                    cells.append(r"\textbf{1.00}")
                elif em == 0.0:
                    cells.append("0.00")
                else:
                    cells.append(f"{em:.2f}")
            else:
                cells.append("--")
        lines.append("  " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def table_execution_times(analysis):
    """Table 6: Execution time statistics."""
    exec_times = analysis["execution_times"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Inference execution time (ms) per task and condition. Protocol overhead excluded.}")
    lines.append(r"\label{tab:exec-times}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Task} & \textbf{Condition} & \textbf{N} & \textbf{Mean} & \textbf{Std} & \textbf{Median} \\")
    lines.append(r"\midrule")

    condition_order = ["C1", "C2", "C3_t0.0", "C3_t0.3", "C3_t0.7"]
    prev_task = None
    for task in ["summarization", "extraction"]:
        for cond in condition_order:
            key = f"{task}_{cond}"
            if key not in exec_times:
                continue
            v = exec_times[key]
            if prev_task is not None and task != prev_task:
                lines.append(r"\midrule")
            task_label = task.capitalize() if task != prev_task else ""
            lines.append(
                f"  {task_label} & {cond} & {v['n_runs']} & "
                f"{v['mean_ms']:.0f} & {v['std_ms']:.0f} & {v['median_ms']:.0f} \\\\"
            )
            prev_task = task

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    analysis = load_analysis()

    tables = {
        "table1_experimental_design": table_experimental_design(),
        "table2_main_results": table_main_results(analysis),
        "table3_overhead": table_overhead(analysis),
        "table4_comparison": table_comparison(),
        "table5_per_abstract": table_per_abstract(analysis),
        "table6_exec_times": table_execution_times(analysis),
    }

    # Save all tables to a single file
    output_path = ANALYSIS_DIR / "latex_tables.tex"
    with open(output_path, "w") as f:
        f.write("% Auto-generated LaTeX tables for JAIR paper\n")
        f.write("% Generated from 190 LLaMA 3 8B experimental runs\n\n")
        for name, content in tables.items():
            f.write(f"% === {name} ===\n")
            f.write(content)
            f.write("\n\n")

    print(f"[OK] LaTeX tables saved to: {output_path}")

    # Also save individually
    for name, content in tables.items():
        individual_path = ANALYSIS_DIR / f"{name}.tex"
        with open(individual_path, "w") as f:
            f.write(content)
        print(f"  -> {individual_path}")


if __name__ == "__main__":
    main()
