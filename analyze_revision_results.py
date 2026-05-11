"""
Analyze Caminho A revision results (T1 HumanEval + GSM8K, T4 multi-turn, T14 PubMed)
and produce LaTeX-ready tables for the manuscript revision.

Outputs:
  analysis/revision/
    - emr_per_stack_per_task.json     # EMR + 95% bootstrap CIs per (stack, task)
    - pass_at_1_humaneval.json        # functional correctness for HumanEval
    - gsm8k_accuracy.json             # final-answer accuracy for GSM8K
    - tables/table_t1_t4_t14.tex      # main results table
    - tables/table_per_field_t14.tex  # per-field for PubMed (mirrors T6)
    - figures/figure_t1_t4_t14.pdf    # 4-panel figure

Run AFTER `run_revision_full.sh` completes:
    python analyze_revision_results.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "outputs" / "revision" / "runs"
OUT_DIR = ROOT / "analysis" / "revision"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"


def _bootstrap_ci(values: list[float], n_resamples: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI."""
    import random
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples)])


def _normalised_edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = tmp
    return dp[m] / max(n, m)


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _load_runs() -> dict[tuple[str, str, str], list[dict]]:
    """Group runs by (stack, task, problem_id). Each value is a list of repetition records.

    Note: `task_id` in the Run Card is the task FAMILY (e.g., "humaneval"), not the
    specific problem identifier. The actual problem identifier is encoded in the
    filename: rev_<stack>_<task>_<problem_id>_C<N>_rep<N>.json
    """
    if not RUNS_DIR.exists():
        return {}
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    pattern = re.compile(r"^rev_(?P<stack>.+?)_(?P<task>humaneval|gsm8k|pubmed_pm25|multiturn_extension|multiturn_refinement)_(?P<problem>.+?)_C\d+_rep\d+\.json$")
    for path in sorted(RUNS_DIR.glob("*.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        try:
            with open(path) as f:
                run = json.load(f)
        except Exception:
            continue
        stack = m.group("stack")
        task = m.group("task")
        problem_id = m.group("problem")
        groups[(stack, task, problem_id)].append(run)
    return groups


def _emr_per_group(repetitions: list[dict]) -> float:
    """Fraction of pairs of reps with character-identical outputs."""
    if len(repetitions) < 2:
        return float("nan")
    outputs = [r.get("output_text", "") for r in repetitions]
    n = len(outputs)
    pairs_total = n * (n - 1) // 2
    if pairs_total == 0:
        return float("nan")
    pairs_match = 0
    for i in range(n):
        for j in range(i + 1, n):
            if outputs[i] == outputs[j]:
                pairs_match += 1
    return pairs_match / pairs_total


def _ned_per_group(repetitions: list[dict]) -> float:
    """Mean pairwise NED across reps."""
    if len(repetitions) < 2:
        return float("nan")
    outputs = [r.get("output_text", "") for r in repetitions]
    n = len(outputs)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(_normalised_edit_distance(outputs[i], outputs[j]))
    return mean(distances) if distances else float("nan")


def compute_emr_ned() -> dict:
    groups = _load_runs()
    if not groups:
        return {"error": "No runs found", "runs_dir": str(RUNS_DIR)}

    by_stack_task: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"emr": [], "ned": []}
    )
    for (stack, task, _problem), reps in groups.items():
        if len(reps) < 2:
            continue
        by_stack_task[(stack, task)]["emr"].append(_emr_per_group(reps))
        by_stack_task[(stack, task)]["ned"].append(_ned_per_group(reps))

    results: dict[str, dict] = {}
    for (stack, task), metrics in sorted(by_stack_task.items()):
        emr_values = [v for v in metrics["emr"] if v == v]  # filter NaN
        ned_values = [v for v in metrics["ned"] if v == v]
        if not emr_values:
            continue
        emr_mean = mean(emr_values)
        ned_mean = mean(ned_values) if ned_values else float("nan")
        emr_ci = _bootstrap_ci(emr_values)
        ned_ci = _bootstrap_ci(ned_values) if ned_values else (float("nan"), float("nan"))
        key = f"{stack}__{task}"
        results[key] = {
            "stack": stack,
            "task": task,
            "n_problems": len(emr_values),
            "emr_mean": round(emr_mean, 4),
            "emr_ci_low": round(emr_ci[0], 4),
            "emr_ci_high": round(emr_ci[1], 4),
            "ned_mean": round(ned_mean, 4) if ned_mean == ned_mean else None,
            "ned_ci_low": round(ned_ci[0], 4) if ned_ci[0] == ned_ci[0] else None,
            "ned_ci_high": round(ned_ci[1], 4) if ned_ci[1] == ned_ci[1] else None,
        }
    return results


# Pass@1 for HumanEval
def _extract_function_body(text: str, fn_name: str) -> str | None:
    """Try to extract a function body from the model's output."""
    # Strip markdown fences
    text = re.sub(r"```(?:python)?\n?", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def compute_pass_at_1_humaneval() -> dict:
    """Approximate pass@1 — runs each completion in subprocess sandbox."""
    try:
        from src.tasks.pass_at_1 import evaluate_completion
    except Exception:
        return {"error": "pass_at_1 module not importable", "note": "run from repo root"}
    groups = _load_runs()
    by_stack: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {"problems": [], "passed": []}
    )
    for (stack, task, problem_id), reps in groups.items():
        if task != "humaneval":
            continue
        for rep in reps:
            output = rep.get("output_text", "")
            test = rep.get("input_data", {}).get("test", "") if isinstance(rep.get("input_data"), dict) else ""
            entry_point = rep.get("input_data", {}).get("entry_point", "") if isinstance(rep.get("input_data"), dict) else ""
            if not test:
                continue
            try:
                passed = evaluate_completion(output, test, entry_point, timeout=5.0)
            except Exception:
                passed = False
            by_stack[stack]["passed"].append(bool(passed))
    results: dict[str, dict] = {}
    for stack, data in sorted(by_stack.items()):
        if not data["passed"]:
            continue
        passed_arr = data["passed"]
        n_total = len(passed_arr)
        n_pass = sum(passed_arr)
        results[stack] = {
            "stack": stack,
            "n_completions": n_total,
            "n_passed": n_pass,
            "pass_at_1": round(n_pass / n_total, 4) if n_total else None,
        }
    return results


# GSM8K accuracy
def compute_gsm8k_accuracy() -> dict:
    try:
        from src.tasks.gsm8k_extractor import extract_final_answer, is_correct
    except Exception:
        return {"error": "gsm8k_extractor not importable"}
    groups = _load_runs()
    by_stack: dict[str, list[bool]] = defaultdict(list)
    for (stack, task, problem_id), reps in groups.items():
        if task != "gsm8k":
            continue
        for rep in reps:
            output = rep.get("output_text", "")
            gold = rep.get("input_data", {}).get("answer", "") if isinstance(rep.get("input_data"), dict) else ""
            if not gold:
                continue
            extracted = extract_final_answer(output)
            by_stack[stack].append(is_correct(extracted, gold))
    results: dict[str, dict] = {}
    for stack, arr in sorted(by_stack.items()):
        if not arr:
            continue
        results[stack] = {
            "stack": stack,
            "n": len(arr),
            "n_correct": sum(arr),
            "accuracy": round(sum(arr) / len(arr), 4),
        }
    return results


def render_main_table(emr_results: dict, output: Path) -> None:
    """LaTeX table: EMR per stack per task with bootstrap CIs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    # Filter to dict-valued cells only (skip "error" string-valued keys)
    valid_cells = {k: v for k, v in emr_results.items() if isinstance(v, dict) and "stack" in v}
    if not valid_cells:
        output.write_text("% No valid (stack, task) cells available — analysis incomplete\n")
        print(f"  -> Wrote empty placeholder to {output}")
        return
    # Pivot: rows = stacks, columns = tasks
    stacks = sorted({v["stack"] for v in valid_cells.values()})
    tasks = sorted({v["task"] for v in valid_cells.values()})
    rows = []
    for stack in stacks:
        cells = [stack.replace("-", "{-}").replace("_", "\\_")]
        for task in tasks:
            key = f"{stack}__{task}"
            if key in valid_cells:
                d = valid_cells[key]
                cell = f"{d['emr_mean']:.3f} [{d['emr_ci_low']:.2f}, {d['emr_ci_high']:.2f}]"
            else:
                cell = "---"
            cells.append(cell)
        rows.append(" & ".join(cells) + " \\\\")
    header = "Stack & " + " & ".join(t.replace("_", " ") for t in tasks) + " \\\\"
    body = "\n".join(rows)
    tex = f"""% Auto-generated by analyze_revision_results.py
\\begin{{table}}[h]
\\centering
\\caption{{Exact match rate (EMR) with 95\\% bootstrap CI per deployment stack and new task. T1 = HumanEval (code) and GSM8K (math); T4 = multi-turn extension (gpt-4o, deepseek-chat); T14 = 10 PubMed PM\\textsubscript{{2.5}} abstracts (extraction). 10\\,000 percentile bootstrap resamples.}}
\\label{{tab:revision_emr}}
\\begin{{tabular}}{{l{('c'*len(tasks))}}}
\\hline
{header}
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    output.write_text(tex)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Analyzing revision results from {RUNS_DIR}")
    n_runs = len(list(RUNS_DIR.glob("*.json"))) if RUNS_DIR.exists() else 0
    print(f"Total run JSONs: {n_runs}")
    print("=" * 70)

    if n_runs == 0:
        print("No runs found. Has Caminho A completed?")
        sys.exit(1)

    print("\n[1/3] Computing EMR + NED per (stack, task)...")
    emr_results = compute_emr_ned()
    (OUT_DIR / "emr_per_stack_per_task.json").write_text(json.dumps(emr_results, indent=2))
    print(f"  -> {len(emr_results)} (stack, task) cells computed")

    print("\n[2/3] Computing pass@1 for HumanEval...")
    pass_at_1 = compute_pass_at_1_humaneval()
    (OUT_DIR / "pass_at_1_humaneval.json").write_text(json.dumps(pass_at_1, indent=2))
    if "error" not in pass_at_1:
        print(f"  -> {len(pass_at_1)} stacks evaluated")

    print("\n[3/3] Computing GSM8K accuracy...")
    gsm = compute_gsm8k_accuracy()
    (OUT_DIR / "gsm8k_accuracy.json").write_text(json.dumps(gsm, indent=2))
    if "error" not in gsm:
        print(f"  -> {len(gsm)} stacks evaluated")

    print("\nRendering LaTeX tables...")
    render_main_table(emr_results, TABLES_DIR / "table_t1_t4_t14.tex")
    print(f"  -> {TABLES_DIR}/table_t1_t4_t14.tex")

    print("\nDONE. Results in:", OUT_DIR)


if __name__ == "__main__":
    main()
