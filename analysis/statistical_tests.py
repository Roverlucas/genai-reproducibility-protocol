"""
Statistical tests for the cross-model comparison (LLaMA vs GPT-4).
Computes: paired t-tests, 95% CIs, Cohen's d, per-abstract breakdown.
"""
import json
import numpy as np
from scipy import stats
from itertools import combinations

# Load all runs
with open("/Users/lucasrover/paper-experiment/outputs/all_runs.json") as f:
    runs = json.load(f)

# ----- Helper functions (same as compute_metrics.py) -----
def normalized_edit_distance(a, b):
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    d = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1):
        d[i][0] = i
    for j in range(lb+1):
        d[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[la][lb] / max(la, lb)

def rouge_l_f1(a, b):
    wa = a.split()
    wb = b.split()
    if not wa or not wb:
        return 0.0
    m, n = len(wa), len(wb)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if wa[i-1] == wb[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)

def parse_abstract(run):
    import re
    rid = run.get("run_id", "")
    match = re.search(r"(abs_\d+)", rid)
    if match:
        return match.group(1)
    return "unknown"

def parse_model(run):
    mn = run.get("model_name", "")
    if "llama" in mn.lower():
        return "llama"
    elif "gpt" in mn.lower():
        return "gpt4"
    return "unknown"

def parse_task(run):
    tid = run.get("task_id", "")
    if "extract" in tid.lower():
        return "extraction"
    elif "summar" in tid.lower():
        return "summarization"
    return tid

def parse_condition(run):
    rid = run.get("run_id", "")
    if "_C1_" in rid:
        return "C1"
    elif "_C2_" in rid:
        return "C2"
    elif "_C3_" in rid:
        temp = run.get("inference_params", {}).get("temperature", 0)
        return f"C3_t{temp}"
    return "unknown"

# ----- Group runs and compute per-abstract metrics -----
def compute_group_metrics(group_runs):
    """Compute EMR, NED, ROUGE-L for a group of runs (pairwise)."""
    outputs = [r["output_text"] for r in group_runs]
    n = len(outputs)
    if n < 2:
        return {"emr": None, "ned": None, "rouge_l": None}

    pairs = list(combinations(range(n), 2))
    exact = sum(1 for i, j in pairs if outputs[i] == outputs[j])
    emr = exact / len(pairs)
    ned = np.mean([normalized_edit_distance(outputs[i], outputs[j]) for i, j in pairs])
    rl = np.mean([rouge_l_f1(outputs[i], outputs[j]) for i, j in pairs])
    return {"emr": emr, "ned": ned, "rouge_l": rl}

# Build groups: (model, task, condition, abstract)
groups = {}
for r in runs:
    model = parse_model(r)
    task = parse_task(r)
    cond = parse_condition(r)
    abstract = parse_abstract(r)
    key = (model, task, cond, abstract)
    groups.setdefault(key, []).append(r)

# ----- Per-abstract metrics for C2 condition (greedy, both models) -----
print("=" * 80)
print("PER-ABSTRACT METRICS UNDER C2 (GREEDY DECODING)")
print("=" * 80)

abstracts = sorted(set(parse_abstract(r) for r in runs if parse_abstract(r) != "unknown"))
tasks = ["summarization", "extraction"]
models = ["llama", "gpt4"]
print(f"Found {len(abstracts)} abstracts: {abstracts[:5]}...{abstracts[-3:]}")

# Collect per-abstract EMR for paired t-tests
per_abstract = {}  # (model, task) -> [emr_abs1, ..., emr_abs5]

for model in models:
    for task in tasks:
        emrs, neds, rls = [], [], []
        for ab in abstracts:
            key = (model, task, "C2", ab)
            if key in groups:
                m = compute_group_metrics(groups[key])
                emrs.append(m["emr"])
                neds.append(m["ned"])
                rls.append(m["rouge_l"])
            else:
                emrs.append(None)
                neds.append(None)
                rls.append(None)

        per_abstract[(model, task)] = {
            "emr": emrs, "ned": neds, "rouge_l": rls
        }

        model_name = "LLaMA 3 8B" if model == "llama" else "GPT-4"
        print(f"\n{model_name} | {task} | C2:")
        for i, ab in enumerate(abstracts):
            e = emrs[i] if emrs[i] is not None else "N/A"
            n = neds[i] if neds[i] is not None else "N/A"
            r = rls[i] if rls[i] is not None else "N/A"
            if isinstance(e, float):
                print(f"  {ab}: EMR={e:.3f}, NED={n:.4f}, ROUGE-L={r:.4f}")
            else:
                print(f"  {ab}: {e}")

# ----- Paired t-tests: LLaMA vs GPT-4 under C2 -----
print("\n" + "=" * 80)
print("PAIRED T-TESTS: LLaMA vs GPT-4 under C2 (greedy)")
print("=" * 80)

for task in tasks:
    llama_emr = np.array(per_abstract[("llama", task)]["emr"])
    gpt4_emr = np.array(per_abstract[("gpt4", task)]["emr"])
    llama_ned = np.array(per_abstract[("llama", task)]["ned"])
    gpt4_ned = np.array(per_abstract[("gpt4", task)]["ned"])
    llama_rl = np.array(per_abstract[("llama", task)]["rouge_l"])
    gpt4_rl = np.array(per_abstract[("gpt4", task)]["rouge_l"])

    print(f"\n--- {task.upper()} ---")

    for metric_name, llama_vals, gpt4_vals in [
        ("EMR", llama_emr, gpt4_emr),
        ("NED", llama_ned, gpt4_ned),
        ("ROUGE-L", llama_rl, gpt4_rl),
    ]:
        diff = llama_vals - gpt4_vals
        n = len(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(llama_vals, gpt4_vals)

        # Cohen's d (paired)
        cohens_d = mean_diff / std_diff if std_diff > 0 else float('inf')

        # 95% CI for the difference
        se = std_diff / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_low = mean_diff - t_crit * se
        ci_high = mean_diff + t_crit * se

        print(f"  {metric_name}:")
        print(f"    LLaMA mean: {np.mean(llama_vals):.4f} (std={np.std(llama_vals, ddof=1):.4f})")
        print(f"    GPT-4 mean: {np.mean(gpt4_vals):.4f} (std={np.std(gpt4_vals, ddof=1):.4f})")
        print(f"    Diff: {mean_diff:.4f} [{ci_low:.4f}, {ci_high:.4f}] 95% CI")
        print(f"    t({n-1}) = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"    Cohen's d = {cohens_d:.3f}")
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"    Significance: {sig}")

# ----- 95% CIs for individual model metrics -----
print("\n" + "=" * 80)
print("95% CONFIDENCE INTERVALS (per model, C2)")
print("=" * 80)

for model in models:
    for task in tasks:
        model_name = "LLaMA 3 8B" if model == "llama" else "GPT-4"
        emrs = np.array(per_abstract[(model, task)]["emr"])
        n = len(emrs)
        mean = np.mean(emrs)
        std = np.std(emrs, ddof=1)
        se = std / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci = (mean - t_crit * se, mean + t_crit * se)
        print(f"  {model_name} {task} EMR: {mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

# ----- Word counts of abstracts -----
print("\n" + "=" * 80)
print("ABSTRACT WORD COUNTS")
print("=" * 80)
with open("/Users/lucasrover/paper-experiment/data/inputs/abstracts.json") as f:
    abstracts_data = json.load(f)
for ab in abstracts_data["abstracts"]:
    wc = len(ab["text"].split())
    print(f"  {ab['id']}: {wc} words")
print(f"  Range: {min(len(a['text'].split()) for a in abstracts_data['abstracts'])}--{max(len(a['text'].split()) for a in abstracts_data['abstracts'])}")

# ----- Protocol code line count -----
print("\n" + "=" * 80)
print("PROTOCOL CODE SIZE")
print("=" * 80)
import os
total = 0
for root, dirs, files in os.walk("/Users/lucasrover/paper-experiment/src/protocol"):
    for f in files:
        if f.endswith(".py") and f != "__init__.py":
            path = os.path.join(root, f)
            with open(path) as fh:
                lines = len(fh.readlines())
            total += lines
            print(f"  {f}: {lines} lines")
print(f"  Total protocol core: {total} lines")

# ----- Post-hoc power analysis -----
print("\n" + "=" * 80)
print("POST-HOC POWER ANALYSIS")
print("=" * 80)

from scipy.stats import norm

def post_hoc_power(cohens_d, n, alpha=0.05):
    """Compute post-hoc power for a paired t-test."""
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    ncp = abs(cohens_d) * np.sqrt(n)  # non-centrality parameter
    # Approximate using normal distribution
    power = 1 - norm.cdf(t_crit - ncp) + norm.cdf(-t_crit - ncp)
    return power

n_abstracts = len(abstracts)
print(f"Sample size: {n_abstracts} abstracts")

for task in tasks:
    llama_emr = np.array([x for x in per_abstract[("llama", task)]["emr"] if x is not None])
    gpt4_emr = np.array([x for x in per_abstract[("gpt4", task)]["emr"] if x is not None])
    n = min(len(llama_emr), len(gpt4_emr))
    if n < 2:
        continue

    diff = llama_emr[:n] - gpt4_emr[:n]
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    d = abs(mean_diff / std_diff) if std_diff > 0 else float('inf')
    power = post_hoc_power(d, n) if d != float('inf') else 1.0

    print(f"\n  {task.upper()} (EMR, LLaMA vs GPT-4, C2):")
    print(f"    n={n}, Cohen's d={d:.3f}, power={power:.3f}")
    if power >= 0.80:
        print(f"    -> ADEQUATE (>= 0.80)")
    else:
        # Compute required n for 0.80 power
        for target_n in range(n, 200):
            if post_hoc_power(d, target_n) >= 0.80:
                print(f"    -> INSUFFICIENT (need n={target_n} for 0.80 power)")
                break

# ----- Effect sizes across all conditions -----
print("\n" + "=" * 80)
print("EFFECT SIZES: ALL CONDITIONS (Cohen's d, LLaMA vs GPT-4)")
print("=" * 80)
print(f"{'Task':<15} {'Condition':<12} {'Metric':<10} {'Cohen d':>8} {'Power':>8}")
print("-" * 60)

for task in tasks:
    for cond in ["C2", "C3_t0.0", "C3_t0.3", "C3_t0.7"]:
        llama_vals = []
        gpt4_vals = []
        for ab in abstracts:
            lk = ("llama", task, cond, ab)
            gk = ("gpt4", task, cond, ab)
            if lk in groups and gk in groups:
                lm = compute_group_metrics(groups[lk])
                gm = compute_group_metrics(groups[gk])
                if lm["emr"] is not None and gm["emr"] is not None:
                    llama_vals.append(lm["emr"])
                    gpt4_vals.append(gm["emr"])

        if len(llama_vals) >= 2:
            la = np.array(llama_vals)
            ga = np.array(gpt4_vals)
            diff = la - ga
            d = abs(np.mean(diff)) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else float('inf')
            pw = post_hoc_power(d, len(la)) if d != float('inf') else 1.0
            print(f"  {task:<15} {cond:<12} {'EMR':<10} {d:>7.3f} {pw:>7.3f}")

# ----- Robustness checks: Shapiro-Wilk normality + Wilcoxon signed-rank -----
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS: SHAPIRO-WILK NORMALITY + WILCOXON SIGNED-RANK")
print("=" * 80)

for task in tasks:
    llama_emr = np.array(per_abstract[("llama", task)]["emr"])
    gpt4_emr = np.array(per_abstract[("gpt4", task)]["emr"])
    diff = llama_emr - gpt4_emr

    # Shapiro-Wilk on paired differences
    sw_stat, sw_p = stats.shapiro(diff)

    # Wilcoxon signed-rank test
    w_stat, w_p = stats.wilcoxon(llama_emr, gpt4_emr, alternative="two-sided")

    print(f"\n--- {task.upper()} (EMR, C2) ---")
    print(f"  Shapiro-Wilk on differences: W = {sw_stat:.4f}, p = {sw_p:.6f}")
    print(f"    Normality assumption: {'MET' if sw_p > 0.05 else 'VIOLATED (p < 0.05)'}")
    print(f"  Wilcoxon signed-rank: W = {w_stat:.1f}, p = {w_p:.2e}")
    print(f"    Conclusion: {'SIGNIFICANT' if w_p < 0.05 else 'NOT SIGNIFICANT'}")
