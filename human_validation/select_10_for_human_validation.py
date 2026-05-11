"""
Select 10 cases from the 30-case LLM-judge pool for human rating.

Strategy:
  - Stratified sampling across the 4 disagreement kinds (direction, magnitude,
    ci, mixed) — target ~2-3 per kind.
  - Coverage of both judge-agreement and judge-disagreement cases (where
    Claude Opus and gpt-4o disagree).
  - Deterministic, seed=42.

Outputs:
  - human_validation/selected_10_cases.json — the 10 selected cases
  - human_validation/01_rating_form_rater_A.md — form for Rater A
  - human_validation/01_rating_form_rater_B.md — form for Rater B
  - human_validation/02_case_packages/case_01.md ... case_10.md — printable cases

Each case file shows the source abstract + Extraction X + Extraction Y with
X/Y order deterministically randomised per case_id. The rater is blind to
which extraction corresponds to which original LLM run (and to which LLM
produced the underlying outputs).
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HV = ROOT / "human_validation"

# Source data files
CASES_V1 = ROOT / "data" / "inputs" / "revision" / "t3_judge_cases.json"
CASES_V2 = ROOT / "data" / "inputs" / "revision" / "t3_judge_cases_extended.json"

# Companion-paper corpus (for abstract text lookup)
CORPUS_500 = Path("/Users/lucasrover/llm-evidence-synthesis-reproducibility/data/corpus/corpus_500.json")

# LLM-judge verdicts (used only for stratification — never shown to raters)
CLAUDE_V1 = ROOT / "outputs" / "revision" / "t3_judge" / "t3_judge_results.json"
CLAUDE_V2 = ROOT / "outputs" / "revision" / "t3_judge" / "extended_claude" / "t3_judge_results.json"
GPT4O_V1 = ROOT / "outputs" / "revision" / "t3_judge" / "gpt4o_original10" / "t3_judge_gpt4o_results.json"
GPT4O_V2 = ROOT / "outputs" / "revision" / "t3_judge" / "gpt4o_extended20" / "t3_judge_gpt4o_results.json"


def load_all_30_cases() -> list[dict]:
    cases: list[dict] = []
    for p in [CASES_V1, CASES_V2]:
        with open(p) as f:
            cases.extend(json.load(f)["cases"])
    assert len({c["case_id"] for c in cases}) == len(cases), "Duplicate case_ids"
    return cases


def load_abstract_lookup() -> dict[str, dict]:
    """Build corpus_id -> {abstract, title} map from companion-paper corpus_500."""
    if not CORPUS_500.exists():
        return {}
    with open(CORPUS_500) as f:
        data = json.load(f)
    return {c["corpus_id"]: {"abstract": c.get("abstract", ""), "title": c.get("title", "")}
            for c in data["corpus"]}


def load_judge_verdicts(*paths: Path) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        for r in data["results"]:
            verdicts[r["case_id"]] = r["verdict"]
    return verdicts


def select_10(cases: list[dict], claude: dict, gpt4o: dict, seed: int = 42) -> list[dict]:
    """Stratified sampling: balance kinds + include agreement/disagreement cases."""
    rng = random.Random(seed)

    # Annotate each case with judge-agreement status
    for c in cases:
        cv = claude.get(c["case_id"])
        gv = gpt4o.get(c["case_id"])
        c["_judge_agree"] = (cv is not None and gv is not None and cv == gv)
        c["_claude_verdict_hidden"] = cv  # never shown, used only for analysis
        c["_gpt4o_verdict_hidden"] = gv

    # Group by kind
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_kind[c["disagreement_kind"]].append(c)

    # Target per-kind counts so total is 10
    # Kinds: direction, magnitude, ci, mixed
    # Aim: 2-3 each, total 10. Adjust if fewer cases available.
    targets = {"direction": 2, "magnitude": 3, "ci": 3, "mixed": 2}
    actual_targets = {}
    remaining = 10
    for kind, target in sorted(targets.items()):
        n_avail = len(by_kind.get(kind, []))
        n_take = min(target, n_avail)
        actual_targets[kind] = n_take
        remaining -= n_take
    # If we underfilled (some kind had < target), redistribute
    if remaining > 0:
        for kind in sorted(by_kind):
            extra_avail = len(by_kind[kind]) - actual_targets[kind]
            take = min(extra_avail, remaining)
            actual_targets[kind] += take
            remaining -= take
            if remaining == 0:
                break

    # Within each kind, prefer 50/50 mix of judge-agree and judge-disagree
    selected: list[dict] = []
    for kind, n_take in actual_targets.items():
        pool = by_kind[kind]
        agree = [c for c in pool if c["_judge_agree"]]
        disagree = [c for c in pool if not c["_judge_agree"]]
        rng.shuffle(agree)
        rng.shuffle(disagree)
        # Take alternating from disagree first (more informative), then agree
        n_disagree = min(n_take // 2 + n_take % 2, len(disagree))
        n_agree = n_take - n_disagree
        if n_agree > len(agree):
            shortfall = n_agree - len(agree)
            n_agree = len(agree)
            extra_disagree = min(shortfall, len(disagree) - n_disagree)
            n_disagree += extra_disagree
        selected.extend(disagree[:n_disagree] + agree[:n_agree])

    rng.shuffle(selected)
    return selected[:10]


def anonymise_xy(case: dict, seed: int) -> tuple[str, str, int]:
    """Randomly assign A↔X or B↔X. Returns (x_text, y_text, x_is_run_a_int)."""
    rng = random.Random(seed)
    if rng.random() < 0.5:
        x = case["run_a"]; y = case["run_b"]; x_is_run_a = 1
    else:
        x = case["run_b"]; y = case["run_a"]; x_is_run_a = 0

    def fmt(run):
        parts = []
        if run.get("effect_measure"): parts.append(f"effect_measure: {run['effect_measure']}")
        if run.get("effect_estimate") is not None: parts.append(f"effect_estimate: {run['effect_estimate']}")
        if run.get("ci_lower") is not None and run.get("ci_upper") is not None:
            parts.append(f"95% CI: [{run['ci_lower']}, {run['ci_upper']}]")
        if run.get("lag"): parts.append(f"lag: {run['lag']}")
        if run.get("outcome_specific"): parts.append(f"outcome: {run['outcome_specific']}")
        if run.get("exposure_increment"): parts.append(f"exposure_increment: {run['exposure_increment']}")
        return "\n".join(parts) if parts else "(no extraction recorded)"

    return fmt(x), fmt(y), x_is_run_a


def write_case_file(case_num: int, case: dict, x_text: str, y_text: str,
                    abstract_text: str, title: str, out_dir: Path) -> Path:
    """Write a printable case file with abstract + X + Y (anonymised)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"case_{case_num:02d}.md"
    abstract = abstract_text.strip() if abstract_text else "(abstract text unavailable)"
    title_line = f"\n**Title:** {title.strip()}\n" if title else ""
    # Anonymous case identifier — NOT revealing model
    anonymous_id = f"HV-CASE-{case_num:03d}"
    content = f"""# Case {case_num:02d}

**Identifier (for your form):** `{anonymous_id}`
{title_line}
---

## Source abstract

{abstract}

---

## Extraction X

```
{x_text}
```

## Extraction Y

```
{y_text}
```

---

## Your rating for case {case_num:02d}

Fill in your responses on the rating form (`01_rating_form_rater_X.md`):

- **Criterion (a) Direction:** same / different / ambiguous
- **Criterion (b) Magnitude (±20%):** same / different / ambiguous
- **Criterion (c) CI overlap:** overlap / disjoint / ambiguous
- **Verdict:** truly_contradictory / semantically_equivalent / ambiguous
- **Rationale:** 1–2 sentences.
"""
    p.write_text(content)
    return p


def write_rating_form(rater_label: str, n_cases: int, out_path: Path) -> Path:
    rows = "\n".join(
        f"| case_{i+1:02d} | ☐ same  ☐ different  ☐ ambiguous | ☐ same  ☐ different  ☐ ambiguous | ☐ overlap  ☐ disjoint  ☐ ambiguous | ☐ truly_contradictory  ☐ semantically_equivalent  ☐ ambiguous | _________________________ |"
        for i in range(n_cases)
    )
    content = f"""# Rating Form — Rater {rater_label}

**Rater name:** _____________________________________________

**Affiliation:** ____________________________________________

**Date of rating:** __________________________________________

---

For each of the 10 cases, mark one box per criterion, one box for the verdict, and write a 1–2 sentence rationale. **Work through the cases in order; do not skip ahead.** Estimated time: 45–90 minutes total.

The three pre-registered criteria are defined in `00_PROTOCOL.md` §4. In brief:

- **(a) Direction**: do both extractions report the same direction (positive / null / negative)?
- **(b) Magnitude**: do the effect estimates agree within ±20% relative to their average?
- **(c) CI overlap**: do the two 95% CIs share any range?

Verdict:
- **truly_contradictory** — at least one criterion fails
- **semantically_equivalent** — all three hold
- **ambiguous** — uncertain or data missing

---

| Case | (a) Direction | (b) Magnitude | (c) CI overlap | Verdict | Rationale (1–2 sentences) |
|------|---------------|---------------|----------------|---------|---------------------------|
{rows}

---

## Notes / general comments (optional)

_(Any feedback on the criteria, edge cases, suggestions for the protocol.)_

---

**When complete:** return this file (or a scan/photo of the printed version) to
Lucas Rover at lucasrover@alunos.utfpr.edu.br. Lucas will transcribe to JSON
and compute Cohen's κ across raters and against the LLM judges.

Thank you for the time you've contributed to this validation.
"""
    out_path.write_text(content)
    return out_path


def main():
    print("=" * 70)
    print(" Human-validation case selector")
    print("=" * 70)

    cases = load_all_30_cases()
    print(f"\nLoaded {len(cases)} cases from the LLM-judge pool.")

    claude = load_judge_verdicts(CLAUDE_V1, CLAUDE_V2)
    gpt4o = load_judge_verdicts(GPT4O_V1, GPT4O_V2)
    print(f"Loaded Claude Opus verdicts: {len(claude)}, gpt-4o verdicts: {len(gpt4o)}")

    selected = select_10(cases, claude, gpt4o, seed=42)
    print(f"\nSelected 10 cases stratified by kind:")
    from collections import Counter
    print(f"  by kind: {dict(Counter(c['disagreement_kind'] for c in selected))}")
    print(f"  by model: {dict(Counter(c['model'] for c in selected))}")
    print(f"  judge-agreement cases: {sum(1 for c in selected if c['_judge_agree'])}/10")

    # Build anonymised case packages
    HV_CASES = HV / "02_case_packages"
    HV_CASES.mkdir(parents=True, exist_ok=True)
    # Wipe old case files
    for old in HV_CASES.glob("case_*.md"): old.unlink()

    abstract_lookup = load_abstract_lookup()
    print(f"Loaded {len(abstract_lookup)} abstracts from companion-paper corpus_500.json.")

    selected_records = []
    for i, c in enumerate(selected, start=1):
        case_seed = int.from_bytes(hashlib.sha256((c["case_id"] + "human").encode()).digest()[:4], "big")
        x_text, y_text, x_is_run_a = anonymise_xy(c, case_seed)
        abs_info = abstract_lookup.get(c["corpus_id"], {})
        write_case_file(i, c, x_text, y_text,
                        abs_info.get("abstract", ""),
                        abs_info.get("title", ""),
                        HV_CASES)
        selected_records.append({
            "case_num": i,
            "case_id": c["case_id"],
            "model_hidden": c["model"],  # never shown
            "corpus_id": c["corpus_id"],
            "disagreement_kind_hidden": c["disagreement_kind"],  # never shown
            "x_is_run_a": x_is_run_a,
            "randomization_seed": case_seed,
            "claude_verdict_hidden": c.get("_claude_verdict_hidden"),
            "gpt4o_verdict_hidden": c.get("_gpt4o_verdict_hidden"),
            "judges_agreed_hidden": c["_judge_agree"],
        })

    with open(HV / "selected_10_cases.json", "w") as f:
        json.dump({
            "n_cases": len(selected_records),
            "seed": 42,
            "source_pools": [str(CASES_V1), str(CASES_V2)],
            "note": "The _hidden fields are NOT shown to human raters; they are used only by compute_human_kappa.py to compute κ vs LLM judges.",
            "cases": selected_records,
        }, f, indent=2)
    print(f"\n  -> {len(selected_records)} cases packaged at {HV_CASES}")
    print(f"  -> Manifest at {HV}/selected_10_cases.json")

    # Build rating forms
    rater_a_form = write_rating_form("A (Profa. Yara de Souza Tadano, UTFPR — coauthor)", 10, HV / "01_rating_form_rater_A.md")
    rater_b_form = write_rating_form("B (external epidemiologist, independent of author list)", 10, HV / "01_rating_form_rater_B.md")
    print(f"  -> Rating form A at {rater_a_form}")
    print(f"  -> Rating form B at {rater_b_form}")

    print("\nNext: distribute the materials (see human_validation/STEP_BY_STEP_EXECUTION.md §3).")


if __name__ == "__main__":
    main()
