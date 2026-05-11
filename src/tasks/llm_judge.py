"""
Claude Opus 4.7 LLM-as-judge for T3 PM2.5 triangulation validation.

Pre-registered judgment criteria (R3 item 6 of NatComms major revision):
  (a) Same direction of the reported effect (positive / null / negative)?
  (b) Same magnitude within +/- 20%?
  (c) Same 95% CI overlap (do the intervals share any range)?

Verdict per case:
  - "truly_contradictory": at least one of (a, b, c) materially differs
  - "semantically_equivalent": all three criteria hold
  - "ambiguous": missing data or judge cannot decide

The judge is BLIND to the source run / model identity. The two extractions
are presented as "Extraction X" and "Extraction Y" in randomised order.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

JUDGE_MODEL = "claude-opus-4-7"  # latest Opus available 2026-05-08
ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Pricing (USD per 1M tokens, Opus pricing)
INPUT_PRICE_PER_M = 15.0
OUTPUT_PRICE_PER_M = 75.0


JUDGE_SYSTEM_PROMPT = """You are an expert in environmental epidemiology evaluating whether two extractions of effect-estimate data from the same scientific abstract are *materially the same* or *materially different* for the purpose of evidence synthesis.

You will see:
1. The source abstract.
2. Two extractions of an effect-estimate row, labelled "Extraction X" and "Extraction Y", presented in randomised order. The two extractions came from independent runs of an LLM-based extraction pipeline applied to the same abstract under documented-deterministic settings.

You must judge against THREE pre-registered criteria:

(a) DIRECTION: do both extractions report the same direction of effect (positive / null / negative)?
    - For ratio measures (RR, OR, HR): "positive" means >1.0, "negative" means <1.0, "null" means ~1.0 with CI overlapping 1.0.
    - For risk differences: "positive" means >0, "negative" means <0.

(b) MAGNITUDE: do the effect estimates agree within +/- 20% relative to their average?

(c) CI OVERLAP: if both 95% CIs are reported, do they share any range (overlap)?

Then issue ONE of three verdicts:
  - truly_contradictory: at least one of (a), (b), (c) materially fails
  - semantically_equivalent: all three criteria hold (textual differences only)
  - ambiguous: data missing or judgment uncertain

Return your judgment STRICTLY as JSON in this exact schema:

{
  "criterion_a_direction": "same" | "different" | "ambiguous",
  "criterion_b_magnitude": "same" | "different" | "ambiguous",
  "criterion_c_ci_overlap": "overlap" | "disjoint" | "ambiguous",
  "verdict": "truly_contradictory" | "semantically_equivalent" | "ambiguous",
  "rationale": "<one or two sentences justifying the verdict>"
}

Do NOT add commentary outside the JSON. Do NOT speculate beyond what the two extractions report.
"""


@dataclass
class JudgeResult:
    case_id: str
    verdict: str
    criterion_a_direction: str
    criterion_b_magnitude: str
    criterion_c_ci_overlap: str
    rationale: str
    randomization_seed: int
    presented_x_run_id: int
    presented_y_run_id: int
    raw_response: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    judge_model: str
    timestamp: float


def _build_user_prompt(
    abstract_text: str,
    run_a_text: str,
    run_b_text: str,
    randomize_seed: int,
) -> tuple[str, int, int]:
    """Build the user prompt with randomised X/Y assignment.

    Returns: (prompt_text, presented_x_idx, presented_y_idx)
    where idx 0 = run_a, 1 = run_b.
    """
    rng = random.Random(randomize_seed)
    if rng.random() < 0.5:
        x_text, y_text, x_idx, y_idx = run_a_text, run_b_text, 0, 1
    else:
        x_text, y_text, x_idx, y_idx = run_b_text, run_a_text, 1, 0

    prompt = f"""SOURCE ABSTRACT:
---
{abstract_text}
---

EXTRACTION X:
---
{x_text}
---

EXTRACTION Y:
---
{y_text}
---

Apply the three pre-registered criteria and return your verdict as JSON only."""
    return prompt, x_idx, y_idx


def _call_anthropic(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 600,
    timeout: int = 60,
) -> dict:
    # NOTE: claude-opus-4-7 deprecates the `temperature` parameter; we omit it.
    # The model defaults to deterministic-leaning sampling appropriate for evaluation.
    payload = {
        "model": JUDGE_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    req = urllib.request.Request(
        ANTHROPIC_API,
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        raise RuntimeError(f"Anthropic API HTTP {e.code}: {body[:500]}") from e


def _parse_judge_json(raw: str) -> dict:
    """Extract the JSON judgment from the model's response, robust to surrounding text."""
    # Try direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # Try fenced ```json ... ```
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try first {...} block
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {
        "verdict": "ambiguous",
        "criterion_a_direction": "ambiguous",
        "criterion_b_magnitude": "ambiguous",
        "criterion_c_ci_overlap": "ambiguous",
        "rationale": f"Could not parse judge JSON. Raw response: {raw[:200]}",
    }


def judge_case(
    case: dict,
    api_key: str,
    randomize_seed: int | None = None,
    dry_run: bool = False,
) -> JudgeResult:
    """Judge a single case. `case` is a dict from t3_judge_cases.json."""
    case_id = case["case_id"]
    if randomize_seed is None:
        # Deterministic per-case randomisation (reproducible)
        randomize_seed = int.from_bytes(
            hashlib.sha256(case_id.encode()).digest()[:4], "big"
        )

    abstract_text = case.get("abstract_text", "")[:4000]  # truncate very long abstracts
    run_a_text = _extraction_to_text(case["run_a"])
    run_b_text = _extraction_to_text(case["run_b"])

    user_prompt, x_idx, y_idx = _build_user_prompt(
        abstract_text, run_a_text, run_b_text, randomize_seed,
    )

    if dry_run:
        # Estimate tokens roughly (chars / 4)
        in_tokens = (len(JUDGE_SYSTEM_PROMPT) + len(user_prompt)) // 4
        out_tokens = 200
        cost = (in_tokens * INPUT_PRICE_PER_M + out_tokens * OUTPUT_PRICE_PER_M) / 1_000_000
        return JudgeResult(
            case_id=case_id,
            verdict="(dry-run)",
            criterion_a_direction="(dry-run)",
            criterion_b_magnitude="(dry-run)",
            criterion_c_ci_overlap="(dry-run)",
            rationale="(dry-run — no API call made)",
            randomization_seed=randomize_seed,
            presented_x_run_id=case["run_a"]["run_id"] if x_idx == 0 else case["run_b"]["run_id"],
            presented_y_run_id=case["run_b"]["run_id"] if y_idx == 1 else case["run_a"]["run_id"],
            raw_response="",
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=cost,
            judge_model=JUDGE_MODEL,
            timestamp=time.time(),
        )

    resp = _call_anthropic(api_key, JUDGE_SYSTEM_PROMPT, user_prompt)
    raw = resp["content"][0]["text"]
    parsed = _parse_judge_json(raw)

    usage = resp.get("usage", {})
    in_tokens = usage.get("input_tokens", 0)
    out_tokens = usage.get("output_tokens", 0)
    cost = (in_tokens * INPUT_PRICE_PER_M + out_tokens * OUTPUT_PRICE_PER_M) / 1_000_000

    return JudgeResult(
        case_id=case_id,
        verdict=parsed.get("verdict", "ambiguous"),
        criterion_a_direction=parsed.get("criterion_a_direction", "ambiguous"),
        criterion_b_magnitude=parsed.get("criterion_b_magnitude", "ambiguous"),
        criterion_c_ci_overlap=parsed.get("criterion_c_ci_overlap", "ambiguous"),
        rationale=parsed.get("rationale", ""),
        randomization_seed=randomize_seed,
        presented_x_run_id=case["run_a"]["run_id"] if x_idx == 0 else case["run_b"]["run_id"],
        presented_y_run_id=case["run_b"]["run_id"] if y_idx == 1 else case["run_a"]["run_id"],
        raw_response=raw,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        cost_usd=cost,
        judge_model=JUDGE_MODEL,
        timestamp=time.time(),
    )


def _extraction_to_text(run: dict) -> str:
    parts = []
    if run.get("effect_measure"):
        parts.append(f"effect_measure: {run['effect_measure']}")
    if run.get("effect_estimate") is not None:
        parts.append(f"effect_estimate: {run['effect_estimate']}")
    if run.get("ci_lower") is not None and run.get("ci_upper") is not None:
        parts.append(f"95% CI: [{run['ci_lower']}, {run['ci_upper']}]")
    if run.get("lag"):
        parts.append(f"lag: {run['lag']}")
    if run.get("outcome_specific"):
        parts.append(f"outcome: {run['outcome_specific']}")
    if run.get("exposure_increment"):
        parts.append(f"exposure_increment: {run['exposure_increment']}")
    return "\n".join(parts) if parts else "(no extraction recorded)"


def judge_all_cases(
    cases_file: Path,
    output_dir: Path,
    api_key: str | None = None,
    dry_run: bool = False,
    sleep_between_calls: float = 1.0,
) -> Path:
    """Run the judge over all cases and save results."""
    if not dry_run:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set and no key provided")

    with open(cases_file) as f:
        bundle = json.load(f)
    cases = bundle["cases"]

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[JudgeResult] = []
    total_cost = 0.0

    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Judging {case['case_id']}...")
        result = judge_case(case, api_key=api_key or "", dry_run=dry_run)
        results.append(result)
        total_cost += result.cost_usd
        # Save per-case Run Card
        per_case_path = output_dir / f"judge_{case['case_id'].replace('/', '_')}.json"
        with open(per_case_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        if not dry_run:
            time.sleep(sleep_between_calls)  # rate limit politeness

    # Save aggregate
    aggregate_path = output_dir / "t3_judge_results.json"
    with open(aggregate_path, "w") as f:
        json.dump({
            "n_cases": len(results),
            "total_cost_usd": total_cost,
            "judge_model": JUDGE_MODEL,
            "verdict_counts": _count_verdicts(results),
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nDone. Total cost: ${total_cost:.4f} USD.")
    print(f"Verdicts: {_count_verdicts(results)}")
    print(f"Aggregate saved to: {aggregate_path}")
    return aggregate_path


def _count_verdicts(results: list[JudgeResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return counts
