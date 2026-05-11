"""
GPT-4o LLM-as-judge — second judge for T3 PM2.5 triangulation (R2).

Mirror implementation of Claude Opus judge (src/tasks/llm_judge.py) but
calling gpt-4o via OpenAI Chat Completions. Used to compute inter-judge
Cohen's kappa for the R3.6 response.

Pre-registered judgment criteria (R3 item 6 of NatComms major revision):
  (a) Same direction of the reported effect (positive / null / negative)?
  (b) Same magnitude within +/- 20%?
  (c) Same 95% CI overlap (do the intervals share any range)?

Verdicts: truly_contradictory | semantically_equivalent | ambiguous

The judge is BLIND to source run / model identity. The two extractions are
presented as "Extraction X" and "Extraction Y" in randomised order.
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

# Re-use system prompt + helpers from the Claude judge
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.tasks.llm_judge import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT, _build_user_prompt, _parse_judge_json, _extraction_to_text,
)

JUDGE_MODEL = "gpt-4o"
OPENAI_API = "https://api.openai.com/v1/chat/completions"

# Pricing (USD per 1M tokens, gpt-4o pricing as of 2026)
INPUT_PRICE_PER_M = 2.50
OUTPUT_PRICE_PER_M = 10.00


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


def _call_openai(api_key: str, system_prompt: str, user_prompt: str,
                 max_tokens: int = 600, timeout: int = 60) -> dict:
    payload = {
        "model": JUDGE_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "seed": 42,
    }
    req = urllib.request.Request(
        OPENAI_API,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        raise RuntimeError(f"OpenAI API HTTP {e.code}: {body[:500]}") from e


def judge_case(case: dict, api_key: str, randomize_seed: int | None = None,
               dry_run: bool = False) -> JudgeResult:
    case_id = case["case_id"]
    if randomize_seed is None:
        randomize_seed = int.from_bytes(
            hashlib.sha256(case_id.encode()).digest()[:4], "big"
        )

    abstract_text = case.get("abstract_text", "")[:4000]
    run_a_text = _extraction_to_text(case["run_a"])
    run_b_text = _extraction_to_text(case["run_b"])

    user_prompt, x_idx, y_idx = _build_user_prompt(
        abstract_text, run_a_text, run_b_text, randomize_seed,
    )

    if dry_run:
        in_tokens = (len(JUDGE_SYSTEM_PROMPT) + len(user_prompt)) // 4
        out_tokens = 200
        cost = (in_tokens * INPUT_PRICE_PER_M + out_tokens * OUTPUT_PRICE_PER_M) / 1_000_000
        return JudgeResult(
            case_id=case_id, verdict="(dry-run)",
            criterion_a_direction="(dry-run)", criterion_b_magnitude="(dry-run)",
            criterion_c_ci_overlap="(dry-run)", rationale="(dry-run)",
            randomization_seed=randomize_seed,
            presented_x_run_id=case["run_a"]["run_id"] if x_idx == 0 else case["run_b"]["run_id"],
            presented_y_run_id=case["run_b"]["run_id"] if y_idx == 1 else case["run_a"]["run_id"],
            raw_response="", input_tokens=in_tokens, output_tokens=out_tokens,
            cost_usd=cost, judge_model=JUDGE_MODEL, timestamp=time.time(),
        )

    resp = _call_openai(api_key, JUDGE_SYSTEM_PROMPT, user_prompt)
    raw = resp["choices"][0]["message"]["content"]
    parsed = _parse_judge_json(raw)

    usage = resp.get("usage", {})
    in_tokens = usage.get("prompt_tokens", 0)
    out_tokens = usage.get("completion_tokens", 0)
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


def judge_all_cases(cases_file: Path, output_dir: Path, api_key: str | None = None,
                    dry_run: bool = False, sleep_between_calls: float = 0.5) -> Path:
    if not dry_run:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    with open(cases_file) as f: bundle = json.load(f)
    cases = bundle["cases"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[JudgeResult] = []
    total_cost = 0.0

    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] [gpt-4o] Judging {case['case_id']}...")
        result = judge_case(case, api_key=api_key or "", dry_run=dry_run)
        results.append(result)
        total_cost += result.cost_usd
        per_case_path = output_dir / f"judge_gpt4o_{case['case_id'].replace('/', '_')}.json"
        with open(per_case_path, "w") as f: json.dump(asdict(result), f, indent=2)
        if not dry_run: time.sleep(sleep_between_calls)

    aggregate_path = output_dir / "t3_judge_gpt4o_results.json"
    counts: dict[str, int] = {}
    for r in results: counts[r.verdict] = counts.get(r.verdict, 0) + 1
    with open(aggregate_path, "w") as f:
        json.dump({
            "n_cases": len(results),
            "total_cost_usd": total_cost,
            "judge_model": JUDGE_MODEL,
            "verdict_counts": counts,
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nDone. Total cost: ${total_cost:.4f} USD.")
    print(f"Verdicts: {counts}")
    print(f"Aggregate saved to: {aggregate_path}")
    return aggregate_path
