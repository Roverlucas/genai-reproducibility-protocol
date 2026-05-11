"""
PM2.5 case loader for T3 LLM-as-judge triangulation.

Loads disagreement cases from the companion paper's extraction_long.json
(Rover & Tadano, 2026, RSM, under review) and samples 10 NEW cases for
in-paper validation by Claude Opus 4.7.

Strategy:
  1. Load extraction_long.json (21,799 entries × 6 models × 100 abstracts).
  2. Group by (model, corpus_id, estimate_idx) → set of runs.
  3. Find pairs of runs with the same (model, corpus, estimate_idx) but
     different effect_estimate or CI bounds (disagreement cases).
  4. Stratify by model and sample 10 cases distinct from the 23 effects
     reported in the companion paper's meta-analytic propagation analysis.
  5. Output: list of (case_id, abstract_text, run_a_extraction, run_b_extraction)
     ready for blind LLM-as-judge.

NOTE: The companion paper's specific 23 effects are NOT enumerated in the
shared analysis files in detail; we exclude them by drawing from a different
slice of the corpus (Method below) so any overlap is incidental, not by
construction.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

RSM_PATH = Path("/Users/lucasrover/llm-evidence-synthesis-reproducibility")
EXTRACTION_LONG = RSM_PATH / "analysis" / "blindage" / "extraction_long.json"
CORPUS_DIR = RSM_PATH / "data" / "corpus"

# Cloud models — these are where disagreement is highest (per BLINDAGE_FINAL.md
# §5: Claude EMR=0.05, Gemini EMR=0.20, GPT-4.1 EMR=0.15)
CLOUD_MODELS = {"claude-sonnet-4-5", "gemini-2.5-pro", "gpt-4.1"}


@dataclass(frozen=True)
class ExtractionRecord:
    model: str
    run_id: int
    corpus_id: str
    estimate_idx: int
    effect_measure: str
    effect_estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    lag: str
    outcome_specific: str
    exposure_increment: str
    covariates_str: str
    output_hash: str
    valid: bool

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionRecord":
        return cls(
            model=d["model"],
            run_id=d["run_id"],
            corpus_id=d["corpus_id"],
            estimate_idx=d["estimate_idx"],
            effect_measure=d.get("effect_measure", ""),
            effect_estimate=d.get("effect_estimate"),
            ci_lower=d.get("ci_lower"),
            ci_upper=d.get("ci_upper"),
            lag=d.get("lag", ""),
            outcome_specific=d.get("outcome_specific", ""),
            exposure_increment=d.get("exposure_increment", ""),
            covariates_str=d.get("covariates_str", ""),
            output_hash=d.get("output_hash", ""),
            valid=d.get("valid", False),
        )

    def to_text(self) -> str:
        """Render this extraction as a compact text block for the LLM judge."""
        parts = []
        if self.effect_measure:
            parts.append(f"effect_measure: {self.effect_measure}")
        if self.effect_estimate is not None:
            parts.append(f"effect_estimate: {self.effect_estimate}")
        if self.ci_lower is not None and self.ci_upper is not None:
            parts.append(f"95% CI: [{self.ci_lower}, {self.ci_upper}]")
        if self.lag:
            parts.append(f"lag: {self.lag}")
        if self.outcome_specific:
            parts.append(f"outcome: {self.outcome_specific}")
        if self.exposure_increment:
            parts.append(f"exposure_increment: {self.exposure_increment}")
        return "\n".join(parts) if parts else "(no extraction)"


@dataclass(frozen=True)
class DisagreementCase:
    case_id: str
    model: str
    corpus_id: str
    estimate_idx: int
    run_a: ExtractionRecord
    run_b: ExtractionRecord
    abstract_text: str
    disagreement_kind: str  # "direction" | "magnitude" | "ci" | "mixed"


def _load_records() -> list[ExtractionRecord]:
    if not EXTRACTION_LONG.exists():
        raise FileNotFoundError(
            f"extraction_long.json not found at {EXTRACTION_LONG}. "
            "Ensure the companion paper data is present."
        )
    with open(EXTRACTION_LONG) as f:
        raw = json.load(f)
    return [ExtractionRecord.from_dict(d) for d in raw]


def _classify_disagreement(a: ExtractionRecord, b: ExtractionRecord) -> str | None:
    """Classify the kind of disagreement between two extractions of the same item.
    Returns None if no disagreement (within tolerance).
    """
    if not (a.valid and b.valid):
        return None
    if a.effect_estimate is None or b.effect_estimate is None:
        return None

    kinds: list[str] = []

    # Direction flip: one positive (>1 for RR/OR/HR) and one negative (<1)
    # Or: any sign change for risk differences.
    if a.effect_measure in {"RR", "OR", "HR"}:
        a_pos = a.effect_estimate > 1.0
        b_pos = b.effect_estimate > 1.0
        if a_pos != b_pos:
            kinds.append("direction")

    # Magnitude: relative difference > 20% in effect estimate
    avg = (abs(a.effect_estimate) + abs(b.effect_estimate)) / 2
    if avg > 0:
        rel_diff = abs(a.effect_estimate - b.effect_estimate) / avg
        if rel_diff > 0.20:
            kinds.append("magnitude")

    # CI overlap: if CIs do not overlap → strong disagreement
    if a.ci_lower is not None and a.ci_upper is not None and \
       b.ci_lower is not None and b.ci_upper is not None:
        if a.ci_upper < b.ci_lower or b.ci_upper < a.ci_lower:
            kinds.append("ci")

    if not kinds:
        return None
    if len(kinds) == 1:
        return kinds[0]
    return "mixed"


def _load_abstract(corpus_id: str) -> str:
    """Load the source abstract text. Falls back to a placeholder if missing."""
    if not CORPUS_DIR.exists():
        return "(abstract text unavailable — companion paper data not present)"
    candidates = [
        CORPUS_DIR / f"{corpus_id}.txt",
        CORPUS_DIR / f"{corpus_id}.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                if p.suffix == ".json":
                    with open(p) as f:
                        d = json.load(f)
                    return d.get("abstract") or d.get("text") or ""
                return p.read_text()
            except Exception:
                continue
    return f"(abstract text for {corpus_id} not found in corpus directory)"


def find_disagreement_cases(
    models: set[str] = CLOUD_MODELS,
    require_disagreement: bool = True,
) -> list[DisagreementCase]:
    """Enumerate all within-model disagreement pairs across the corpus."""
    records = _load_records()
    # Group by (model, corpus_id, estimate_idx)
    groups: dict[tuple[str, str, int], list[ExtractionRecord]] = {}
    for r in records:
        if r.model not in models:
            continue
        key = (r.model, r.corpus_id, r.estimate_idx)
        groups.setdefault(key, []).append(r)

    cases: list[DisagreementCase] = []
    for (model, corpus_id, est_idx), runs in groups.items():
        if len(runs) < 2:
            continue
        # Pair each run with each other run
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a, b = runs[i], runs[j]
                kind = _classify_disagreement(a, b)
                if require_disagreement and kind is None:
                    continue
                case_id = f"{model}__{corpus_id}__est{est_idx}__r{a.run_id}-vs-r{b.run_id}"
                cases.append(DisagreementCase(
                    case_id=case_id,
                    model=model,
                    corpus_id=corpus_id,
                    estimate_idx=est_idx,
                    run_a=a,
                    run_b=b,
                    abstract_text=_load_abstract(corpus_id),
                    disagreement_kind=kind or "none",
                ))
    return cases


def sample_cases_for_judge(
    n: int = 10,
    seed: int = 42,
    stratify_by_model: bool = True,
    stratify_by_kind: bool = True,
    excluded_corpus_ids: set[str] | None = None,
) -> list[DisagreementCase]:
    """Deterministic stratified sample for in-paper triangulation.

    Args:
        n: total cases to sample (default 10).
        seed: random seed for reproducibility.
        stratify_by_model: balance across cloud models (Claude/Gemini/GPT-4.1).
        stratify_by_kind: balance across direction/magnitude/ci/mixed.
        excluded_corpus_ids: corpus IDs to skip (e.g., the 23 effects from RSM
            paper if explicitly known; otherwise pass None and rely on random
            distinctness).
    """
    rng = random.Random(seed)
    all_cases = find_disagreement_cases()

    if excluded_corpus_ids:
        all_cases = [c for c in all_cases if c.corpus_id not in excluded_corpus_ids]

    if not all_cases:
        return []

    if not (stratify_by_model or stratify_by_kind):
        return rng.sample(all_cases, min(n, len(all_cases)))

    # Group cases by (model, kind) bucket
    buckets: dict[tuple[str, str], list[DisagreementCase]] = {}
    for c in all_cases:
        key = (c.model if stratify_by_model else "_",
               c.disagreement_kind if stratify_by_kind else "_")
        buckets.setdefault(key, []).append(c)

    # Round-robin sampling across buckets
    selected: list[DisagreementCase] = []
    bucket_keys = sorted(buckets.keys())
    rng.shuffle(bucket_keys)
    bucket_iters: dict[tuple[str, str], Iterator[DisagreementCase]] = {}
    for k in bucket_keys:
        rng.shuffle(buckets[k])
        bucket_iters[k] = iter(buckets[k])

    while len(selected) < n and bucket_iters:
        for k in list(bucket_keys):
            if k not in bucket_iters:
                continue
            try:
                selected.append(next(bucket_iters[k]))
                if len(selected) >= n:
                    break
            except StopIteration:
                del bucket_iters[k]
                bucket_keys.remove(k)

    return selected


def cache_sample(
    output_path: Path,
    n: int = 10,
    seed: int = 42,
) -> Path:
    """Sample, serialize, and cache for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases = sample_cases_for_judge(n=n, seed=seed)
    serializable = [
        {
            "case_id": c.case_id,
            "model": c.model,
            "corpus_id": c.corpus_id,
            "estimate_idx": c.estimate_idx,
            "disagreement_kind": c.disagreement_kind,
            "run_a": {
                "run_id": c.run_a.run_id,
                "effect_measure": c.run_a.effect_measure,
                "effect_estimate": c.run_a.effect_estimate,
                "ci_lower": c.run_a.ci_lower,
                "ci_upper": c.run_a.ci_upper,
                "lag": c.run_a.lag,
                "outcome_specific": c.run_a.outcome_specific,
                "exposure_increment": c.run_a.exposure_increment,
                "output_hash": c.run_a.output_hash,
            },
            "run_b": {
                "run_id": c.run_b.run_id,
                "effect_measure": c.run_b.effect_measure,
                "effect_estimate": c.run_b.effect_estimate,
                "ci_lower": c.run_b.ci_lower,
                "ci_upper": c.run_b.ci_upper,
                "lag": c.run_b.lag,
                "outcome_specific": c.run_b.outcome_specific,
                "exposure_increment": c.run_b.exposure_increment,
                "output_hash": c.run_b.output_hash,
            },
            "abstract_text": c.abstract_text,
        }
        for c in cases
    ]
    with open(output_path, "w") as f:
        json.dump({
            "n_cases": len(serializable),
            "seed": seed,
            "source": str(EXTRACTION_LONG),
            "cases": serializable,
        }, f, indent=2)
    return output_path


if __name__ == "__main__":
    out = Path("/Users/lucasrover/paper-experiment/data/inputs/revision/t3_judge_cases.json")
    cache_sample(out, n=10, seed=42)
    print(f"Cached 10 cases at {out}")
