"""Per-call API cost estimation for the NatComms revision pipeline.

The pipeline operates under a hard $50 USD budget. Every API call is
priced before execution; the runner aborts (or skips a stack) when the
projected running total would exceed the cap.

Pricing (USD per 1 M tokens, validated 2026-05-08):

    GPT-4o                $2.50 / $10.00
    GPT-4-turbo           $10.00 / $30.00
    GPT-4 (gpt-4-0613)    $30.00 / $60.00
    Claude Sonnet 4.5     $3.00 / $15.00
    Claude Opus 4.7       $15.00 / $75.00
    Gemini 2.5 Pro        $1.25 / $5.00     (free tier limited)
    DeepSeek Chat         $0.27 / $1.10
    Together AI           $0.20 / $0.20     (approx; depends on model)
    Local (Ollama)        $0.00 / $0.00

Token counting:
    We estimate tokens by characters / 4 (GPT-style heuristic) when the
    ``tiktoken`` package is unavailable. The estimate is intentionally
    conservative — round-tripping through a real tokenizer would slow the
    dry-run loop and the result is only used for budget planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ─── Pricing table (USD per 1 M tokens) ──────────────────────────────────────

PRICING: dict[str, dict[str, float]] = {
    # OpenAI snapshots
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-0613": {"input": 30.00, "output": 60.00},
    # Anthropic
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-opus-4-7": {"input": 15.00, "output": 75.00},
    "claude-opus-4-7-20251201": {"input": 15.00, "output": 75.00},
    # Google
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2-5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    # DeepSeek
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # Together AI (approximate; varies per model)
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite": {"input": 0.10, "output": 0.10},
    "together-llama3": {"input": 0.20, "output": 0.20},
    # Local — free
    "llama3:8b": {"input": 0.0, "output": 0.0},
    "mistral:7b": {"input": 0.0, "output": 0.0},
    "gemma2:9b": {"input": 0.0, "output": 0.0},
}


# Friendly aliases used by the runner CLI.
ALIASES: dict[str, str] = {
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude-opus-4-7": "claude-opus-4-7-20251201",
    "gemini-2-5-pro": "gemini-2.5-pro",
    "llama3-8b-local": "llama3:8b",
    "mistral-7b-local": "mistral:7b",
    "gemma2-9b-local": "gemma2:9b",
}


@dataclass
class CostEstimate:
    stack: str
    n_calls: int
    input_tokens: int
    output_tokens: int
    input_usd: float
    output_usd: float
    notes: list[str] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return self.input_usd + self.output_usd

    def to_dict(self) -> dict:
        return {
            "stack": self.stack,
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_usd": round(self.input_usd, 4),
            "output_usd": round(self.output_usd, 4),
            "total_usd": round(self.total_usd, 4),
            "notes": self.notes,
        }


# ─── Token counting ───────────────────────────────────────────────────────────

def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate tokens. Uses ``tiktoken`` when available, else a 4 chars/token
    heuristic. Returns at least 1 for non-empty text."""
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.encoding_for_model(model or "gpt-4")  # type: ignore
        return len(enc.encode(text))
    except Exception:
        # 4 chars/token is the standard rule-of-thumb for English.
        return max(1, len(text) // 4)


# ─── Cost computation ─────────────────────────────────────────────────────────

def resolve_model(stack: str) -> str:
    """Map a CLI stack alias to the canonical pricing key."""
    return ALIASES.get(stack, stack)


def get_pricing(stack: str) -> dict[str, float]:
    """Return the {input, output} dict for ``stack``. Unknown -> zeros + warning."""
    canonical = resolve_model(stack)
    if canonical not in PRICING:
        return {"input": 0.0, "output": 0.0}
    return PRICING[canonical]


def estimate_call_cost(
    stack: str,
    input_text: str,
    max_output_tokens: int,
    n_reps: int = 1,
) -> CostEstimate:
    """Cost of running a single prompt N times against ``stack``.

    ``max_output_tokens`` is treated as the *budgeted* output length; real
    completions may be shorter, so this is a conservative upper bound.
    """
    canonical = resolve_model(stack)
    pricing = get_pricing(stack)
    input_tokens = count_tokens(input_text, canonical) * n_reps
    output_tokens = max_output_tokens * n_reps
    input_usd = input_tokens * pricing["input"] / 1_000_000
    output_usd = output_tokens * pricing["output"] / 1_000_000
    notes = []
    if pricing["input"] == 0.0 and pricing["output"] == 0.0:
        if canonical in PRICING:
            notes.append("local stack — no API cost")
        else:
            notes.append("UNKNOWN STACK — zero pricing assumed; verify before run")
    return CostEstimate(
        stack=canonical,
        n_calls=n_reps,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_usd=input_usd,
        output_usd=output_usd,
        notes=notes,
    )


def estimate_task_cost(
    stack: str,
    prompts: list[str],
    max_output_tokens: int,
    n_reps: int,
    n_turns: int = 1,
) -> CostEstimate:
    """Aggregate the cost of running ``n_reps`` repetitions of every prompt
    in ``prompts`` against ``stack``.

    For multi-turn tasks, set ``n_turns > 1``: input cost grows roughly
    quadratically with turns because each turn re-sends prior context.
    """
    pricing = get_pricing(stack)
    canonical = resolve_model(stack)
    total_input = 0
    total_output = 0
    notes = []
    for prompt in prompts:
        prompt_tokens = count_tokens(prompt, canonical)
        if n_turns > 1:
            # Approximate: turn t re-sends (prompt + sum of previous outputs).
            # Conservative cap: total = prompt * n_turns + max_out * n_turns * (n_turns-1)/2.
            per_call_in = prompt_tokens * n_turns + max_output_tokens * n_turns * (
                n_turns - 1
            ) // 2
        else:
            per_call_in = prompt_tokens
        total_input += per_call_in * n_reps
        total_output += max_output_tokens * n_turns * n_reps

    input_usd = total_input * pricing["input"] / 1_000_000
    output_usd = total_output * pricing["output"] / 1_000_000
    if pricing["input"] == 0.0 and pricing["output"] == 0.0:
        if canonical in PRICING:
            notes.append("local stack — no API cost")
        else:
            notes.append("UNKNOWN STACK — zero pricing assumed")

    return CostEstimate(
        stack=canonical,
        n_calls=len(prompts) * n_reps,
        input_tokens=total_input,
        output_tokens=total_output,
        input_usd=input_usd,
        output_usd=output_usd,
        notes=notes,
    )


# ─── Budget enforcement ───────────────────────────────────────────────────────

class BudgetGuard:
    """Tracks running spend and aborts when ``cap`` would be exceeded.

    Usage::

        guard = BudgetGuard(cap_usd=50.00)
        if not guard.would_allow(estimated_cost):
            log_skip(...)
            continue
        guard.charge(actual_cost)
    """

    def __init__(self, cap_usd: float = 50.00):
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.charges: list[dict] = []

    def would_allow(self, additional_usd: float) -> bool:
        return (self.spent_usd + additional_usd) <= self.cap_usd

    def charge(self, additional_usd: float, label: str = "") -> float:
        self.spent_usd += additional_usd
        self.charges.append(
            {"label": label, "usd": round(additional_usd, 6),
             "total_after": round(self.spent_usd, 6)}
        )
        return self.spent_usd

    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    def summary(self) -> dict:
        return {
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent_usd, 4),
            "remaining_usd": round(self.remaining(), 4),
            "n_charges": len(self.charges),
        }
