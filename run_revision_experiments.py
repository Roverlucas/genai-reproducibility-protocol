#!/usr/bin/env python3
"""Unified, resumable, checkpointed experiment runner for the NatComms revision.

This script consolidates the four new experiment families requested by the
editor and reviewers:

    1. ``humaneval``           — code generation, pass@1 metric.
    2. ``gsm8k``               — math reasoning, exact-match accuracy.
    3. ``pubmed_pm25``         — non-AI/ML domain, structured extraction (T14).
    4. ``multiturn_extension`` — extend Task 4 multi-turn refinement to
                                 GPT-4 + DeepSeek (Claude/Gemini already done).

Design principles:
    * Resume from a JSON checkpoint after every run.
    * Estimate API cost before each call and abort when the global $50
      budget cap would be exceeded.
    * Reuse the existing protocol modules (``RunLogger``, ``RunCard``,
      PROV) so every revision run is auditable in the same way as the
      original experiments.
    * Be dry-run-first: no API calls happen until ``--execute`` is set.
    * Be graceful: API errors log and continue rather than crash.

USAGE
-----

Dry run (NO API CALLS), full revision scope:

    python run_revision_experiments.py --task all --stack all \\
        --condition all --n-reps 5 --dry-run

Execute one task on one stack (after Lucas's go-ahead):

    python run_revision_experiments.py --task humaneval \\
        --stack claude-sonnet-4-5 --condition C1 --n-problems 30 \\
        --n-reps 5 --execute --resume

Execute everything that fits the budget:

    python run_revision_experiments.py --task all --stack all \\
        --condition C1 --n-reps 5 --budget-usd 50 --execute --resume

CLI flags
---------

    --task       {humaneval,gsm8k,pubmed_pm25,multiturn_extension,all}
    --stack      Stack ID or 'all'. See ``STACK_REGISTRY``.
    --condition  C1 | C2 | C3 | all   (matches the paper's 3 conditions).
    --n-problems Items per task (default: 30 / 30 / 10).
    --n-reps     Repetitions per item (default: 5).
    --output-dir Default ``outputs/revision/runs``.
    --resume     Skip runs whose JSON already exists (default ON).
    --dry-run    Print cost estimate + run plan only.
    --execute    Required to actually call APIs.
    --budget-usd Hard cap (default 50.0).
    --checkpoint Path to checkpoint JSON
                 (default ``outputs/revision/checkpoints/checkpoint.json``).
    --checkpoint-every  Save checkpoint after every N runs (default 5).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.protocol.logger import RunLogger
from src.protocol.run_card import RunCard
from src.protocol.hasher import hash_text
from src.experiments.config import (
    RESEARCHER_ID,
    AFFILIATION,
    SEEDS,
    TEMPERATURES,
    EXTRACTION_PROMPT,
)
from src.cost_estimator import (
    BudgetGuard,
    estimate_call_cost,
    estimate_task_cost,
    resolve_model,
)
from src.tasks import humaneval_loader, gsm8k_loader, pubmed_loader
from src.tasks import gsm8k_extractor


OUTPUT_DIR_DEFAULT = PROJECT_ROOT / "outputs" / "revision" / "runs"
CHECKPOINT_DEFAULT = (
    PROJECT_ROOT / "outputs" / "revision" / "checkpoints" / "checkpoint.json"
)


# ─── Stack registry ───────────────────────────────────────────────────────────

STACK_REGISTRY: dict[str, dict] = {
    "gpt-4": {
        "runner": "gpt4_runner", "model": "gpt-4", "kind": "api",
    },
    "gpt-4-turbo": {
        "runner": "gpt4_runner", "model": "gpt-4-turbo", "kind": "api",
    },
    "gpt-4o": {
        "runner": "gpt4_runner", "model": "gpt-4o-2024-11-20", "kind": "api",
    },
    "claude-sonnet-4-5": {
        "runner": "claude_runner",
        "model": "claude-sonnet-4-5-20250929",
        "kind": "api",
    },
    "claude-opus-4-7": {
        "runner": "claude_runner",
        "model": "claude-opus-4-7-20251201",
        "kind": "api",
    },
    "gemini-2-5-pro": {
        "runner": "gemini_runner", "model": "gemini-2.5-pro", "kind": "api",
    },
    "deepseek-chat": {
        "runner": "deepseek_runner", "model": "deepseek-chat", "kind": "api",
    },
    "llama3-8b-local": {
        "runner": "llama_runner", "model": "llama3:8b", "kind": "local",
    },
    "mistral-7b-local": {
        "runner": "llama_runner", "model": "mistral:7b", "kind": "local",
    },
    "gemma2-9b-local": {
        "runner": "llama_runner", "model": "gemma2:9b", "kind": "local",
    },
    "together-llama3": {
        "runner": "together_runner",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "kind": "api",
    },
}


def _load_runner(name: str):
    """Lazily import a runner module by name."""
    from importlib import import_module
    return import_module(f"src.models.{name}")


# ─── Task prompts ─────────────────────────────────────────────────────────────

HUMANEVAL_SYSTEM = (
    "You are an expert Python programmer. Complete the function below. "
    "Return ONLY the function body (no explanation, no markdown fences, "
    "no surrounding text). Preserve the function signature."
)

GSM8K_SYSTEM = (
    "You are a careful mathematics tutor. Solve grade-school word "
    "problems by reasoning step by step, then state the final answer "
    "after '####' as a single integer."
)

# T14 reuses the original paper's structured-extraction prompt.
PUBMED_SYSTEM = (
    "You are a structured information extraction assistant. Extract "
    "fields from the abstract using ONLY information in the abstract."
)

# Multi-turn refinement (mirrors run_claude_multiturn.py for cross-stack parity)
REFINEMENT_SYSTEM_PROMPT = (
    "You are a structured information extraction assistant. "
    "You extract information from scientific abstracts into JSON format."
)
REFINEMENT_TURN1_PROMPT = (
    "Read the following scientific abstract and extract the information "
    "into the exact JSON format below. Use only information explicitly "
    "stated in the abstract. If a field is not mentioned, use null.\n\n"
    "Output format (JSON only, no explanation):\n"
    "{\n"
    '  "objective": "string",\n'
    '  "method": "string",\n'
    '  "key_result": "string",\n'
    '  "model_or_system": "string",\n'
    '  "benchmark": "string"\n'
    "}"
)
REFINEMENT_TURN2_PROMPT = (
    "Review your extraction above. Check each field carefully against "
    "the original abstract. If any field is incomplete, imprecise, or "
    "missing quantitative details that are in the abstract, correct it. "
    "Output only the corrected JSON, no explanation."
)
REFINEMENT_TURN3_PROMPT = (
    "Now produce the final verified extraction. Ensure all fields contain "
    "the most precise information from the abstract. If quantitative "
    "results are mentioned (percentages, scores, metrics), they must "
    "appear in key_result. Output only the final JSON."
)


# ─── Conditions ───────────────────────────────────────────────────────────────

# (condition_id, temperature, seed_strategy)
CONDITIONS = {
    "C1": ("C1_fixed_seed", 0.0, "fixed"),       # seed=42 fixed
    "C2": ("C2_var_seed", 0.0, "variable"),      # cycle SEEDS
    "C3": ("C3_var_temp", None, "temp_sweep"),   # cycle TEMPERATURES
}


# ─── Checkpoint ───────────────────────────────────────────────────────────────

class Checkpoint:
    """JSON-backed registry of completed run IDs and budget state."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {
            "completed_run_ids": [],
            "errors": [],
            "spent_usd": 0.0,
            "last_updated": None,
        }
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass

    def is_complete(self, run_id: str) -> bool:
        return run_id in set(self.data.get("completed_run_ids", []))

    def mark_complete(self, run_id: str, cost_usd: float = 0.0):
        self.data.setdefault("completed_run_ids", []).append(run_id)
        self.data["spent_usd"] = round(
            self.data.get("spent_usd", 0.0) + cost_usd, 6
        )
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()

    def log_error(self, run_id: str, message: str):
        self.data.setdefault("errors", []).append(
            {"run_id": run_id, "error": message,
             "timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


# ─── Run ID convention ────────────────────────────────────────────────────────

def _safe(s: str) -> str:
    return s.replace(":", "_").replace("/", "_").replace(" ", "_").replace(".", "_")


def make_run_id(
    stack: str, task: str, item_id: str, condition: str, rep: int,
    suffix: str = "",
) -> str:
    base = f"rev_{_safe(stack)}_{task}_{_safe(item_id)}_{condition}_rep{rep}"
    if suffix:
        base += f"_{_safe(suffix)}"
    return base


# ─── Dispatch helpers ─────────────────────────────────────────────────────────

def _build_inference_kwargs(
    stack_cfg: dict, condition_id: str, rep: int, max_tokens: int,
) -> dict:
    """Map condition + rep to (temperature, seed) for the active stack."""
    label, temp, strategy = CONDITIONS[condition_id]
    temperature = temp
    seed: Optional[int] = None
    if strategy == "fixed":
        seed = SEEDS[0]  # 42
    elif strategy == "variable":
        seed = SEEDS[rep % len(SEEDS)]
    elif strategy == "temp_sweep":
        # Cycle through TEMPERATURES; seed varies by rep for diversity.
        temperature = TEMPERATURES[rep % len(TEMPERATURES)]
        seed = SEEDS[rep % len(SEEDS)]

    kwargs = {"temperature": temperature, "max_tokens": max_tokens}
    # Anthropic's seed support is logged-only; gemini accepts it; gpt4 + deepseek accept it.
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs


def _call_runner_inference(
    runner_module, stack_cfg: dict, prompt: str, input_text: str,
    condition_id: str, rep: int, max_tokens: int, system_prompt: str = "",
) -> dict:
    """Invoke the runner's ``run_inference`` with the appropriate signature.

    Different runners accept slightly different kwargs (e.g., Anthropic
    has no ``seed``; Ollama uses ``top_k``). We probe the supported
    parameters before calling.
    """
    base_kwargs = _build_inference_kwargs(stack_cfg, condition_id, rep, max_tokens)
    fn = runner_module.run_inference
    import inspect
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    call_kwargs = {"prompt": prompt, "input_text": input_text,
                   "model": stack_cfg["model"]}
    for k, v in base_kwargs.items():
        if k in accepted:
            call_kwargs[k] = v
    if system_prompt and "system_prompt" in accepted:
        call_kwargs["system_prompt"] = system_prompt

    # For Anthropic, seed cannot be passed — it must be logged separately.
    if stack_cfg["runner"] == "claude_runner":
        call_kwargs.pop("seed", None)
    return fn(**call_kwargs)


# ─── Run plan items ───────────────────────────────────────────────────────────

class RunItem:
    """A single planned run: stack × task × item × condition × rep."""

    __slots__ = (
        "stack", "task", "item", "condition", "rep", "prompt", "system",
        "max_tokens", "n_turns", "task_category", "interaction_regime",
    )

    def __init__(self, stack, task, item, condition, rep, prompt, system,
                 max_tokens, n_turns, task_category, interaction_regime):
        self.stack = stack
        self.task = task
        self.item = item
        self.condition = condition
        self.rep = rep
        self.prompt = prompt
        self.system = system
        self.max_tokens = max_tokens
        self.n_turns = n_turns
        self.task_category = task_category
        self.interaction_regime = interaction_regime

    @property
    def run_id(self) -> str:
        return make_run_id(
            self.stack, self.task, self.item.get("id", "unk"),
            self.condition, self.rep,
        )


# ─── Plan builders (one per task family) ──────────────────────────────────────

def plan_humaneval(stack: str, condition: str, n_problems: int,
                   n_reps: int) -> list[RunItem]:
    problems = humaneval_loader.load_humaneval()
    sampled = humaneval_loader.stratified_sample(problems, n=n_problems)
    items = []
    for p in sampled:
        item = {
            "id": p["task_id"].replace("/", "_"),
            "task_id_orig": p["task_id"],
            "prompt": p["prompt"],
            "test": p["test"],
            "entry_point": p["entry_point"],
        }
        for rep in range(n_reps):
            items.append(RunItem(
                stack=stack, task="humaneval", item=item, condition=condition,
                rep=rep, prompt=p["prompt"], system=HUMANEVAL_SYSTEM,
                max_tokens=512, n_turns=1, task_category="code_generation",
                interaction_regime="single-turn",
            ))
    return items


def plan_gsm8k(stack: str, condition: str, n_problems: int,
               n_reps: int) -> list[RunItem]:
    problems = gsm8k_loader.load_gsm8k()
    sampled = gsm8k_loader.attach_id(
        gsm8k_loader.sample_problems(problems, n=n_problems)
    )
    items = []
    for p in sampled:
        item = {
            "id": p["id"],
            "question": p["question"],
            "gold_answer": p["answer"],
        }
        prompt = gsm8k_loader.format_prompt(p)
        for rep in range(n_reps):
            items.append(RunItem(
                stack=stack, task="gsm8k", item=item, condition=condition,
                rep=rep, prompt=prompt, system=GSM8K_SYSTEM,
                max_tokens=512, n_turns=1, task_category="math_reasoning",
                interaction_regime="single-turn",
            ))
    return items


def plan_pubmed(stack: str, condition: str, n_problems: int,
                n_reps: int) -> list[RunItem]:
    abstracts = pubmed_loader.load_pubmed_pm25()
    abstracts = abstracts[:n_problems] if n_problems else abstracts
    items = []
    for a in abstracts:
        for rep in range(n_reps):
            items.append(RunItem(
                stack=stack, task="pubmed_pm25", item=a,
                condition=condition, rep=rep, prompt=EXTRACTION_PROMPT,
                system=PUBMED_SYSTEM, max_tokens=512, n_turns=1,
                task_category="structured_extraction_health",
                interaction_regime="single-turn",
            ))
    return items


def plan_multiturn_extension(stack: str, condition: str, n_problems: int,
                             n_reps: int) -> list[RunItem]:
    """3-turn refinement on the original 10 abstracts.

    Per REVISION_PLAN T4, extend Claude/Gemini's existing multi-turn runs
    to GPT-4 + DeepSeek. We use the same abstracts.json corpus.
    """
    with open(PROJECT_ROOT / "data" / "inputs" / "abstracts.json") as f:
        abstracts = json.load(f)["abstracts"][:n_problems]
    items = []
    full_prompt = (
        f"[SYSTEM] {REFINEMENT_SYSTEM_PROMPT}\n"
        f"[TURN1] {REFINEMENT_TURN1_PROMPT}\n"
        f"[TURN2] {REFINEMENT_TURN2_PROMPT}\n"
        f"[TURN3] {REFINEMENT_TURN3_PROMPT}"
    )
    for a in abstracts:
        for rep in range(n_reps):
            items.append(RunItem(
                stack=stack, task="multiturn_refinement", item=a,
                condition=condition, rep=rep, prompt=full_prompt,
                system=REFINEMENT_SYSTEM_PROMPT, max_tokens=1024,
                n_turns=3, task_category="iterative_structured_extraction",
                interaction_regime="multi-turn",
            ))
    return items


PLAN_BUILDERS: dict[str, Callable] = {
    "humaneval": plan_humaneval,
    "gsm8k": plan_gsm8k,
    "pubmed_pm25": plan_pubmed,
    "multiturn_extension": plan_multiturn_extension,
}

DEFAULT_N_PROBLEMS = {
    "humaneval": 30,
    "gsm8k": 30,
    "pubmed_pm25": 10,
    "multiturn_extension": 10,
}


# ─── Execution helpers ────────────────────────────────────────────────────────

def _execute_single_turn(runner_module, stack_cfg: dict, item: RunItem) -> dict:
    """Run a single-turn inference and return the result dict."""
    return _call_runner_inference(
        runner_module, stack_cfg,
        prompt=item.prompt, input_text=item.item.get("text",
            item.item.get("question", "")),
        condition_id=item.condition, rep=item.rep,
        max_tokens=item.max_tokens, system_prompt=item.system,
    )


def _execute_multiturn(runner_module, stack_cfg: dict, item: RunItem) -> dict:
    """Run the 3-turn refinement protocol via the runner's chat API.

    Currently supports gpt4_runner and deepseek_runner via OpenAI-style
    messages; falls back to sequential single-turn calls (lossy: no shared
    state) if the runner exposes no chat helper.
    """
    abstract_text = item.item.get("text", "")
    base_kwargs = _build_inference_kwargs(
        stack_cfg, item.condition, item.rep, item.max_tokens
    )
    temperature = base_kwargs.get("temperature", 0.0)
    seed = base_kwargs.get("seed")

    if stack_cfg["runner"] == "gpt4_runner":
        from openai import OpenAI
        from src.models.gpt4_runner import get_client
        client = get_client()
        messages = [
            {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
            {"role": "user",
             "content": f"{REFINEMENT_TURN1_PROMPT}\n\n{abstract_text}"},
        ]
        prompts = [REFINEMENT_TURN2_PROMPT, REFINEMENT_TURN3_PROMPT]
        turn_outputs = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
        finish_reason = ""
        model_id = ""
        for turn_idx in range(3):
            kwargs = dict(model=stack_cfg["model"], messages=messages,
                          temperature=temperature, max_tokens=item.max_tokens)
            if seed is not None:
                kwargs["seed"] = seed
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content or ""
            messages.append({"role": "assistant", "content": content})
            turn_outputs.append({"output_text": content,
                                 "finish_reason": choice.finish_reason})
            if response.usage:
                usage_total["prompt_tokens"] += response.usage.prompt_tokens
                usage_total["completion_tokens"] += response.usage.completion_tokens
            finish_reason = choice.finish_reason
            model_id = response.model
            if turn_idx < len(prompts):
                messages.append({"role": "user", "content": prompts[turn_idx]})
        return {
            "output_text": turn_outputs[-1]["output_text"],
            "inference_duration_ms": 0.0,  # filled by caller via wall clock
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": usage_total["prompt_tokens"],
                "completion_tokens": usage_total["completion_tokens"],
                "total_tokens": (
                    usage_total["prompt_tokens"] + usage_total["completion_tokens"]
                ),
            },
            "model_id_returned": model_id,
            "turns": turn_outputs,
        }

    if stack_cfg["runner"] == "deepseek_runner":
        from src.models import deepseek_runner
        import urllib.request
        import time as _time
        api_key = deepseek_runner._get_api_key()
        messages = [
            {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
            {"role": "user",
             "content": f"{REFINEMENT_TURN1_PROMPT}\n\n{abstract_text}"},
        ]
        prompts = [REFINEMENT_TURN2_PROMPT, REFINEMENT_TURN3_PROMPT]
        turn_outputs = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
        finish_reason = ""
        model_id = ""
        for turn_idx in range(3):
            payload = {
                "model": stack_cfg["model"],
                "messages": messages,
                "max_tokens": item.max_tokens,
                "temperature": temperature,
            }
            if seed is not None:
                payload["seed"] = seed
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                deepseek_runner.DEEPSEEK_API_URL, data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
            )
            t0 = _time.perf_counter()
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            finish_reason = result["choices"][0].get("finish_reason", "")
            model_id = result.get("model", "")
            messages.append({"role": "assistant", "content": content})
            turn_outputs.append({"output_text": content,
                                 "finish_reason": finish_reason})
            usage = result.get("usage", {})
            usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
            usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
            if turn_idx < len(prompts):
                messages.append({"role": "user", "content": prompts[turn_idx]})
        return {
            "output_text": turn_outputs[-1]["output_text"],
            "inference_duration_ms": 0.0,
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": usage_total["prompt_tokens"],
                "completion_tokens": usage_total["completion_tokens"],
                "total_tokens": (
                    usage_total["prompt_tokens"] + usage_total["completion_tokens"]
                ),
            },
            "model_id_returned": model_id,
            "turns": turn_outputs,
        }

    # Fallback for stacks without chat helpers — single-turn dispatch with
    # full conversational prompt concatenated. Less faithful to the
    # multi-turn semantics but still exercises the pipeline.
    return _execute_single_turn(runner_module, stack_cfg, item)


# ─── Per-run executor ─────────────────────────────────────────────────────────

def execute_run(
    item: RunItem,
    output_dir: Path,
    run_card_dir: Path,
    dry_run: bool,
    budget_guard: BudgetGuard,
    checkpoint: Checkpoint,
) -> dict:
    """Execute (or simulate) one ``RunItem``. Returns a status dict."""
    run_id = item.run_id
    run_path = output_dir / f"{run_id}.json"
    if checkpoint.is_complete(run_id) or run_path.exists():
        return {"run_id": run_id, "status": "skipped"}

    stack_cfg = STACK_REGISTRY[item.stack]

    # Cost estimate (always computed, even in dry-run).
    full_prompt = item.system + "\n" + item.prompt
    if "text" in item.item:
        full_prompt += "\n" + item.item["text"]
    elif "question" in item.item:
        full_prompt += "\n" + item.item["question"]
    cost_est = estimate_call_cost(
        stack=item.stack, input_text=full_prompt,
        max_output_tokens=item.max_tokens, n_reps=item.n_turns or 1,
    )
    est_usd = cost_est.total_usd

    if dry_run:
        return {
            "run_id": run_id, "status": "planned",
            "estimated_usd": est_usd, "stack": item.stack, "task": item.task,
        }

    # Budget check.
    if not budget_guard.would_allow(est_usd):
        return {
            "run_id": run_id, "status": "skipped_over_budget",
            "estimated_usd": est_usd,
            "remaining_usd": budget_guard.remaining(),
        }

    # Real call.
    runner_module = _load_runner(stack_cfg["runner"])
    inference_params = {
        "temperature": _build_inference_kwargs(
            stack_cfg, item.condition, item.rep, item.max_tokens
        ).get("temperature"),
        "max_tokens": item.max_tokens,
        "decoding_strategy": "greedy",
    }
    seed_used = _build_inference_kwargs(
        stack_cfg, item.condition, item.rep, item.max_tokens
    ).get("seed")
    if seed_used is not None:
        inference_params["seed"] = seed_used
        if stack_cfg["runner"] == "claude_runner":
            inference_params["seed_note"] = "logged-only-not-sent-to-api"

    model_info_fn = getattr(runner_module, "get_model_info", lambda m: {})
    model_info = model_info_fn(stack_cfg["model"]) or {}

    logger = RunLogger(str(output_dir))
    logger.start_run(
        run_id=run_id, task_id=item.task,
        task_category=item.task_category,
        prompt_text=item.prompt, model_name=model_info.get("model_name", stack_cfg["model"]),
        model_version=model_info.get("model_version", "unknown"),
        inference_params=inference_params,
        researcher_id=RESEARCHER_ID, affiliation=AFFILIATION,
        input_text=item.item.get("text", item.item.get("question", "")),
        weights_hash=model_info.get("weights_hash", ""),
        model_source=model_info.get("model_source", ""),
        interaction_regime=item.interaction_regime,
    )

    output_text = ""
    system_logs = ""
    errors: list[str] = []
    try:
        if item.n_turns > 1:
            result = _execute_multiturn(runner_module, stack_cfg, item)
        else:
            result = _execute_single_turn(runner_module, stack_cfg, item)
        output_text = result.get("output_text", "")
        system_logs = json.dumps(
            {k: v for k, v in result.items() if k != "output_text"},
            default=str,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            OSError, json.JSONDecodeError) as e:
        errors = [f"{type(e).__name__}: {e}"]
    except Exception as e:  # noqa: BLE001 — we log and continue
        errors = [f"{type(e).__name__}: {e}",
                  traceback.format_exc(limit=3)]

    # Per-task output_metrics — compute pass@1 / accuracy when applicable.
    extra_metrics: dict = {}
    if item.task == "humaneval" and not errors:
        from src.tasks import pass_at_1
        try:
            pa1 = pass_at_1.run_pass_at_1(
                prompt=item.item["prompt"], completion=output_text,
                test=item.item["test"], entry_point=item.item["entry_point"],
                timeout=5.0,
            )
            extra_metrics["pass_at_1"] = pa1
        except Exception as e:
            extra_metrics["pass_at_1_error"] = str(e)
    elif item.task == "gsm8k" and not errors:
        is_correct = gsm8k_extractor.is_correct(
            output_text, item.item.get("gold_answer", "")
        )
        extra_metrics["gsm8k_correct"] = bool(is_correct)
        extra_metrics["gold_answer"] = item.item.get("gold_answer", "")
        extra_metrics["extracted_pred"] = gsm8k_extractor.extract_final_answer(
            output_text
        )

    logger.log_output(
        output_text=output_text, metrics=extra_metrics,
        system_logs=system_logs, errors=errors,
    )
    logger.save()

    rc = RunCard(str(run_card_dir))
    card = rc.create_from_run(
        logger.run_data,
        prompt_card_ref=f"prompt_card_{item.task}_v1_0.json",
    )
    rc.save(card)

    # Best-effort cost charge (use estimate; real usage is logged in system_logs).
    if not errors:
        budget_guard.charge(est_usd, label=run_id)
        checkpoint.mark_complete(run_id, cost_usd=est_usd)
    else:
        checkpoint.log_error(run_id, "; ".join(errors))

    return {
        "run_id": run_id,
        "status": "error" if errors else "ok",
        "estimated_usd": est_usd,
        "errors": errors,
        "output_chars": len(output_text),
    }


# ─── Plan + dry-run reporting ─────────────────────────────────────────────────

def expand_tasks(arg: str) -> list[str]:
    if arg == "all":
        return list(PLAN_BUILDERS.keys())
    return [arg]


def expand_stacks(arg: str) -> list[str]:
    if arg == "all":
        return list(STACK_REGISTRY.keys())
    return [arg]


def expand_conditions(arg: str) -> list[str]:
    if arg == "all":
        return list(CONDITIONS.keys())
    return [arg]


def build_plan(args) -> list[RunItem]:
    plan = []
    for task in expand_tasks(args.task):
        n_problems = args.n_problems or DEFAULT_N_PROBLEMS[task]
        for stack in expand_stacks(args.stack):
            for cond in expand_conditions(args.condition):
                builder = PLAN_BUILDERS[task]
                try:
                    items = builder(stack, cond, n_problems, args.n_reps)
                except FileNotFoundError as e:
                    print(f"[WARN] Skipping {task}/{stack}/{cond}: {e}",
                          file=sys.stderr)
                    items = []
                plan.extend(items)
    return plan


def summarize_plan(plan: list[RunItem]) -> dict:
    """Compute a dry-run summary: total runs, estimated cost per stack/task."""
    by_stack_task: dict[tuple[str, str], dict] = {}
    grand = {"n_runs": 0, "input_usd": 0.0, "output_usd": 0.0,
             "input_tokens": 0, "output_tokens": 0}
    for item in plan:
        key = (item.stack, item.task)
        e = by_stack_task.setdefault(
            key, {"n_runs": 0, "input_tokens": 0, "output_tokens": 0,
                  "input_usd": 0.0, "output_usd": 0.0}
        )
        full_prompt = item.system + "\n" + item.prompt + "\n" + (
            item.item.get("text", "") + item.item.get("question", "")
        )
        cost = estimate_call_cost(
            stack=item.stack, input_text=full_prompt,
            max_output_tokens=item.max_tokens, n_reps=item.n_turns or 1,
        )
        e["n_runs"] += 1
        e["input_tokens"] += cost.input_tokens
        e["output_tokens"] += cost.output_tokens
        e["input_usd"] += cost.input_usd
        e["output_usd"] += cost.output_usd
        grand["n_runs"] += 1
        grand["input_usd"] += cost.input_usd
        grand["output_usd"] += cost.output_usd
        grand["input_tokens"] += cost.input_tokens
        grand["output_tokens"] += cost.output_tokens

    return {
        "by_stack_task": [
            {
                "stack": k[0], "task": k[1], **{kk: round(vv, 4)
                                                if isinstance(vv, float) else vv
                                                for kk, vv in v.items()},
                "total_usd": round(v["input_usd"] + v["output_usd"], 4),
            }
            for k, v in sorted(by_stack_task.items())
        ],
        "grand_total": {
            "n_runs": grand["n_runs"],
            "input_tokens": grand["input_tokens"],
            "output_tokens": grand["output_tokens"],
            "input_usd": round(grand["input_usd"], 4),
            "output_usd": round(grand["output_usd"], 4),
            "total_usd": round(grand["input_usd"] + grand["output_usd"], 4),
        },
    }


def print_dry_run(summary: dict, budget_cap: float):
    print("\n" + "=" * 78)
    print("DRY RUN — no API calls will be made")
    print("=" * 78)
    print(f"\n{'Stack':<30} {'Task':<22} {'Runs':>5} {'In Tok':>10} "
          f"{'Out Tok':>10} {'USD':>9}")
    print("-" * 78)
    for row in summary["by_stack_task"]:
        print(f"{row['stack']:<30} {row['task']:<22} {row['n_runs']:>5} "
              f"{row['input_tokens']:>10} {row['output_tokens']:>10} "
              f"${row['total_usd']:>8.4f}")
    g = summary["grand_total"]
    print("-" * 78)
    print(f"{'GRAND TOTAL':<30} {'':<22} {g['n_runs']:>5} "
          f"{g['input_tokens']:>10} {g['output_tokens']:>10} "
          f"${g['total_usd']:>8.4f}")
    print(f"\nBudget cap: ${budget_cap:.2f} USD")
    if g['total_usd'] > budget_cap:
        print(f"  >>> OVER BUDGET by ${g['total_usd'] - budget_cap:.4f} <<<")
    else:
        print(f"  Headroom: ${budget_cap - g['total_usd']:.4f}")
    print("=" * 78 + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified NatComms-revision experiment runner."
    )
    parser.add_argument(
        "--task", required=True,
        choices=list(PLAN_BUILDERS.keys()) + ["all"],
    )
    parser.add_argument(
        "--stack", required=True,
        choices=list(STACK_REGISTRY.keys()) + ["all"],
    )
    parser.add_argument(
        "--condition", default="C1", choices=list(CONDITIONS.keys()) + ["all"],
    )
    parser.add_argument("--n-problems", type=int, default=None)
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--output-dir", type=str,
                        default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--checkpoint", type=str,
                        default=str(CHECKPOINT_DEFAULT))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--budget-usd", type=float, default=50.0)
    parser.add_argument("--resume", action="store_true",
                        help="(default) Skip runs already in checkpoint.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run everything, ignoring checkpoint.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan + cost only — no API calls.")
    parser.add_argument("--execute", action="store_true",
                        help="Required to actually run experiments.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_card_dir = output_dir.parent / "run_cards"
    run_card_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("NatComms Revision — Unified Experiment Runner")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Task: {args.task}  |  Stack: {args.stack}  |  Cond: {args.condition}")
    print(f"n_problems={args.n_problems or 'default'}  n_reps={args.n_reps}")
    print(f"Output dir: {output_dir}")
    print(f"Budget cap: ${args.budget_usd}")
    print("=" * 78)

    plan = build_plan(args)
    if not plan:
        print("[INFO] Empty plan. Exiting.")
        return 0

    summary = summarize_plan(plan)
    print_dry_run(summary, budget_cap=args.budget_usd)

    if args.dry_run or not args.execute:
        if not args.execute:
            print("[INFO] --execute not set. Pass --execute to run for real.")
        return 0

    # Real execution.
    checkpoint = Checkpoint(Path(args.checkpoint))
    if args.no_resume:
        checkpoint.data["completed_run_ids"] = []
    budget_guard = BudgetGuard(cap_usd=args.budget_usd)
    budget_guard.spent_usd = checkpoint.data.get("spent_usd", 0.0)

    n_done = 0
    n_skipped = 0
    n_errors = 0
    started = time.time()
    for idx, item in enumerate(plan, 1):
        result = execute_run(
            item, output_dir=output_dir, run_card_dir=run_card_dir,
            dry_run=False, budget_guard=budget_guard, checkpoint=checkpoint,
        )
        status = result["status"]
        if status == "ok":
            n_done += 1
        elif status.startswith("skipped"):
            n_skipped += 1
        else:
            n_errors += 1
        flag = {
            "ok": "[OK]", "skipped": "[SKIP]",
            "skipped_over_budget": "[BUDGET]",
            "error": "[ERR]",
        }.get(status, "[??]")
        print(
            f"  {flag} ({idx}/{len(plan)}) {result['run_id']} "
            f"| spent=${budget_guard.spent_usd:.4f}/{budget_guard.cap_usd:.2f}",
            flush=True,
        )

        if idx % args.checkpoint_every == 0:
            checkpoint.save()

    checkpoint.save()
    elapsed = time.time() - started
    print(f"\n{'=' * 78}")
    print(
        f"Finished: ok={n_done}  skipped={n_skipped}  errors={n_errors}  "
        f"elapsed={elapsed:.1f}s"
    )
    print(f"Total spent (est.): ${budget_guard.spent_usd:.4f} / "
          f"${budget_guard.cap_usd:.2f}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
