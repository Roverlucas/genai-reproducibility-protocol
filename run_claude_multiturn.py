#!/usr/bin/env python3
"""Multi-turn experiment runner for Claude (Anthropic API).

Runs Task 3 (multi-turn refinement) and Task 4 (RAG extraction) under
condition C1 (fixed seed=42, temperature=0.0, 5 repetitions) using
Claude Sonnet 4.5 via the Anthropic Messages API.

Total: 10 abstracts x 5 reps x 2 tasks = 100 runs.

Skips any run whose output file already exists.

Usage:
    python run_claude_multiturn.py                                    # Default
    python run_claude_multiturn.py --model claude-sonnet-4-5-20250929  # Explicit model
    python run_claude_multiturn.py --abstracts 5                      # Fewer abstracts
    python run_claude_multiturn.py --scenario refinement              # Only Task 3
    python run_claude_multiturn.py --scenario rag                     # Only Task 4
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.protocol.logger import RunLogger
from src.protocol.run_card import RunCard
from src.protocol.hasher import hash_text
from src.models import claude_runner
from src.experiments.config import (
    RESEARCHER_ID,
    AFFILIATION,
    SEEDS,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_N_ABSTRACTS = 10
N_REPS = 5
MAX_RETRIES = 3


# ─── Prompts (same as run_multiturn.py for cross-model parity) ───────────────

REFINEMENT_SYSTEM_PROMPT = (
    "You are a structured information extraction assistant. "
    "You extract information from scientific abstracts into JSON format."
)

REFINEMENT_TURN1_PROMPT = (
    "Read the following scientific abstract and extract the information "
    "into the exact JSON format below. Use only information explicitly stated "
    "in the abstract. If a field is not mentioned, use null.\n\n"
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
    "Review your extraction above. Check each field carefully against the "
    "original abstract. If any field is incomplete, imprecise, or missing "
    "quantitative details that are in the abstract, correct it. "
    "Output only the corrected JSON, no explanation."
)

REFINEMENT_TURN3_PROMPT = (
    "Now produce the final verified extraction. Ensure all fields contain "
    "the most precise information from the abstract. If quantitative results "
    "are mentioned (percentages, scores, metrics), they must appear in "
    "key_result. Output only the final JSON."
)

RAG_SYSTEM_PROMPT = (
    "You are a structured information extraction assistant with access to "
    "retrieved context from related papers. Use both the abstract and the "
    "retrieved context to produce the most accurate extraction possible."
)

RAG_PROMPT = (
    "You are given a scientific abstract and additional context retrieved "
    "from related papers. Extract information from the PRIMARY abstract into "
    "the JSON format below. Use the retrieved context only to disambiguate "
    "or enrich fields, but the abstract is the authoritative source.\n\n"
    "Output format (JSON only, no explanation):\n"
    "{\n"
    '  "objective": "string",\n'
    '  "method": "string",\n'
    '  "key_result": "string",\n'
    '  "model_or_system": "string",\n'
    '  "benchmark": "string"\n'
    "}"
)

# Simulated retrieved contexts (same as run_multiturn.py)
RETRIEVED_CONTEXTS = {
    "abs_001": (
        "Related work: The sequence-to-sequence architecture (Sutskever et al., 2014) "
        "established the encoder-decoder paradigm for neural machine translation. "
        "Bahdanau et al. (2015) introduced attention mechanisms to address the "
        "information bottleneck in fixed-length encoding. The WMT benchmark "
        "(Workshop on Machine Translation) is the standard evaluation for MT systems."
    ),
    "abs_002": (
        "Related work: ELMo (Peters et al., 2018) demonstrated that deep "
        "contextualized word representations improve NLP tasks. GPT (Radford et al., "
        "2018) showed that generative pre-training followed by discriminative "
        "fine-tuning achieves strong results. The GLUE benchmark (Wang et al., 2018) "
        "and SQuAD (Rajpurkar et al., 2016) are standard NLU evaluation suites."
    ),
    "abs_003": (
        "Related work: GPT-2 (Radford et al., 2019) explored language model scaling "
        "up to 1.5B parameters. Scaling laws (Kaplan et al., 2020) showed predictable "
        "performance improvements with model size. In-context learning was first "
        "observed in large autoregressive models performing tasks through prompting."
    ),
    "abs_004": (
        "Related work: BERT (Devlin et al., 2019) established masked language modeling "
        "for pre-training. UniLM (Dong et al., 2019) unified different pre-training "
        "objectives. The Colossal Clean Crawled Corpus (C4) was introduced specifically "
        "for studying transfer learning at scale."
    ),
    "abs_005": (
        "Related work: Few-shot prompting (Brown et al., 2020) showed that large "
        "models can perform tasks with minimal examples. Self-consistency (Wang et al., "
        "2022) improves reasoning by sampling multiple chains. GSM8K (Cobbe et al., "
        "2021) contains 8.5K grade school math word problems for evaluating reasoning."
    ),
    "abs_006": (
        "Related work: Variational Autoencoders (Kingma & Welling, 2014) offered an "
        "alternative generative approach. Wasserstein GAN (Arjovsky et al., 2017) "
        "improved training stability. The minimax game formulation connects to "
        "Nash equilibria in game theory."
    ),
    "abs_007": (
        "Related work: VGGNet (Simonyan & Zisserman, 2015) showed benefits of "
        "increased depth up to 19 layers. GoogLeNet (Szegedy et al., 2015) used "
        "inception modules. The ImageNet Large Scale Visual Recognition Challenge "
        "(ILSVRC) has been the primary benchmark for image classification since 2010."
    ),
    "abs_008": (
        "Related work: The wake-sleep algorithm (Hinton et al., 1995) and "
        "Helmholtz machines are predecessors for variational inference in latent "
        "variable models. The reparameterization trick enables gradient-based "
        "optimization through stochastic layers."
    ),
    "abs_009": (
        "Related work: Word2Vec (Mikolov et al., 2013) showed that word embeddings "
        "capture semantic relationships. Skip-gram and CBOW architectures provided "
        "efficient training on large corpora. GloVe (Pennington et al., 2014) "
        "combined count-based and prediction-based methods."
    ),
    "abs_010": (
        "Related work: Attention mechanisms (Bahdanau et al., 2015) allow models to "
        "focus on relevant input parts. Multi-head attention (Vaswani et al., 2017) "
        "enables attending to different representation subspaces. Self-attention "
        "has become fundamental in modern NLP architectures."
    ),
}


# ─── Claude Multi-turn API Helper ────────────────────────────────────────────

def claude_multiturn_inference(
    messages: list,
    model: str = DEFAULT_MODEL,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 90,
) -> dict:
    """Call Anthropic Messages API with a multi-turn messages array.

    Uses urllib directly (same approach as claude_runner.py) to avoid
    external dependencies.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str} dicts.
                  Must NOT include system messages (use system_prompt param).
        model: Anthropic model ID.
        system_prompt: Optional system prompt (separate from messages).
        temperature: Sampling temperature (0.0 = greedy).
        max_tokens: Maximum output tokens.
        timeout: Request timeout in seconds.

    Returns:
        Dict with output_text, inference_duration_ms, usage, etc.
    """
    api_key = claude_runner._get_api_key()

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    if temperature > 0.0:
        payload["temperature"] = temperature
    if system_prompt:
        payload["system"] = system_prompt

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": claude_runner.ANTHROPIC_API_VERSION,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        claude_runner.ANTHROPIC_API_URL, data=data, headers=headers
    )

    start_time = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    end_time = time.perf_counter()

    inference_duration_ms = (end_time - start_time) * 1000

    content_text = ""
    for block in result.get("content", []):
        if block.get("type") == "text":
            content_text += block.get("text", "")

    usage = result.get("usage", {})

    return {
        "output_text": content_text,
        "inference_duration_ms": round(inference_duration_ms, 2),
        "finish_reason": result.get("stop_reason", ""),
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
        "model_id_returned": result.get("model", ""),
        "response_id": result.get("id", ""),
    }


def _api_call_with_retry(call_fn, max_retries: int = MAX_RETRIES) -> dict:
    """Execute an API call with exponential backoff on failure.

    Args:
        call_fn: A zero-argument callable that performs the API call.
        max_retries: Max number of retry attempts.

    Returns:
        The result dict from the API call.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return call_fn()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError, json.JSONDecodeError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2s, 4s
                print(f"      [RETRY] Attempt {attempt + 1}/{max_retries} "
                      f"failed: {e}. Waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"      [FAIL] All {max_retries} attempts exhausted: {e}",
                      flush=True)
    raise last_error


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_abstracts(n: int = DEFAULT_N_ABSTRACTS) -> list:
    """Load scientific abstracts (first n)."""
    with open(Path(__file__).parent / "data" / "inputs" / "abstracts.json") as f:
        data = json.load(f)
    return data["abstracts"][:n]


def run_exists(run_id: str) -> bool:
    """Check if a run output file already exists."""
    return (OUTPUT_DIR / "runs" / f"{run_id}.json").exists()


def make_run_id(model_name: str, task_id: str, abstract_id: str,
                condition: str, rep: int) -> str:
    """Build a standardized run_id matching existing Claude naming convention."""
    # Shorten model name: claude-sonnet-4-5-20250929 -> sonnet-4-5
    short = model_name.replace("claude-", "").replace("-20250929", "")
    run_id = f"{short}_{task_id}_{abstract_id}_{condition}_rep{rep}"
    return run_id.replace(":", "_").replace(" ", "_").replace(".", "_")


# ─── Task 3: Multi-turn Refinement ───────────────────────────────────────────

def run_refinement_experiment(
    model_name: str, abstract: dict, rep: int,
    temperature: float = 0.0, seed: Optional[int] = None,
) -> dict:
    """Run a 3-turn iterative refinement extraction via Claude API.

    Turn 1: Extract structured info from abstract.
    Turn 2: Send feedback requesting review/correction.
    Turn 3: Final verification.

    Returns the logger's run_data dict, or None if the run already exists.
    """
    run_id = make_run_id(model_name, "multiturn_refinement",
                         abstract["id"], "C1_fixed_seed", rep)

    if run_exists(run_id):
        return None

    # Inference params (seed logged but not sent to API)
    inference_params = claude_runner.get_inference_params(
        temperature=temperature, max_tokens=1024,
    )
    if seed is not None:
        inference_params["seed"] = seed
        inference_params["seed_note"] = "logged-only-not-sent-to-api"

    model_info = claude_runner.get_model_info(model_name)

    # Build the full multi-turn prompt text for hashing (same format as run_multiturn.py)
    full_prompt = (f"[SYSTEM] {REFINEMENT_SYSTEM_PROMPT}\n"
                   f"[TURN1] {REFINEMENT_TURN1_PROMPT}\n"
                   f"[TURN2] {REFINEMENT_TURN2_PROMPT}\n"
                   f"[TURN3] {REFINEMENT_TURN3_PROMPT}")

    logger = RunLogger(str(OUTPUT_DIR / "runs"))
    logger.start_run(
        run_id=run_id,
        task_id="multiturn_refinement",
        task_category="iterative_structured_extraction",
        prompt_text=full_prompt,
        model_name=model_info.get("model_name", model_name),
        model_version=model_info.get("model_version", "unknown"),
        inference_params=inference_params,
        researcher_id=RESEARCHER_ID,
        affiliation=AFFILIATION,
        input_text=abstract["text"],
        weights_hash=model_info.get("weights_hash", ""),
        model_source=model_info.get("model_source", ""),
        interaction_regime="multi-turn",
    )

    try:
        # Claude API: system prompt is a top-level parameter, not in messages.
        # Messages array contains only user/assistant turns.
        messages = []

        # Turn 1: Initial extraction
        messages.append({
            "role": "user",
            "content": f"{REFINEMENT_TURN1_PROMPT}\n\n{abstract['text']}"
        })
        turn1 = _api_call_with_retry(
            lambda: claude_multiturn_inference(
                messages=list(messages),
                model=model_name,
                system_prompt=REFINEMENT_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=1024,
            )
        )
        messages.append({"role": "assistant", "content": turn1["output_text"]})

        # Turn 2: Review and correct
        messages.append({"role": "user", "content": REFINEMENT_TURN2_PROMPT})
        turn2 = _api_call_with_retry(
            lambda: claude_multiturn_inference(
                messages=list(messages),
                model=model_name,
                system_prompt=REFINEMENT_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=1024,
            )
        )
        messages.append({"role": "assistant", "content": turn2["output_text"]})

        # Turn 3: Final verification
        messages.append({"role": "user", "content": REFINEMENT_TURN3_PROMPT})
        turn3 = _api_call_with_retry(
            lambda: claude_multiturn_inference(
                messages=list(messages),
                model=model_name,
                system_prompt=REFINEMENT_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=1024,
            )
        )

        # Final output is turn 3
        output_text = turn3["output_text"]

        # Build full conversation for logging (matches run_multiturn.py format)
        full_conversation = (
            [{"role": "system", "content": REFINEMENT_SYSTEM_PROMPT}]
            + messages
            + [{"role": "assistant", "content": turn3["output_text"]}]
        )
        conversation_history_hash = hash_text(
            json.dumps(full_conversation, sort_keys=True, ensure_ascii=True)
        )

        system_logs = json.dumps({
            "turn1": {
                "output": turn1["output_text"],
                "duration_ms": turn1["inference_duration_ms"],
                "usage": turn1["usage"],
                "response_id": turn1["response_id"],
            },
            "turn2": {
                "output": turn2["output_text"],
                "duration_ms": turn2["inference_duration_ms"],
                "usage": turn2["usage"],
                "response_id": turn2["response_id"],
            },
            "turn3": {
                "output": turn3["output_text"],
                "duration_ms": turn3["inference_duration_ms"],
                "usage": turn3["usage"],
                "response_id": turn3["response_id"],
            },
            "total_turns": 3,
            "conversation_history_hash": conversation_history_hash,
            "model_id_returned": turn3["model_id_returned"],
            "full_conversation": full_conversation,
        }, default=str)
        errors = []

    except Exception as e:
        output_text = ""
        system_logs = json.dumps({"error": str(e)}, default=str)
        errors = [str(e)]

    logger.log_output(output_text=output_text, system_logs=system_logs, errors=errors)
    logger.save()

    rc = RunCard(str(OUTPUT_DIR / "run_cards"))
    run_card = rc.create_from_run(
        logger.run_data,
        prompt_card_ref="prompt_card_multiturn_refinement_v1_0.json",
    )
    rc.save(run_card)

    return logger.run_data


# ─── Task 4: RAG-enhanced Extraction ─────────────────────────────────────────

def run_rag_experiment(
    model_name: str, abstract: dict, rep: int,
    temperature: float = 0.0, seed: Optional[int] = None,
) -> dict:
    """Run RAG-enhanced extraction with retrieved context via Claude API.

    Single-turn: provides retrieved context alongside the abstract.

    Returns the logger's run_data dict, or None if the run already exists.
    """
    run_id = make_run_id(model_name, "rag_extraction",
                         abstract["id"], "C1_fixed_seed", rep)

    if run_exists(run_id):
        return None

    abstract_id = abstract["id"]
    retrieved_context = RETRIEVED_CONTEXTS.get(
        abstract_id, "No additional context available."
    )

    # Inference params (seed logged but not sent to API)
    inference_params = claude_runner.get_inference_params(
        temperature=temperature, max_tokens=1024,
    )
    if seed is not None:
        inference_params["seed"] = seed
        inference_params["seed_note"] = "logged-only-not-sent-to-api"

    model_info = claude_runner.get_model_info(model_name)

    # Construct user message with both abstract and context
    user_content = (
        f"PRIMARY ABSTRACT:\n{abstract['text']}\n\n"
        f"RETRIEVED CONTEXT (from related papers):\n{retrieved_context}"
    )

    logger = RunLogger(str(OUTPUT_DIR / "runs"))
    logger.start_run(
        run_id=run_id,
        task_id="rag_extraction",
        task_category="rag_structured_extraction",
        prompt_text=RAG_PROMPT,
        model_name=model_info.get("model_name", model_name),
        model_version=model_info.get("model_version", "unknown"),
        inference_params=inference_params,
        researcher_id=RESEARCHER_ID,
        affiliation=AFFILIATION,
        input_text=abstract["text"],
        retrieval_context={
            "source": "simulated_retrieval",
            "query": abstract["source"],
            "retrieved_text": retrieved_context,
            "n_chunks": 1,
        },
        weights_hash=model_info.get("weights_hash", ""),
        model_source=model_info.get("model_source", ""),
        interaction_regime="single-turn-with-context",
    )

    try:
        messages = [
            {"role": "user", "content": f"{RAG_PROMPT}\n\n{user_content}"},
        ]
        result = _api_call_with_retry(
            lambda: claude_multiturn_inference(
                messages=messages,
                model=model_name,
                system_prompt=RAG_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=1024,
            )
        )

        output_text = result["output_text"]
        system_logs = json.dumps(
            {k: v for k, v in result.items() if k != "output_text"},
            default=str,
        )
        errors = []

    except Exception as e:
        output_text = ""
        system_logs = json.dumps({"error": str(e)}, default=str)
        errors = [str(e)]

    logger.log_output(output_text=output_text, system_logs=system_logs, errors=errors)
    logger.save()

    rc = RunCard(str(OUTPUT_DIR / "run_cards"))
    run_card = rc.create_from_run(
        logger.run_data,
        prompt_card_ref="prompt_card_rag_extraction_v1_0.json",
    )
    rc.save(run_card)

    return logger.run_data


# ─── Progress Printing ────────────────────────────────────────────────────────

def _print_progress(run_data, done, total, scenario):
    run_id = run_data.get("run_id", "?")
    duration = run_data.get("execution_duration_ms", 0)
    overhead = run_data.get("logging_overhead_ms", 0)
    has_error = len(run_data.get("errors", [])) > 0
    status = "ERR" if has_error else "OK"
    out_len = len(run_data.get("output_text", ""))
    pct = (done / total * 100) if total > 0 else 0
    print(f"    [{status}] ({done}/{total} {pct:.0f}%) {run_id} | "
          f"{duration:.0f}ms | oh={overhead:.1f}ms | out={out_len}c",
          flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Claude Multi-turn Experiment Runner (Tasks 3 & 4)"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Claude model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--abstracts", type=int, default=DEFAULT_N_ABSTRACTS,
                        help=f"Number of abstracts (default: {DEFAULT_N_ABSTRACTS})")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=["refinement", "rag"],
                        help="Run only this scenario (default: both)")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else ["refinement", "rag"]

    print("=" * 70)
    print("GenAI Reproducibility - Claude Multi-Turn Experiments (Tasks 3 & 4)")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Model: {args.model}")
    print(f"Abstracts: {args.abstracts}")
    print(f"Scenarios: {scenarios}")
    print(f"Condition: C1 (fixed seed=42, temp=0.0, {N_REPS} reps)")
    print(f"Total expected runs: {args.abstracts * N_REPS * len(scenarios)}")
    print("=" * 70)

    # Test API connectivity
    print("\nTesting API connectivity...")
    try:
        test = claude_runner.run_inference(
            prompt="Say OK", model=args.model, max_tokens=5, timeout=15,
        )
        print(f"  [OK] API works. Model: {test.get('model_id_returned', '?')}")
    except Exception as e:
        print(f"\nERROR: Cannot reach Anthropic API: {e}")
        sys.exit(1)

    # Load data
    abstracts = load_abstracts(args.abstracts)
    print(f"\nLoaded {len(abstracts)} abstracts")

    all_runs = []
    start = time.time()

    print(f"\n{'=' * 70}")
    print(f"MODEL: {args.model}")
    print("=" * 70)

    # Task 3: Multi-turn Refinement
    if "refinement" in scenarios:
        print(f"\n  --- Task 3: Iterative Refinement (3-turn) ---")
        print(f"  Condition C1: seed=42 (logged only), temp=0.0, {N_REPS} reps")
        total = len(abstracts) * N_REPS
        done = 0
        for abstract in abstracts:
            for rep in range(N_REPS):
                run_data = run_refinement_experiment(
                    model_name=args.model,
                    abstract=abstract,
                    rep=rep,
                    temperature=0.0,
                    seed=SEEDS[0],  # 42
                )
                done += 1
                if run_data:
                    all_runs.append(run_data)
                    _print_progress(run_data, done, total, "refinement")
                else:
                    pct = (done / total * 100) if total > 0 else 0
                    run_id = make_run_id(args.model, "multiturn_refinement",
                                         abstract["id"], "C1_fixed_seed", rep)
                    print(f"    [SKIP] ({done}/{total} {pct:.0f}%) {run_id}",
                          flush=True)

    # Task 4: RAG-enhanced Extraction
    if "rag" in scenarios:
        print(f"\n  --- Task 4: RAG-Enhanced Extraction ---")
        print(f"  Condition C1: seed=42 (logged only), temp=0.0, {N_REPS} reps")
        total = len(abstracts) * N_REPS
        done = 0
        for abstract in abstracts:
            for rep in range(N_REPS):
                run_data = run_rag_experiment(
                    model_name=args.model,
                    abstract=abstract,
                    rep=rep,
                    temperature=0.0,
                    seed=SEEDS[0],  # 42
                )
                done += 1
                if run_data:
                    all_runs.append(run_data)
                    _print_progress(run_data, done, total, "rag")
                else:
                    pct = (done / total * 100) if total > 0 else 0
                    run_id = make_run_id(args.model, "rag_extraction",
                                         abstract["id"], "C1_fixed_seed", rep)
                    print(f"    [SKIP] ({done}/{total} {pct:.0f}%) {run_id}",
                          flush=True)

    elapsed = time.time() - start

    # Summary
    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {len(all_runs)} new runs in {elapsed:.1f}s")
    refinement_count = sum(
        1 for r in all_runs if "refinement" in r.get("task_id", "")
    )
    rag_count = sum(
        1 for r in all_runs if "rag" in r.get("task_id", "")
    )
    error_count = sum(
        1 for r in all_runs if len(r.get("errors", [])) > 0
    )
    print(f"  Refinement (Task 3): {refinement_count}")
    print(f"  RAG (Task 4):        {rag_count}")
    if error_count > 0:
        print(f"  Errors:              {error_count}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
