#!/usr/bin/env python3
"""Multi-turn experiment runner for Gemini 2.5 Pro (Google AI Studio API).

Runs Task 3 (multi-turn refinement) and Task 4 (RAG extraction) under
condition C1 (fixed seed=42, temperature=0.0, 5 repetitions) using
Gemini 2.5 Pro via the Google Generative Language API.

Total: 10 abstracts x 5 reps x 2 tasks = 100 runs.

Skips any run whose output file already exists.

Usage:
    python run_gemini_multiturn.py                          # Default
    python run_gemini_multiturn.py --model gemini-2.5-pro   # Explicit model
    python run_gemini_multiturn.py --abstracts 5            # Fewer abstracts
    python run_gemini_multiturn.py --scenario refinement    # Only Task 3
    python run_gemini_multiturn.py --scenario rag           # Only Task 4
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
from src.models import gemini_runner
from src.experiments.config import (
    RESEARCHER_ID,
    AFFILIATION,
    SEEDS,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_N_ABSTRACTS = 10
N_REPS = 5
MAX_RETRIES = 3


# ─── Prompts (same as run_claude_multiturn.py for cross-model parity) ────────

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

# Simulated retrieved contexts (identical to run_claude_multiturn.py)
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


# ─── API Helper with Retry ───────────────────────────────────────────────────

def _api_call_with_retry(call_fn, max_retries: int = MAX_RETRIES) -> dict:
    """Execute an API call with exponential backoff on failure."""
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
    """Build a standardized run_id for Gemini.

    Convention: gemini-1.5-pro_{task}_abs_{NNN}_C1_fixed_seed_rep{N}
    """
    run_id = f"{model_name}_{task_id}_{abstract_id}_{condition}_rep{rep}"
    return run_id.replace(":", "_").replace(" ", "_").replace(".", "_")


# ─── Task 3: Multi-turn Refinement ───────────────────────────────────────────

def run_refinement_experiment(
    model_name: str, abstract: dict, rep: int,
    temperature: float = 0.0, seed: Optional[int] = None,
) -> dict:
    """Run a 3-turn iterative refinement extraction via Gemini API.

    Turn 1: Extract structured info from abstract.
    Turn 2: Send feedback requesting review/correction.
    Turn 3: Final verification.

    Returns the logger's run_data dict, or None if the run already exists.
    """
    run_id = make_run_id(model_name, "multiturn_refinement",
                         abstract["id"], "C1_fixed_seed", rep)

    if run_exists(run_id):
        return None

    # Inference params (seed IS sent to Gemini API)
    inference_params = gemini_runner.get_inference_params(
        temperature=temperature, max_tokens=8192, seed=seed,
    )
    inference_params["note"] = "maxOutputTokens=8192 to accommodate thinking tokens"

    model_info = gemini_runner.get_model_info(model_name)

    # Build the full multi-turn prompt text for hashing
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
        # Gemini API: system instruction is a top-level parameter.
        # contents[] array uses role: "user" / "model".
        contents = []

        # Turn 1: Initial extraction
        contents.append({
            "role": "user",
            "parts": [{"text": f"{REFINEMENT_TURN1_PROMPT}\n\n{abstract['text']}"}],
        })
        turn1 = _api_call_with_retry(
            lambda: gemini_runner.run_multiturn_inference(
                contents=list(contents),
                model=model_name,
                system_prompt=REFINEMENT_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=8192,
                seed=seed,
            )
        )
        contents.append({
            "role": "model",
            "parts": [{"text": turn1["output_text"]}],
        })

        # Turn 2: Review and correct
        contents.append({
            "role": "user",
            "parts": [{"text": REFINEMENT_TURN2_PROMPT}],
        })
        turn2 = _api_call_with_retry(
            lambda: gemini_runner.run_multiturn_inference(
                contents=list(contents),
                model=model_name,
                system_prompt=REFINEMENT_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=8192,
                seed=seed,
            )
        )
        contents.append({
            "role": "model",
            "parts": [{"text": turn2["output_text"]}],
        })

        # Turn 3: Final verification
        contents.append({
            "role": "user",
            "parts": [{"text": REFINEMENT_TURN3_PROMPT}],
        })
        turn3 = _api_call_with_retry(
            lambda: gemini_runner.run_multiturn_inference(
                contents=list(contents),
                model=model_name,
                system_prompt=REFINEMENT_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=8192,
                seed=seed,
            )
        )

        # Final output is turn 3
        output_text = turn3["output_text"]

        # Build full conversation for logging (matches run_claude_multiturn.py format)
        full_conversation = (
            [{"role": "system", "content": REFINEMENT_SYSTEM_PROMPT}]
            + [
                {"role": "user", "content": contents[0]["parts"][0]["text"]},
                {"role": "assistant", "content": turn1["output_text"]},
                {"role": "user", "content": contents[2]["parts"][0]["text"]},
                {"role": "assistant", "content": turn2["output_text"]},
                {"role": "user", "content": contents[4]["parts"][0]["text"]},
                {"role": "assistant", "content": turn3["output_text"]},
            ]
        )
        conversation_history_hash = hash_text(
            json.dumps(full_conversation, sort_keys=True, ensure_ascii=True)
        )

        system_logs = json.dumps({
            "turn1": {
                "output": turn1["output_text"],
                "duration_ms": turn1["inference_duration_ms"],
                "usage": turn1["usage"],
            },
            "turn2": {
                "output": turn2["output_text"],
                "duration_ms": turn2["inference_duration_ms"],
                "usage": turn2["usage"],
            },
            "turn3": {
                "output": turn3["output_text"],
                "duration_ms": turn3["inference_duration_ms"],
                "usage": turn3["usage"],
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
    """Run RAG-enhanced extraction with retrieved context via Gemini API.

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

    # Inference params (seed IS sent to Gemini API)
    inference_params = gemini_runner.get_inference_params(
        temperature=temperature, max_tokens=8192, seed=seed,
    )

    model_info = gemini_runner.get_model_info(model_name)

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
        contents = [{
            "role": "user",
            "parts": [{"text": f"{RAG_PROMPT}\n\n{user_content}"}],
        }]
        result = _api_call_with_retry(
            lambda: gemini_runner.run_multiturn_inference(
                contents=contents,
                model=model_name,
                system_prompt=RAG_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=8192,
                seed=seed,
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
        description="Gemini Multi-turn Experiment Runner (Tasks 3 & 4)"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Gemini model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--abstracts", type=int, default=DEFAULT_N_ABSTRACTS,
                        help=f"Number of abstracts (default: {DEFAULT_N_ABSTRACTS})")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=["refinement", "rag"],
                        help="Run only this scenario (default: both)")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else ["refinement", "rag"]

    print("=" * 70)
    print("GenAI Reproducibility - Gemini Multi-Turn Experiments (Tasks 3 & 4)")
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
        test = gemini_runner.run_inference(
            prompt="Say OK",
            model=args.model,
            max_tokens=5,
            seed=42,
            timeout=15,
        )
        print(f"  [OK] API works. Model version: {test.get('model_id_returned', '?')}")
    except Exception as e:
        print(f"\nERROR: Cannot reach Gemini API: {e}")
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
        print(f"  Condition C1: seed=42, temp=0.0, {N_REPS} reps")
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
        print(f"  Condition C1: seed=42, temp=0.0, {N_REPS} reps")
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
