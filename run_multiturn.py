#!/usr/bin/env python3
"""Multi-turn experiment runner for the GenAI reproducibility protocol.

Implements two interaction scenarios beyond single-turn:
  1. Iterative Refinement: Extract → Feedback → Refine (3-turn conversation)
  2. RAG-enhanced Extraction: Provide retrieved context alongside the abstract

This validates that the protocol captures multi-turn provenance correctly
and measures whether multi-turn interactions affect reproducibility.

Usage:
    python run_multiturn.py                            # All models, all scenarios
    python run_multiturn.py --model llama3:8b           # Specific model
    python run_multiturn.py --scenario refinement       # Specific scenario
    python run_multiturn.py --scenario rag              # RAG scenario only
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.protocol.logger import RunLogger
from src.protocol.prompt_card import PromptCard
from src.protocol.run_card import RunCard
from src.models import llama_runner
from src.experiments.config import (
    RESEARCHER_ID,
    AFFILIATION,
    SEEDS,
    EXTRACTION_PROMPT,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"

# Models to test multi-turn with
MULTITURN_MODELS = ["llama3:8b", "mistral:7b", "gemma2:9b"]

# Number of repetitions per condition
N_REPS = 5

# Number of abstracts for multi-turn (subset for tractability)
N_ABSTRACTS = 10


# ─── Scenario 1: Iterative Refinement ────────────────────────────────────────

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


# ─── Scenario 2: RAG-enhanced Extraction ─────────────────────────────────────

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

# Simulated retrieved contexts for first 10 abstracts
# These are synthetic "related paper snippets" providing enrichment context
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


def load_abstracts(n: int = N_ABSTRACTS) -> list:
    with open(Path(__file__).parent / "data" / "inputs" / "abstracts.json") as f:
        data = json.load(f)
    return data["abstracts"][:n]


def load_prompt_cards():
    pc_dir = OUTPUT_DIR / "prompt_cards"
    sum_card = ext_card = None
    for f in pc_dir.glob("*.json"):
        with open(f) as fp:
            card = json.load(fp)
        if "summarization" in card.get("prompt_id", ""):
            sum_card = card
        elif "extraction" in card.get("prompt_id", ""):
            ext_card = card
    return sum_card, ext_card


def run_exists(run_id: str) -> bool:
    return (OUTPUT_DIR / "runs" / f"{run_id}.json").exists()


def ollama_chat(model: str, messages: list, temperature: float = 0.0,
                seed: Optional[int] = None, max_tokens: int = 1024) -> dict:
    """Run multi-turn chat via Ollama /api/chat."""
    options = {
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": 0,
        "num_predict": max_tokens,
    }
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    start = time.perf_counter()
    result = llama_runner._ollama_api("/api/chat", payload, timeout=180)
    end = time.perf_counter()

    message = result.get("message", {})
    return {
        "output_text": message.get("content", ""),
        "inference_duration_ms": round((end - start) * 1000, 2),
        "model_reported_duration_ns": result.get("total_duration", 0),
        "eval_count": result.get("eval_count", 0),
        "prompt_eval_count": result.get("prompt_eval_count", 0),
        "done": result.get("done", False),
    }


# ─── Scenario 1: Iterative Refinement ────────────────────────────────────────

def run_refinement_experiment(model: str, abstract: dict, rep: int,
                               temperature: float = 0.0,
                               seed: Optional[int] = None) -> dict:
    """Run a 3-turn iterative refinement extraction."""
    run_id = (f"{model}_multiturn_refinement_{abstract['id']}"
              f"_C1_fixed_seed_rep{rep}").replace(":", "_")

    if run_exists(run_id):
        return None

    inference_params = llama_runner.get_inference_params(
        temperature=temperature, seed=seed, max_tokens=1024,
    )
    model_info = llama_runner.get_model_info(model)

    # Build the full multi-turn prompt text for hashing (reproducibility)
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
        model_name=model_info.get("model_name", model),
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
        messages = [{"role": "system", "content": REFINEMENT_SYSTEM_PROMPT}]

        # Turn 1: Initial extraction
        messages.append({
            "role": "user",
            "content": f"{REFINEMENT_TURN1_PROMPT}\n\n{abstract['text']}"
        })
        turn1 = ollama_chat(model, messages, temperature, seed)
        messages.append({"role": "assistant", "content": turn1["output_text"]})

        # Turn 2: Review and correct
        messages.append({"role": "user", "content": REFINEMENT_TURN2_PROMPT})
        turn2 = ollama_chat(model, messages, temperature, seed)
        messages.append({"role": "assistant", "content": turn2["output_text"]})

        # Turn 3: Final verification
        messages.append({"role": "user", "content": REFINEMENT_TURN3_PROMPT})
        turn3 = ollama_chat(model, messages, temperature, seed)

        # The final output is turn 3; log all turns in system_logs
        output_text = turn3["output_text"]
        system_logs = json.dumps({
            "turn1": {"output": turn1["output_text"],
                      "duration_ms": turn1["inference_duration_ms"]},
            "turn2": {"output": turn2["output_text"],
                      "duration_ms": turn2["inference_duration_ms"]},
            "turn3": {"output": turn3["output_text"],
                      "duration_ms": turn3["inference_duration_ms"]},
            "total_turns": 3,
            "full_conversation": [m for m in messages] + [
                {"role": "assistant", "content": turn3["output_text"]}
            ],
        }, default=str)
        errors = []
    except Exception as e:
        output_text = ""
        system_logs = ""
        errors = [str(e)]

    logger.log_output(output_text=output_text, system_logs=system_logs, errors=errors)
    logger.save()

    rc = RunCard(str(OUTPUT_DIR / "run_cards"))
    run_card = rc.create_from_run(logger.run_data,
                                   prompt_card_ref="prompt_card_multiturn_refinement_v1_0.json")
    rc.save(run_card)

    return logger.run_data


# ─── Scenario 2: RAG-enhanced Extraction ─────────────────────────────────────

def run_rag_experiment(model: str, abstract: dict, rep: int,
                       temperature: float = 0.0,
                       seed: Optional[int] = None) -> dict:
    """Run RAG-enhanced extraction with retrieved context."""
    run_id = (f"{model}_rag_extraction_{abstract['id']}"
              f"_C1_fixed_seed_rep{rep}").replace(":", "_")

    if run_exists(run_id):
        return None

    abstract_id = abstract["id"]
    retrieved_context = RETRIEVED_CONTEXTS.get(abstract_id, "No additional context available.")

    inference_params = llama_runner.get_inference_params(
        temperature=temperature, seed=seed, max_tokens=1024,
    )
    model_info = llama_runner.get_model_info(model)

    # Construct the user message with both abstract and context
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
        model_name=model_info.get("model_name", model),
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
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"{RAG_PROMPT}\n\n{user_content}"},
        ]
        result = ollama_chat(model, messages, temperature, seed)
        output_text = result["output_text"]
        system_logs = json.dumps(
            {k: v for k, v in result.items() if k != "output_text"}, default=str
        )
        errors = []
    except Exception as e:
        output_text = ""
        system_logs = ""
        errors = [str(e)]

    logger.log_output(output_text=output_text, system_logs=system_logs, errors=errors)
    logger.save()

    rc = RunCard(str(OUTPUT_DIR / "run_cards"))
    run_card = rc.create_from_run(logger.run_data,
                                   prompt_card_ref="prompt_card_rag_extraction_v1_0.json")
    rc.save(run_card)

    return logger.run_data


def create_multiturn_prompt_cards():
    """Create prompt cards for the multi-turn scenarios."""
    pc = PromptCard(str(OUTPUT_DIR / "prompt_cards"))

    # Refinement prompt card
    ref_card = pc.create(
        prompt_id="multiturn_refinement",
        prompt_text=(f"[SYSTEM] {REFINEMENT_SYSTEM_PROMPT}\n"
                     f"[TURN1] {REFINEMENT_TURN1_PROMPT}\n"
                     f"[TURN2] {REFINEMENT_TURN2_PROMPT}\n"
                     f"[TURN3] {REFINEMENT_TURN3_PROMPT}"),
        task_category="iterative_structured_extraction",
        objective="Extract structured JSON via 3-turn iterative refinement conversation.",
        designed_by=RESEARCHER_ID,
        version="1.0",
        assumptions=[
            "Model can maintain conversation context across turns",
            "Iterative refinement improves extraction quality",
            "Output at each turn should be valid JSON",
        ],
        limitations=[
            "Fixed refinement strategy (no adaptive feedback)",
            "3 turns may be insufficient for complex abstracts",
        ],
        target_models=MULTITURN_MODELS,
        expected_output_format="JSON object with 5 string fields",
        interaction_regime="multi-turn",
    )
    pc.save(ref_card)

    # RAG prompt card
    rag_card = pc.create(
        prompt_id="rag_extraction",
        prompt_text=RAG_PROMPT,
        task_category="rag_structured_extraction",
        objective="Extract structured JSON using both abstract and retrieved context.",
        designed_by=RESEARCHER_ID,
        version="1.0",
        assumptions=[
            "Retrieved context provides useful enrichment",
            "Model can distinguish primary source from context",
            "Abstract is authoritative; context is supplementary",
        ],
        limitations=[
            "Simulated retrieval (not live search)",
            "Single chunk per abstract",
            "Context quality varies by abstract",
        ],
        target_models=MULTITURN_MODELS,
        expected_output_format="JSON object with 5 string fields",
        interaction_regime="single-turn-with-context",
    )
    pc.save(rag_card)

    print("[OK] Multi-turn prompt cards created")
    return ref_card, rag_card


def _print_progress(run_data, done, total, scenario):
    run_id = run_data.get("run_id", "?")
    duration = run_data.get("execution_duration_ms", 0)
    has_error = len(run_data.get("errors", [])) > 0
    status = "ERR" if has_error else "OK"
    pct = (done / total * 100) if total > 0 else 0
    print(f"    [{status}] ({done}/{total} {pct:.0f}%) {run_id} | {duration:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Multi-turn Experiment Runner")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=["refinement", "rag"],
                        help="Run only this scenario")
    parser.add_argument("--abstracts", type=int, default=N_ABSTRACTS)
    args = parser.parse_args()

    models = [args.model] if args.model else MULTITURN_MODELS
    scenarios = [args.scenario] if args.scenario else ["refinement", "rag"]

    print("=" * 70)
    print("GenAI Reproducibility - Multi-Turn Experiments")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Models: {models}")
    print(f"Scenarios: {scenarios}")
    print(f"Abstracts: {args.abstracts}")
    print("=" * 70)

    # Verify models
    for model in models:
        try:
            info = llama_runner.get_model_info(model)
            if "error" in info:
                print(f"ERROR: {model} not available. Pull it first.")
                sys.exit(1)
            print(f"  [OK] {model}")
        except Exception:
            print(f"ERROR: Cannot reach Ollama for {model}")
            sys.exit(1)

    abstracts = load_abstracts(args.abstracts)
    print(f"\nLoaded {len(abstracts)} abstracts")

    # Create prompt cards
    create_multiturn_prompt_cards()

    all_runs = []

    for model in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model}")
        print("=" * 70)

        if "refinement" in scenarios:
            print(f"\n  --- Scenario: Iterative Refinement (3-turn) ---")
            total = len(abstracts) * N_REPS
            done = 0
            for abstract in abstracts:
                for rep in range(N_REPS):
                    run_data = run_refinement_experiment(
                        model, abstract, rep,
                        temperature=0.0, seed=SEEDS[0],
                    )
                    done += 1
                    if run_data:
                        all_runs.append(run_data)
                        _print_progress(run_data, done, total, "refinement")

        if "rag" in scenarios:
            print(f"\n  --- Scenario: RAG-Enhanced Extraction ---")
            total = len(abstracts) * N_REPS
            done = 0
            for abstract in abstracts:
                for rep in range(N_REPS):
                    run_data = run_rag_experiment(
                        model, abstract, rep,
                        temperature=0.0, seed=SEEDS[0],
                    )
                    done += 1
                    if run_data:
                        all_runs.append(run_data)
                        _print_progress(run_data, done, total, "rag")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {len(all_runs)} new multi-turn runs")
    refinement_count = sum(1 for r in all_runs if "refinement" in r.get("task_id", ""))
    rag_count = sum(1 for r in all_runs if "rag" in r.get("task_id", ""))
    print(f"  Refinement: {refinement_count}")
    print(f"  RAG: {rag_count}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
