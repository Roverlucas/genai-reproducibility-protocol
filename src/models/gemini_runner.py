"""Gemini runner via Google AI Studio REST API for the GenAI reproducibility protocol.

Provides a unified interface to run Gemini models via the Google
Generative Language API, with full parameter logging and integration
with the protocol's RunLogger.

Uses urllib only (no external dependencies), following the same pattern
as claude_runner.py.
"""

import json
import os
import time
import urllib.request
from typing import Optional


GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
)


def _get_api_key() -> str:
    """Get the Gemini API key from environment."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return key


def get_model_info(model: str = "gemini-2.5-pro") -> dict:
    """Get model metadata for Gemini."""
    return {
        "model_name": model,
        "model_version": "api-managed",
        "weights_hash": "proprietary-not-available",
        "model_source": "google-ai-studio",
        "license": "proprietary",
    }


def run_inference(
    prompt: str,
    input_text: str = "",
    model: str = "gemini-2.5-pro",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    max_tokens: int = 8192,
    system_prompt: str = "",
    seed: Optional[int] = None,
    timeout: int = 60,
) -> dict:
    """Run a single inference with Gemini via the Google AI Studio API.

    Note: Gemini 2.5 Pro is a "thinking" model that uses internal reasoning
    tokens counted against maxOutputTokens. We default to 8192 to ensure
    enough room for both thinking and actual output.

    Returns a dict with output_text, duration_ms, and metadata.
    """
    api_key = _get_api_key()

    user_content = prompt
    if input_text:
        user_content = f"{prompt}\n\n{input_text}"

    contents = [{"role": "user", "parts": [{"text": user_content}]}]

    generation_config = {
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }
    if top_p < 1.0:
        generation_config["topP"] = top_p
    if top_k is not None:
        generation_config["topK"] = top_k
    if seed is not None:
        generation_config["seed"] = seed

    payload = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    url = f"{GEMINI_API_URL}/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)

    start_time = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    end_time = time.perf_counter()

    inference_duration_ms = (end_time - start_time) * 1000

    # Extract text from response
    content_text = ""
    candidates = result.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            content_text += part.get("text", "")

    finish_reason = ""
    if candidates:
        finish_reason = candidates[0].get("finishReason", "")

    usage = result.get("usageMetadata", {})

    return {
        "output_text": content_text,
        "inference_duration_ms": round(inference_duration_ms, 2),
        "finish_reason": finish_reason,
        "usage": {
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
            "thoughts_tokens": usage.get("thoughtsTokenCount", 0),
        },
        "model_id_returned": result.get("modelVersion", ""),
        "response_id": result.get("responseId", ""),
    }


def run_multiturn_inference(
    contents: list,
    model: str = "gemini-2.5-pro",
    system_prompt: str = "",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    max_tokens: int = 8192,
    seed: Optional[int] = None,
    timeout: int = 90,
) -> dict:
    """Call Gemini API with a multi-turn contents array.

    Args:
        contents: List of {"role": "user"|"model", "parts": [{"text": str}]}
                  dicts. System instruction is separate.
        model: Gemini model ID.
        system_prompt: Optional system instruction.
        temperature: Sampling temperature (0.0 = greedy).
        max_tokens: Maximum output tokens.
        seed: Optional seed for reproducibility.
        timeout: Request timeout in seconds.

    Returns:
        Dict with output_text, inference_duration_ms, usage, etc.
    """
    api_key = _get_api_key()

    generation_config = {
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }
    if top_p < 1.0:
        generation_config["topP"] = top_p
    if top_k is not None:
        generation_config["topK"] = top_k
    if seed is not None:
        generation_config["seed"] = seed

    payload = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    url = f"{GEMINI_API_URL}/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)

    start_time = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    end_time = time.perf_counter()

    inference_duration_ms = (end_time - start_time) * 1000

    content_text = ""
    candidates = result.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            content_text += part.get("text", "")

    finish_reason = ""
    if candidates:
        finish_reason = candidates[0].get("finishReason", "")

    usage = result.get("usageMetadata", {})

    return {
        "output_text": content_text,
        "inference_duration_ms": round(inference_duration_ms, 2),
        "finish_reason": finish_reason,
        "usage": {
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
            "thoughts_tokens": usage.get("thoughtsTokenCount", 0),
        },
        "model_id_returned": result.get("modelVersion", ""),
        "response_id": result.get("responseId", ""),
    }


def get_inference_params(
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    max_tokens: int = 1024,
    seed: Optional[int] = None,
) -> dict:
    """Build a standardized inference parameters dict for logging."""
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "decoding_strategy": "greedy" if temperature == 0.0 else "sampling",
    }
    if top_k is not None:
        params["top_k"] = top_k
    if seed is not None:
        params["seed"] = seed
    return params
