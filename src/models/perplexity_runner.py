"""Perplexity runner via Perplexity API for the GenAI reproducibility protocol.

Provides a unified interface to run Perplexity models via the Perplexity API,
with full parameter logging and integration with the protocol's RunLogger.

Note: Perplexity's sonar model is an online model with search augmentation.
This makes it an interesting case study for reproducibility since the model
may incorporate real-time web data into responses.
"""

import json
import os
import time
import urllib.request
from typing import Optional


PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


def _get_api_key() -> str:
    """Get the Perplexity API key from environment."""
    key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set")
    return key


def get_model_info(model: str = "sonar") -> dict:
    """Get model metadata for Perplexity."""
    return {
        "model_name": model,
        "model_version": "api-managed",
        "weights_hash": "proprietary-not-available",
        "model_source": "perplexity-api",
        "license": "proprietary",
        "note": "Online model with search augmentation",
    }


def run_inference(
    prompt: str,
    input_text: str = "",
    model: str = "sonar",
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    max_tokens: int = 1024,
    system_prompt: str = "",
    timeout: int = 120,
) -> dict:
    """Run a single inference with Perplexity via the Perplexity API.

    Returns a dict with output_text, duration_ms, and metadata.
    """
    api_key = _get_api_key()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = prompt
    if input_text:
        user_content = f"{prompt}\n\n{input_text}"
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else 0.0,
        "top_p": top_p,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(PERPLEXITY_API_URL, data=data, headers=headers)

    start_time = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    end_time = time.perf_counter()

    inference_duration_ms = (end_time - start_time) * 1000

    content_text = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})

    return {
        "output_text": content_text,
        "inference_duration_ms": round(inference_duration_ms, 2),
        "finish_reason": result["choices"][0].get("finish_reason", ""),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "model_id_returned": result.get("model", ""),
        "response_id": result.get("id", ""),
    }


def get_inference_params(
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    max_tokens: int = 1024,
) -> dict:
    """Build a standardized inference parameters dict for logging."""
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "decoding_strategy": "greedy" if temperature == 0.0 else "sampling",
    }
    if seed is not None:
        params["seed"] = seed
    return params
