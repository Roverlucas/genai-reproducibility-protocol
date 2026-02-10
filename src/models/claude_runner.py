"""Claude runner via Anthropic API for the GenAI reproducibility protocol.

Provides a unified interface to run Claude models via the Anthropic API,
with full parameter logging and integration with the protocol's RunLogger.
"""

import json
import os
import time
import urllib.request
from typing import Optional


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"


def _get_api_key() -> str:
    """Get the Anthropic API key from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return key


def get_model_info(model: str = "claude-sonnet-4-5-20250929") -> dict:
    """Get model metadata for Claude."""
    return {
        "model_name": model,
        "model_version": "api-managed",
        "weights_hash": "proprietary-not-available",
        "model_source": "anthropic-api",
        "license": "proprietary",
    }


def run_inference(
    prompt: str,
    input_text: str = "",
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    max_tokens: int = 1024,
    system_prompt: str = "",
    timeout: int = 60,
) -> dict:
    """Run a single inference with Claude via the Anthropic API.

    Returns a dict with output_text, duration_ms, and metadata.
    """
    api_key = _get_api_key()

    user_content = prompt
    if input_text:
        user_content = f"{prompt}\n\n{input_text}"

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_content}],
    }

    if temperature > 0.0:
        payload["temperature"] = temperature
    if top_p < 1.0:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if system_prompt:
        payload["system"] = system_prompt

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(ANTHROPIC_API_URL, data=data, headers=headers)

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


def get_inference_params(
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    max_tokens: int = 1024,
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
    return params
