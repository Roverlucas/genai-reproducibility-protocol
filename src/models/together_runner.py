"""Together AI runner for the GenAI reproducibility protocol.

Provides a unified interface to run LLaMA 3 8B via Together AI's
OpenAI-compatible API, with full parameter logging and integration
with the protocol's RunLogger.

Uses urllib only (no external dependencies), following the same pattern
as claude_runner.py and gemini_runner.py.
"""

import json
import os
import time
import urllib.request
from typing import Optional


TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Together AI serves LLaMA 3 8B Instruct as an INT4-quantized "Lite" endpoint.
# This is the same architecture as the local Ollama llama3:8b model, enabling
# direct comparison of local vs cloud-served inference on the same model family.
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"


def _get_api_key() -> str:
    """Get the Together AI API key from environment."""
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    return key


def get_model_info(model: str = DEFAULT_MODEL) -> dict:
    """Get model metadata for Together AI-served LLaMA 3."""
    return {
        "model_name": model,
        "model_version": "api-managed",
        "weights_hash": "open-weight-cloud-served",
        "model_source": "together-ai",
        "license": "llama3-community",
        "quantization": "INT4 (Together AI Lite endpoint)",
        "deployment": "cloud-api",
    }


def run_inference(
    prompt: str,
    input_text: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    max_tokens: int = 1024,
    seed: Optional[int] = None,
    system_prompt: str = "",
    timeout: int = 60,
) -> dict:
    """Run a single inference with LLaMA 3 via Together AI.

    Uses OpenAI-compatible chat completions API format.

    Returns a dict with output_text, duration_ms, and metadata.
    """
    api_key = _get_api_key()

    user_content = prompt
    if input_text:
        user_content = f"{prompt}\n\n{input_text}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if top_p < 1.0:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if seed is not None:
        payload["seed"] = seed

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "genai-reproducibility-protocol/1.0",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(TOGETHER_API_URL, data=data, headers=headers)

    start_time = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    end_time = time.perf_counter()

    inference_duration_ms = (end_time - start_time) * 1000

    # Extract text from OpenAI-compatible response
    content_text = ""
    choices = result.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content_text = message.get("content", "")

    finish_reason = ""
    if choices:
        finish_reason = choices[0].get("finish_reason", "")

    usage = result.get("usage", {})

    return {
        "output_text": content_text,
        "inference_duration_ms": round(inference_duration_ms, 2),
        "finish_reason": finish_reason,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
        "model_id_returned": result.get("model", ""),
        "response_id": result.get("id", ""),
        "system_fingerprint": result.get("system_fingerprint", ""),
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
