"""GPT-4 runner via OpenAI API for the GenAI reproducibility protocol.

Provides a unified interface to run GPT-4 via the OpenAI API,
with full parameter logging and integration with the protocol's RunLogger.
"""

import os
import time
from typing import Optional

from openai import OpenAI


def get_client() -> OpenAI:
    """Get an OpenAI client using the OPENAI_API_KEY env variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def get_model_info(model: str = "gpt-4") -> dict:
    """Get model metadata for GPT-4.

    Note: The API may return a specific version (e.g., 'gpt-4-0613')
    in the response's model field, which is captured in run_inference()
    as 'model_id_returned'.
    """
    return {
        "model_name": model,
        "model_version": "api-managed",
        "weights_hash": "proprietary-not-available",
        "model_source": "openai-api",
        "license": "proprietary",
    }


def run_inference(
    prompt: str,
    input_text: str = "",
    model: str = "gpt-4",
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    max_tokens: int = 1024,
    system_prompt: str = "",
) -> dict:
    """Run a single inference with GPT-4 via OpenAI API.

    Returns a dict with output_text, duration_ms, and metadata.
    """
    client = get_client()

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = prompt
    if input_text:
        user_content = f"{prompt}\n\n{input_text}"
    messages.append({"role": "user", "content": user_content})

    # Build API call kwargs
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        kwargs["seed"] = seed

    start_time = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    end_time = time.perf_counter()

    inference_duration_ms = (end_time - start_time) * 1000
    choice = response.choices[0]

    return {
        "output_text": choice.message.content or "",
        "inference_duration_ms": round(inference_duration_ms, 2),
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        },
        "model_id_returned": response.model,
        "system_fingerprint": response.system_fingerprint or "",
        "response_id": response.id,
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
