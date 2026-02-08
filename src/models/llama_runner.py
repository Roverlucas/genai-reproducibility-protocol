"""LLaMA 3 runner via Ollama for the GenAI reproducibility protocol.

Provides a unified interface to run LLaMA 3 locally through Ollama,
with full support for seed control, parameter logging, and integration
with the protocol's RunLogger.
"""

import json
import subprocess
import time
from typing import Optional

import urllib.request


OLLAMA_API_URL = "http://localhost:11434"


def _ollama_api(endpoint: str, payload: dict, timeout: int = 120) -> dict:
    """Make a POST request to the Ollama REST API."""
    url = f"{OLLAMA_API_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ollama_get(endpoint: str, timeout: int = 10) -> dict:
    """Make a GET request to the Ollama REST API."""
    url = f"{OLLAMA_API_URL}{endpoint}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_ollama_version() -> str:
    """Query the Ollama server version."""
    try:
        result = _ollama_get("/api/version")
        return result.get("version", "unknown")
    except Exception:
        return "unknown"


def get_model_info(model: str = "llama3:8b") -> dict:
    """Get model metadata from Ollama, including digest and server version."""
    try:
        result = _ollama_api("/api/show", {"name": model})
        ollama_ver = get_ollama_version()
        return {
            "model_name": model,
            "model_version": result.get("details", {}).get("parameter_size", ""),
            "family": result.get("details", {}).get("family", ""),
            "format": result.get("details", {}).get("format", ""),
            "quantization": result.get("details", {}).get("quantization_level", ""),
            "weights_hash": result.get("digest", ""),
            "model_source": "ollama-local",
            "ollama_version": ollama_ver,
        }
    except Exception as e:
        return {"model_name": model, "error": str(e), "model_source": "ollama-local"}


def run_inference(
    prompt: str,
    input_text: str = "",
    model: str = "llama3:8b",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: Optional[int] = None,
    max_tokens: int = 1024,
    system_prompt: str = "",
    timeout: int = 180,
) -> dict:
    """Run a single inference with LLaMA 3 via Ollama.

    Returns a dict with output_text, duration_ms, and metadata.
    """
    # Build the full prompt
    full_prompt = prompt
    if input_text:
        full_prompt = f"{prompt}\n\n{input_text}"

    # Build options
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_predict": max_tokens,
    }
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": options,
    }
    if system_prompt:
        payload["system"] = system_prompt

    start_time = time.perf_counter()
    result = _ollama_api("/api/generate", payload, timeout=timeout)
    end_time = time.perf_counter()

    inference_duration_ms = (end_time - start_time) * 1000

    return {
        "output_text": result.get("response", ""),
        "inference_duration_ms": round(inference_duration_ms, 2),
        "model_reported_duration_ns": result.get("total_duration", 0),
        "eval_count": result.get("eval_count", 0),
        "prompt_eval_count": result.get("prompt_eval_count", 0),
        "done": result.get("done", False),
        "done_reason": result.get("done_reason", ""),
    }


def get_inference_params(
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: Optional[int] = None,
    max_tokens: int = 1024,
) -> dict:
    """Build a standardized inference parameters dict for logging."""
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "decoding_strategy": "greedy" if temperature == 0.0 else "sampling",
    }
    if seed is not None:
        params["seed"] = seed
    return params
