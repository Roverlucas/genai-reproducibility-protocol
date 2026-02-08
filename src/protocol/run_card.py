"""Run Card generator for the GenAI reproducibility protocol.

A Run Card is a structured record for each experimental run, consolidating
all protocol-required artifacts: prompt, model, parameters, context, output,
and provenance information into a single auditable document.
"""

import json
from pathlib import Path
from typing import Optional


class RunCard:
    """Generates Run Card documentation artifacts from run data."""

    def __init__(self, output_dir: str = "outputs/run_cards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_from_run(self, run_data: dict, prompt_card_ref: str = "") -> dict:
        """Create a Run Card from a completed run record."""
        card = {
            "run_card_version": "1.0",
            # Run identification
            "run_id": run_data.get("run_id"),
            "task_id": run_data.get("task_id"),
            "task_category": run_data.get("task_category"),
            # Prompt reference
            "prompt_card_ref": prompt_card_ref,
            "prompt_hash": run_data.get("prompt_hash"),
            # Model
            "model_name": run_data.get("model_name"),
            "model_version": run_data.get("model_version"),
            "weights_hash": run_data.get("weights_hash", ""),
            "model_source": run_data.get("model_source", ""),
            # Parameters
            "inference_params": run_data.get("inference_params"),
            "params_hash": run_data.get("params_hash"),
            "interaction_regime": run_data.get("interaction_regime"),
            # Context
            "retrieval_context": run_data.get("retrieval_context"),
            # Environment
            "environment_hash": run_data.get("environment_hash"),
            "code_commit": run_data.get("code_commit"),
            "environment_details": {
                "os": run_data.get("environment", {}).get("os"),
                "python_version": run_data.get("environment", {}).get(
                    "python_version", ""
                ).split()[0],
                "architecture": run_data.get("environment", {}).get("architecture"),
            },
            # Execution
            "timestamp_start": run_data.get("timestamp_start"),
            "timestamp_end": run_data.get("timestamp_end"),
            "execution_duration_ms": run_data.get("execution_duration_ms"),
            # Output
            "output_hash": run_data.get("output_hash"),
            "output_metrics": run_data.get("output_metrics"),
            # Overhead
            "logging_overhead_ms": run_data.get("logging_overhead_ms"),
            "storage_kb": run_data.get("storage_kb"),
            # Researcher
            "researcher_id": run_data.get("researcher_id"),
            "affiliation": run_data.get("affiliation"),
            # Errors
            "errors": run_data.get("errors", []),
        }

        return card

    def save(self, card: dict) -> str:
        """Save a Run Card to JSON. Returns filepath."""
        run_id = card.get("run_id", "unknown")
        filepath = self.output_dir / f"run_card_{run_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)
        return str(filepath)
