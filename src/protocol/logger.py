"""Run logger for the GenAI reproducibility protocol.

Implements systematic logging of every experimental run, capturing all
artifacts required by the minimal publishable checklist.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .hasher import (
    get_code_commit,
    get_environment_hash,
    get_environment_metadata,
    hash_dict,
    hash_text,
)


class RunLogger:
    """Logs a single experimental run with all protocol-required fields."""

    def __init__(self, output_dir: str = "outputs/runs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_data = {}
        self._start_time = None
        self._end_time = None
        self._overhead_start = None
        self._logging_overhead_ms = 0.0

    def start_run(
        self,
        run_id: str,
        task_id: str,
        task_category: str,
        prompt_text: str,
        model_name: str,
        model_version: str,
        inference_params: dict,
        researcher_id: str = "anonymous",
        affiliation: str = "",
        input_text: str = "",
        retrieval_context: Optional[dict] = None,
        weights_hash: str = "",
        model_source: str = "",
        interaction_regime: str = "single-turn",
    ) -> dict:
        """Initialize a run record with all pre-execution metadata."""
        self._overhead_start = time.perf_counter()
        self._start_time = datetime.now(timezone.utc)

        prompt_hash = hash_text(prompt_text)
        input_hash = hash_text(input_text) if input_text else ""
        params_hash = hash_dict(inference_params)
        env_metadata = get_environment_metadata()
        env_hash = get_environment_hash()
        code_commit = get_code_commit()

        self._run_data = {
            # Run identification
            "run_id": run_id,
            "task_id": task_id,
            "task_category": task_category,
            "interaction_regime": interaction_regime,
            # Prompt artifact
            "prompt_hash": prompt_hash,
            "prompt_text": prompt_text,
            # Input data
            "input_text": input_text,
            "input_hash": input_hash,
            # Model artifact
            "model_name": model_name,
            "model_version": model_version,
            "weights_hash": weights_hash,
            "model_source": model_source,
            # Inference parameters artifact
            "inference_params": inference_params,
            "params_hash": params_hash,
            # Retrieval context (if RAG)
            "retrieval_context": retrieval_context,
            # Environment
            "environment": env_metadata,
            "environment_hash": env_hash,
            "code_commit": code_commit,
            # Researcher agent
            "researcher_id": researcher_id,
            "affiliation": affiliation,
            # Timestamps
            "timestamp_start": self._start_time.isoformat(),
            "timestamp_end": None,
            # Output (to be filled)
            "output_text": None,
            "output_metrics": {},
            # Execution metadata
            "execution_duration_ms": None,
            "logging_overhead_ms": None,
            "system_logs": "",
            "errors": [],
        }

        overhead_end = time.perf_counter()
        self._logging_overhead_ms += (overhead_end - self._overhead_start) * 1000

        return self._run_data

    def log_output(
        self,
        output_text: str,
        metrics: Optional[dict] = None,
        system_logs: str = "",
        errors: Optional[list] = None,
    ) -> dict:
        """Record the output and post-execution data for a run."""
        overhead_start = time.perf_counter()
        self._end_time = datetime.now(timezone.utc)

        self._run_data["output_text"] = output_text
        self._run_data["output_hash"] = hash_text(output_text)
        self._run_data["output_metrics"] = metrics or {}
        self._run_data["timestamp_end"] = self._end_time.isoformat()
        self._run_data["system_logs"] = system_logs
        self._run_data["errors"] = errors or []

        if self._start_time and self._end_time:
            duration = (self._end_time - self._start_time).total_seconds() * 1000
            self._run_data["execution_duration_ms"] = round(duration, 2)

        overhead_end = time.perf_counter()
        self._logging_overhead_ms += (overhead_end - overhead_start) * 1000
        self._run_data["logging_overhead_ms"] = round(self._logging_overhead_ms, 2)

        return self._run_data

    def save(self) -> str:
        """Save the run record to a JSON file. Returns the filepath."""
        run_id = self._run_data.get("run_id", "unknown")
        filepath = self.output_dir / f"{run_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._run_data, f, indent=2, ensure_ascii=False)

        # Also compute and store the file size for overhead metrics
        file_size_kb = os.path.getsize(filepath) / 1024
        self._run_data["storage_kb"] = round(file_size_kb, 2)

        # Re-save with storage info
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._run_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    @property
    def run_data(self) -> dict:
        return self._run_data

    @property
    def logging_overhead_ms(self) -> float:
        return self._logging_overhead_ms
