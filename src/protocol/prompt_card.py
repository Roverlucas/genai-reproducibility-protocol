"""Prompt Card generator for the GenAI reproducibility protocol.

A Prompt Card is a structured documentation of a prompt, analogous to
Model Cards for models and Datasheets for datasets. It records the
prompt's objective, assumptions, structure, versions, and usage context.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .hasher import hash_text


class PromptCard:
    """Generates and manages Prompt Card documentation artifacts."""

    def __init__(self, output_dir: str = "outputs/prompt_cards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        prompt_id: str,
        prompt_text: str,
        task_category: str,
        objective: str,
        designed_by: str,
        version: str = "1.0",
        assumptions: Optional[list] = None,
        limitations: Optional[list] = None,
        few_shot_examples: Optional[list] = None,
        system_instructions: str = "",
        interaction_regime: str = "single-turn",
        target_models: Optional[list] = None,
        expected_output_format: str = "",
        change_log: Optional[list] = None,
    ) -> dict:
        """Create a Prompt Card with all required fields."""
        prompt_hash = hash_text(prompt_text)

        card = {
            "prompt_card_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            # Identification
            "prompt_id": prompt_id,
            "prompt_hash": prompt_hash,
            "version": version,
            # Content
            "prompt_text": prompt_text,
            "system_instructions": system_instructions,
            "few_shot_examples": few_shot_examples or [],
            # Documentation
            "task_category": task_category,
            "objective": objective,
            "designed_by": designed_by,
            "assumptions": assumptions or [],
            "limitations": limitations or [],
            # Usage context
            "interaction_regime": interaction_regime,
            "target_models": target_models or [],
            "expected_output_format": expected_output_format,
            # Change tracking
            "change_log": change_log or [
                {
                    "version": version,
                    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "description": "Initial version",
                }
            ],
        }

        return card

    def save(self, card: dict) -> str:
        """Save a Prompt Card to JSON. Returns filepath."""
        prompt_id = card.get("prompt_id", "unknown")
        version = card.get("version", "1.0").replace(".", "_")
        filepath = self.output_dir / f"prompt_card_{prompt_id}_v{version}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)
        return str(filepath)
