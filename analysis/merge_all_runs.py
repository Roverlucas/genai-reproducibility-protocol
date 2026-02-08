#!/usr/bin/env python3
"""Merge all run records (LLaMA + GPT-4) into a single all_runs.json."""

import json
from pathlib import Path

RUNS_DIR = Path(__file__).parent.parent / "outputs" / "runs"
OUTPUT = Path(__file__).parent.parent / "outputs" / "all_runs.json"


def main():
    all_runs = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        with open(f) as fp:
            run = json.load(fp)
            all_runs.append(run)

    with open(OUTPUT, "w") as fp:
        json.dump(all_runs, fp, indent=2, ensure_ascii=False)

    # Count by model
    llama = sum(1 for r in all_runs if "llama" in r.get("model_name", "").lower())
    gpt4 = sum(1 for r in all_runs if "gpt" in r.get("model_name", "").lower())
    print(f"Merged {len(all_runs)} runs: LLaMA={llama}, GPT-4={gpt4}")


if __name__ == "__main__":
    main()
