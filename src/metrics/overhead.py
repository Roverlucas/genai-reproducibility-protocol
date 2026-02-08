"""Overhead metrics for measuring the cost of applying the protocol.

Calculates time overhead and storage overhead introduced by the
systematic logging, versioning, and provenance generation.
"""

import os
from pathlib import Path
from typing import List

import numpy as np


def compute_logging_overhead(run_records: List[dict]) -> dict:
    """Compute logging overhead statistics from a list of run records."""
    overheads = []
    for r in run_records:
        overhead = r.get("logging_overhead_ms", 0.0)
        overheads.append(overhead)

    if not overheads:
        return {"mean_ms": 0.0, "std_ms": 0.0, "total_ms": 0.0}

    return {
        "mean_ms": float(np.mean(overheads)),
        "std_ms": float(np.std(overheads)),
        "min_ms": float(np.min(overheads)),
        "max_ms": float(np.max(overheads)),
        "total_ms": float(np.sum(overheads)),
        "n_runs": len(overheads),
    }


def compute_storage_overhead(run_records: List[dict]) -> dict:
    """Compute storage overhead statistics from a list of run records."""
    sizes = []
    for r in run_records:
        size = r.get("storage_kb", 0.0)
        sizes.append(size)

    if not sizes:
        return {"mean_kb": 0.0, "total_kb": 0.0}

    return {
        "mean_kb": float(np.mean(sizes)),
        "std_kb": float(np.std(sizes)),
        "min_kb": float(np.min(sizes)),
        "max_kb": float(np.max(sizes)),
        "total_kb": float(np.sum(sizes)),
        "n_runs": len(sizes),
    }


def compute_overhead_ratio(run_records: List[dict]) -> dict:
    """Compute the ratio of logging overhead to total execution time."""
    ratios = []
    for r in run_records:
        overhead = r.get("logging_overhead_ms", 0.0)
        total = r.get("execution_duration_ms", 0.0)
        if total > 0:
            ratios.append(overhead / total)

    if not ratios:
        return {"mean_ratio": 0.0, "mean_percent": 0.0}

    return {
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "mean_percent": float(np.mean(ratios)) * 100,
        "max_percent": float(np.max(ratios)) * 100,
        "n_runs": len(ratios),
    }


def compute_directory_size(directory: str) -> dict:
    """Compute total size of all files in a directory tree."""
    total_bytes = 0
    file_count = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)
                file_count += 1

    return {
        "total_kb": total_bytes / 1024,
        "total_mb": total_bytes / (1024 * 1024),
        "file_count": file_count,
    }
