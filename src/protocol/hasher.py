"""Cryptographic hashing utilities for the GenAI reproducibility protocol."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def hash_text(text: str) -> str:
    """Generate SHA-256 hash of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(filepath: str) -> str:
    """Generate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def hash_dict(d: dict) -> str:
    """Generate SHA-256 hash of a dictionary (deterministic JSON serialization)."""
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=True)
    return hash_text(canonical)


def get_environment_hash() -> str:
    """Generate a hash representing the current execution environment.

    Returns:
        SHA-256 hex digest of the environment information dictionary.
    """
    env_info: dict[str, str] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": sys.version,
        "processor": platform.processor(),
    }
    return hash_dict(env_info)


def get_environment_metadata(anonymize_hostname: bool = False) -> dict:
    """Collect detailed environment metadata.

    Args:
        anonymize_hostname: If True, replaces hostname with its SHA-256 hash.
            Recommended for privacy-sensitive deployments where the hostname
            may reveal institutional information.
    """
    hostname = platform.node()
    if anonymize_hostname:
        hostname = hash_text(hostname)[:16]  # truncated hash
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "hostname": hostname,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_code_commit() -> str:
    """Get the current git commit hash, or 'no-git' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "no-git-repo"
