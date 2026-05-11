"""Sandboxed pass@1 evaluation for HumanEval completions.

WARNING — Security model:
    Running arbitrary LLM-generated code is INHERENTLY UNSAFE. We mitigate
    this with three layers:

      1. Subprocess isolation: each candidate runs in its own ``python3 -c``
         child process so that ``sys.exit``, ``os._exit``, infinite loops,
         and unhandled exceptions cannot poison the parent.
      2. Wall-clock timeout (default 5 s): SIGKILL the child if it hangs.
      3. Resource caps via ``resource.setrlimit`` (best-effort, POSIX only):
         CPU time, address space, file size, max processes. Any caller
         needing stronger guarantees (network blocking, filesystem
         containment) should run inside Docker/Firecracker — see
         ``REVISION_PLAN.md``. The current threat model assumes benign
         models that may produce buggy or non-terminating code, NOT
         adversarial models trying to escape.

    DO NOT run this against an adversarial model on an untrusted host.

The public API is::

    extract_function_body(prompt, completion, entry_point) -> str
    run_pass_at_1(prompt, completion, test, entry_point, timeout=5)
        -> {"passed": bool, "stdout": str, "stderr": str, "duration_ms": float}
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


# ─── Completion parsing ───────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """If the model wrapped its answer in ``` ... ``` blocks, return the
    contents of the LAST fenced block (most often the final answer)."""
    matches = _FENCE_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text


def extract_function_body(prompt: str, completion: str, entry_point: str) -> str:
    """Build a runnable Python program from the HumanEval prompt + completion.

    Models behave inconsistently:
      * Some return only the body (indented). We append it verbatim.
      * Some return the full ``def <entry_point>(...): body`` again.
      * Some wrap everything in ``` ``` fences.

    Strategy: prefer model-defined function (full def) if found, else
    treat the completion as a body and concatenate after the prompt.
    """
    body = _strip_code_fences(completion)

    # If the completion already defines the function, use it directly
    # alongside imports / helpers from the prompt header.
    def_pattern = re.compile(
        rf"^def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE
    )
    if def_pattern.search(body):
        # Pull the prompt header (everything before the function def) so
        # imports and helpers remain available.
        header = ""
        prompt_def = def_pattern.search(prompt)
        if prompt_def:
            header = prompt[: prompt_def.start()]
        return header + body

    # Otherwise, treat as body to be appended after the prompt's def line.
    # Ensure indentation is at least 4 spaces.
    if body and not body.lstrip().startswith("def "):
        body_lines = []
        for line in body.splitlines():
            if line.strip() and not line.startswith((" ", "\t")):
                body_lines.append("    " + line)
            else:
                body_lines.append(line)
        body = "\n".join(body_lines)
    return prompt + body + "\n"


# ─── Sandboxed execution ──────────────────────────────────────────────────────

_RESOURCE_PRELUDE = """
import sys
try:
    import resource, signal
    # 4 s CPU; wall-clock is enforced by the parent via subprocess timeout.
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    # ~1 GiB address space.
    try:
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024,) * 2)
    except (ValueError, OSError):
        pass
    # No new files larger than 16 MiB.
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024,) * 2)
    except (ValueError, OSError):
        pass
except Exception:
    pass
"""


def _build_program(
    prompt: str, completion: str, test: str, entry_point: str
) -> str:
    """Compose the full program: prelude + candidate + test harness + check."""
    candidate = extract_function_body(prompt, completion, entry_point)
    return (
        _RESOURCE_PRELUDE
        + "\n"
        + candidate
        + "\n\n"
        + test
        + "\n"
        + f"check({entry_point})\n"
        + "print('__OK__')\n"
    )


def run_pass_at_1(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    timeout: float = 5.0,
    python_executable: Optional[str] = None,
) -> dict:
    """Execute a candidate against its HumanEval test harness.

    Returns a dict with::

        {
            "passed":      bool,
            "stdout":      str,
            "stderr":      str,
            "duration_ms": float,
            "timed_out":   bool,
            "error":       Optional[str],
        }
    """
    program = _build_program(prompt, completion, test, entry_point)
    py = python_executable or sys.executable

    started = time.perf_counter()
    timed_out = False
    err_msg = None
    try:
        proc = subprocess.run(
            [py, "-I", "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Empty environment to reduce side effects (TMPDIR is preserved).
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": tempfile.gettempdir(),
                "TMPDIR": tempfile.gettempdir(),
                "PYTHONHASHSEED": "0",
            },
            cwd=tempfile.gettempdir(),
        )
        stdout = proc.stdout
        stderr = proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc = -1
        timed_out = True
        err_msg = f"timeout after {timeout}s"
    except Exception as e:
        stdout, stderr = "", ""
        rc = -1
        err_msg = str(e)

    duration_ms = (time.perf_counter() - started) * 1000.0
    passed = (rc == 0) and ("__OK__" in (stdout or ""))

    return {
        "passed": passed,
        "stdout": (stdout or "")[-2048:],
        "stderr": (stderr or "")[-2048:],
        "return_code": rc,
        "duration_ms": round(duration_ms, 2),
        "timed_out": timed_out,
        "error": err_msg,
    }


# ─── Aggregation ──────────────────────────────────────────────────────────────

def pass_at_1_metric(results: list) -> float:
    """Mean of the boolean ``passed`` field over a list of result dicts."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("passed")) / len(results)
