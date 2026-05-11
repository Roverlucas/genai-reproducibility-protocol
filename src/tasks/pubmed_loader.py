"""PubMed PM2.5 / respiratory abstracts loader (Task 14, T14 light).

Source corpus: paper-irmão (RSM submission) at
``/Users/lucasrover/llm-evidence-synthesis-reproducibility/data/corpus/``
which provides 500 PubMed records labelled by inclusion / exclusion
heuristic. We sample 10 abstracts focused on PM2.5 and respiratory
endpoints to constitute the non-AI/ML domain probe requested by the
editor (decision D1, item T14).

A snapshot of the chosen 10 is cached in this repo at
``data/inputs/revision/pubmed_pm25_t14.json`` for reproducibility — once
written, the loader uses it offline. To regenerate, delete the cache
file or call ``select_and_cache()``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

DEFAULT_CACHE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "inputs"
    / "revision"
    / "pubmed_pm25_t14.json"
)

DEFAULT_SOURCE = Path(
    "/Users/lucasrover/llm-evidence-synthesis-reproducibility/data/corpus/"
    "corpus_500.json"
)


# ─── Selection ────────────────────────────────────────────────────────────────

def _is_pm25_respiratory(rec: dict) -> bool:
    """Heuristic: prefer abstracts mentioning PM2.5 AND a respiratory endpoint."""
    text = (rec.get("abstract") or "") + " " + (rec.get("title") or "")
    text_lc = text.lower()
    pm25 = "pm2.5" in text_lc or "pm 2.5" in text_lc or "fine particulate" in text_lc
    resp = any(
        kw in text_lc
        for kw in [
            "asthma",
            "copd",
            "bronchi",
            "lung",
            "respiratory",
            "pulmonary",
            "wheez",
        ]
    )
    return pm25 and resp


def select_pm25_abstracts(
    corpus: list, n: int = 10, seed: int = 42, min_chars: int = 400
) -> list:
    """Sample ``n`` abstracts that mention PM2.5 + a respiratory endpoint.

    Filters out very short records (likely metadata-only) via ``min_chars``.
    Reproducible via fixed ``seed``.
    """
    pool = [
        r for r in corpus
        if _is_pm25_respiratory(r) and len((r.get("abstract") or "")) >= min_chars
    ]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def _normalize(rec: dict, idx: int) -> dict:
    """Map a corpus record to the schema used by the existing pipeline.

    Existing abstracts.json uses {id, source, doi, text}. We mirror that.
    """
    pmid = rec.get("pmid") or ""
    journal = rec.get("journal") or ""
    year = rec.get("year") or ""
    title = (rec.get("title") or "").rstrip(".")
    source = f"PubMed:{pmid} | {title} | {journal} {year}".strip()
    return {
        "id": f"pubmed_{idx:03d}",
        "source": source,
        "doi": rec.get("doi") or "",
        "text": rec.get("abstract") or "",
        "pmid": pmid,
        "corpus_id": rec.get("corpus_id") or "",
    }


# ─── Caching ──────────────────────────────────────────────────────────────────

def select_and_cache(
    source_path: Path = DEFAULT_SOURCE,
    cache_path: Path = DEFAULT_CACHE,
    n: int = 10,
    seed: int = 42,
) -> list:
    """Read the sibling-project corpus and emit a 10-abstract cache file.

    Run once interactively to generate ``cache_path``. Subsequent calls
    to ``load_pubmed_pm25`` will use the cached file deterministically.
    """
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source corpus not found at {source_path}. "
            f"Either symlink the paper-irmão corpus or pass an explicit path."
        )
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    corpus = data["corpus"] if isinstance(data, dict) else data
    chosen = select_pm25_abstracts(corpus, n=n, seed=seed)
    out = [_normalize(rec, i + 1) for i, rec in enumerate(chosen)]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "source": str(source_path),
                    "selection_seed": seed,
                    "n": n,
                    "selection_method": "PM2.5+respiratory keyword filter",
                },
                "abstracts": out,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return out


def load_pubmed_pm25(
    cache_path: Path = DEFAULT_CACHE,
    source_path: Path = DEFAULT_SOURCE,
    auto_select: bool = True,
    n: int = 10,
    seed: int = 42,
) -> list:
    """Return the 10 PM2.5 abstracts.

    Resolution:
        1. Use cached file if it exists.
        2. If ``auto_select=True``, sample from the sibling corpus and
           write the cache.
        3. Else raise ``FileNotFoundError``.
    """
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["abstracts"]
    if auto_select:
        return select_and_cache(
            source_path=source_path, cache_path=cache_path, n=n, seed=seed
        )
    raise FileNotFoundError(f"No cache at {cache_path} and auto_select disabled.")
