# Same Prompt, Different Answer

**Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility**

Lucas Rover, Hugo Valadares Siqueira, Eduardo Tadeu Bacalhau, Anibal Tavares de Azevedo & Yara de Souza Tadano

UTFPR --- Universidade Tecnológica Federal do Paraná

**Status:** Submitted to *Nature Communications* (March 2026) | Major Revision (May 2026) | **Revision package finalised 2026-05-13** — awaiting coauthor sign-off and MTS upload

[![Figshare DOI](https://img.shields.io/badge/Figshare-10.6084%2Fm9.figshare.31653373-blue)](https://doi.org/10.6084/m9.figshare.31653373)
[![OSF (companion)](https://img.shields.io/badge/OSF%20(companion)-10.17605%2FOSF.IO%2FVR934-blue)](https://doi.org/10.17605/OSF.IO/VR934)
[![Release tag](https://img.shields.io/badge/Tag-v1.1-green)](https://github.com/Roverlucas/genai-reproducibility-protocol/tree/v1.1)
[![License: code](https://img.shields.io/badge/Code-MIT-yellow)](LICENSE)
[![License: data](https://img.shields.io/badge/Data-CC--BY--4.0-orange)](LICENSE-DATA)

## Overview

This repository contains the reference implementation, experimental data, analysis scripts, and manuscript for a study demonstrating that API-served large language models fail to reproduce their own outputs under documented "deterministic" settings. We provide a lightweight provenance protocol grounded in W3C PROV that makes this invisible variation visible, auditable, and attributable.

**Headline finding (original + revision):** Across 7,004 controlled experiments on nine *deployment stacks* — tuples of (weights, provider, infrastructure, API) — and six task families (extraction, summarisation, multi-turn refinement, RAG, code generation, math reasoning), API-served stacks reproduce their own outputs as little as 1% of the time under temperature-zero greedy decoding with fixed seeds, while local-stack averages reach 93–98%.

**Companion paper (OSF preregistration 2026-05-12):** Rover, L. & de Souza Tadano, Y. *Reproducibility of Pollution–Health Evidence Synthesis using LLM-Assisted Screening and Extraction.* OSF (2026). https://doi.org/10.17605/OSF.IO/VR934. Quantifies the empirical impact of the phenomenon documented in this paper on a 500-abstract evidence base in environmental health (23 study-level effect estimates appear/disappear depending on which run is used). Available for parallel editorial consideration at *Nature Communications*.

## Repository Structure

```
├── article/                          # Manuscript and submission materials
│   ├── ncomms_main.tex               # Revised manuscript (Nature Communications, 28p)
│   ├── ncomms_main.pdf
│   ├── supplementary_nature_mi.tex   # Supplementary Information (S1–S14, 26p)
│   ├── supplementary_nature_mi.pdf
│   ├── CODE_SOFTWARE_CHECKLIST.md    # Nature Code/Software submission checklist
│   ├── ML_CHECKLIST_FILLED.md        # Machine Learning checklist
│   ├── REPORTING_SUMMARY_FILLED.md   # Reporting Summary
│   ├── references.bib                # Bibliography (56 entries, 0 orphan)
│   ├── sn-jnl.cls + sn-nature.bst    # Springer Nature template
│   └── figures/                      # 6 publication figures (PDF, 600 DPI)
├── src/                              # Reference implementation
│   ├── protocol/                     # Core protocol (logger, hasher, run/prompt cards, PROV)
│   ├── models/                       # Model runners (llama, gpt4, claude, gemini)
│   ├── metrics/                      # EMR, NED, ROUGE-L, BERTScore, validation, overhead
│   ├── tasks/                        # Revision-batch task modules (NEW)
│   │   ├── humaneval_loader.py       # HumanEval code generation
│   │   ├── gsm8k_loader.py           # GSM8K math reasoning
│   │   ├── pubmed_loader.py          # 10 PubMed PM2.5 abstracts
│   │   ├── pass_at_1.py              # Sandboxed code execution
│   │   ├── gsm8k_extractor.py        # Math final-answer regex
│   │   ├── llm_judge.py              # Claude Opus 4.7 LLM-as-judge
│   │   └── pm25_case_loader.py       # T3 case sampling
│   └── cost_estimator.py             # Per-call cost estimation + budget guard (NEW)
├── data/
│   └── inputs/
│       ├── abstracts.json            # 30 AI/ML abstracts (original)
│       └── revision/                 # Revision-batch inputs (NEW)
│           ├── humaneval.jsonl       # 164 HumanEval problems (30 sampled)
│           ├── gsm8k_test.jsonl      # 1,319 GSM8K problems (30 sampled)
│           ├── pubmed_pm25_t14.json  # 10 PubMed PM2.5 abstracts
│           └── t3_judge_cases.json   # 10 T3 LLM-judge sample cases
├── outputs/
│   ├── runs/                         # Original 4,104 PROV records
│   ├── run_cards/, prov/, prompt_cards/
│   └── revision/                     # NEW: 2,900 revision runs + T3 judge
│       ├── runs/                     # 2,900 JSONs (HumanEval, GSM8K, PubMed, multi-turn)
│       ├── t3_judge/                 # 10 Claude Opus verdict records
│       └── checkpoint.json           # Budget + resume state
├── analysis/                         # Analysis scripts and results
│   ├── regenerate_figures_nature_mi.py
│   ├── bootstrap_cis.json
│   ├── bertscore_per_field.py        # NEW: per-field BERTScore (R1.5)
│   ├── bertscore_per_field_results.json
│   ├── tables/
│   │   ├── table_per_field_metrics.tex
│   │   └── table_t1_t4_t14.tex       # NEW: revision EMR table
│   ├── figures/per_field_radar.pdf
│   └── revision/                     # NEW: revision-batch analyses
├── tests/                            # 102 tests (51 original + 51 revision)
├── run_experiments.py                # Original experiment runner
├── run_revision_experiments.py       # Unified revision pipeline
├── run_revision_full.sh              # Resumable orchestrator (~$25 budget guard)
├── run_t3_validation.py              # T3 LLM-judge runner (Claude Opus)
├── run_t3_extended.py                # T3 extended LLM-judge runner (gpt-4o)
├── analyze_revision_results.py       # Post-hoc revision analysis
├── REVISION_PLAN.md                  # Revision strategy (archival, 2026-05-08)
├── STATUS.md                         # Current state checkpoint
├── requirements.txt
└── LICENSE                           # MIT (code) + CC-BY 4.0 (data via Figshare)
```

## Key Results

### Original analysis (4,104 experiments, March 2026 submission)

| Model | Deployment | Extraction EMR | Summarisation EMR |
|-------|-----------|----------------|-------------------|
| Gemma 2 9B | Local | **1.000** [1.00, 1.00] | **1.000** [1.00, 1.00] |
| LLaMA 3 8B | Local | **0.987** [0.96, 1.00] | 0.947 [0.89, 0.99] |
| Mistral 7B | Local | **0.960** [0.88, 1.00] | 0.840 [0.72, 0.96] |
| DeepSeek Chat | API | 0.800 | 0.760 |
| GPT-4 (gpt-4-0613) | API | 0.443 [0.32, 0.57] | 0.230 [0.16, 0.30] |
| Claude Sonnet 4.5 | API | 0.190 [0.05, 0.40] | 0.020 [0.00, 0.05] |
| Gemini 2.5 Pro | API | Multi-turn: 0.010 [0.00, 0.03] | RAG: 0.070 [0.02, 0.13] |
| Perplexity Sonar | API | 0.100 | 0.010 |

All comparisons survive Holm-Bonferroni correction (51/68 tests significant). Cliff's delta: 0.784–0.896.

### Revision additions (2,900 new experiments, May 2026)

Following the editor's request for experiments beyond summarisation, and Reviewers 1 and 3:

#### HumanEval (code generation, 30 problems × 5 reps × 8 stacks)

| Stack | EMR | 95% CI |
|-------|-----|--------|
| Locals (Gemma 2 9B, LLaMA 3 8B, Mistral 7B) + Together AI + Gemini 2.5 Pro | 0.92–1.000 | — |
| deepseek-chat | 0.837 | [0.72, 0.93] |
| gpt-4o (gpt-4o-2024-11-20) | 0.837 | [0.75, 0.92] |
| **Claude Sonnet 4.5** | **0.393** | [0.27, 0.52] |

#### GSM8K (math reasoning, 30 problems × 5 reps × 8 stacks)

| Stack | EMR | 95% CI |
|-------|-----|--------|
| Locals + Together AI + Gemini | 0.84–1.000 | — |
| deepseek-chat | 0.370 | [0.26, 0.49] |
| gpt-4o | 0.267 | [0.17, 0.37] |
| **Claude Sonnet 4.5** | **0.063** | [0.03, 0.10] |

#### Multi-turn refinement extension to gpt-4o + deepseek-chat (R3.3)

| Stack | EMR | 95% CI |
|-------|-----|--------|
| deepseek-chat | 0.350 | [0.13, 0.60] |
| gpt-4o | **0.090** | [0.02, 0.16] |

Confirms the near-zero EMR pattern previously reported for Claude (0.040) and Gemini (0.010) is universal across major cloud-served interactive pipelines.

#### PubMed PM2.5/respiratory health (10 abstracts, out-of-AI/ML probe)

| Stack | EMR | 95% CI |
|-------|-----|--------|
| Locals + Together AI | 0.96–1.000 | — |
| deepseek-chat | 0.660 | [0.46, 0.85] |
| gemini-2.5-pro | 0.490 | [0.26, 0.72] |
| gpt-4o | 0.420 | [0.27, 0.58] |
| **Claude Sonnet 4.5** | **0.010** | [0.00, 0.03] |

#### T3 LLM-as-judge triangulation (R3.6)

10 PM2.5 disagreement cases judged blind by Claude Opus 4.7 with three pre-registered criteria (direction, magnitude ±20%, CI overlap):

- 5 **truly_contradictory**
- 3 **semantically_equivalent**
- 2 **ambiguous**

**Half of the divergences are substantive contradictions** even though BERTScore F1 > 0.97 on the same outputs.

#### Per-field reproducibility analysis (R1.5)

BERTScore F1 saturates across all five extracted fields (Δ = 0.001, paired Cohen's d = -0.10), while EMR exposes a paired Cohen's **d = +1.41** between conclusion-relevant fields (mean EMR = 0.455) and metadata fields (mean EMR = 0.684). BERTScore alone is structurally unable to detect substantive payload divergence.

### Revision total: 7,004 experiments across 9 deployment stacks and 6 task families

## Mechanisms vs. experimental controls (mirror of Methods Table 3)

The manuscript (Methods Table 3, *"Mapping of non-determinism mechanisms to deployment stacks"*) lists six mechanisms of distributed-inference non-determinism. This harness **cannot toggle provider-side kernels** — they are closed. What it *can* do is (a) pin every **client-side** variable so residual variation is attributable to the stack rather than to input/parameter drift, and (b) **isolate** which mechanisms are active by comparing single-GPU local serving and the Together AI quasi-isolation stack against the proprietary APIs. The table below connects each manuscript mechanism to the concrete control in this repository.

| Mechanism (Methods Table 3) | How this harness pins or isolates it | Code handle |
|---|---|---|
| Non-associative FP arithmetic | Held constant client-side; exposed identically to all stacks | Condition `C1` (`C1_fixed_seed`): `temperature=0.0`, `seed=42` logged per run |
| Mixed-precision BF16/FP16 | Mitigated on local stacks (Q4 integer arithmetic via Ollama); exposed on Together AI / APIs — the local-vs-cloud contrast isolates it | `--stack` selects local (`ollama`, Q4) vs `together`/API stacks |
| Tensor parallelism | Absent on single-GPU local serving; the L-LLaMA3 vs C-LLaMA3 (Together) vs API comparison isolates it | single-GPU Ollama baseline vs `--stack together`/API |
| FlashAttention kernel | Not invoked on the single-GPU local path; isolated by the same local/Together/API contrast | local stack baseline |
| Dynamic batching | Eliminated by per-request local execution; present on Together AI and APIs | local (one request at a time) vs cloud `--stack` |
| Speculative decoding | Not used locally; possible on Together, likely on APIs | local baseline vs API `--stack` |

In all cases the **snapshot identifier is logged per run** (`gpt-4-0613`, `gpt-4o-2024-11-20`, etc.) for point-in-time auditability, and the seed handling is recorded via `seed_status` (e.g. `logged-only-not-sent-to-api` for the Anthropic surface, which accepts no seed). The Together AI quasi-isolation stack (`C-LLaMA3`) and the local single-GPU stacks (`L-*`) are therefore the *controls* that let the analysis attribute residual variation to mechanisms 2–6; the harness flags above hold mechanism 1 and all client-side parameters fixed. See `run_experiments.py` (conditions `C1_fixed_seed` / `C2_var_seed`) and `run_revision_experiments.py --condition C1 --stack <stack>`.

## Reproducing the Experiments

### Prerequisites

- Python 3.10+ (developed and tested with Python 3.14.3)
- [Ollama](https://ollama.com/) v0.15+ (for local models)
- API keys (optional): OpenAI, Anthropic, Google Gemini, DeepSeek, Together AI

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3:8b && ollama pull mistral:7b && ollama pull gemma2:9b
```

### Running Original Experiments

```bash
# Local models (no API keys needed)
python run_experiments.py

# Expanded experiments (30 abstracts, all conditions)
python run_expanded_experiments.py

# API models (require respective API keys)
export OPENAI_API_KEY="..." && python run_experiments.py --gpt4-only
export ANTHROPIC_API_KEY="..." && python run_claude_multiturn.py
export GEMINI_API_KEY="..." && python run_gemini_multiturn.py
```

### Running Revision Experiments (May 2026)

```bash
# Dry-run (no API calls, cost estimate only)
python run_revision_experiments.py --task all --stack all --condition C1 --dry-run

# Single task on a single stack
python run_revision_experiments.py \
  --task humaneval --stack gpt-4o --condition C1 \
  --n-problems 30 --n-reps 5 --execute

# Full revision batch (orchestrator with checkpoint + resume)
bash run_revision_full.sh

# T3 LLM-as-judge triangulation (PM2.5)
python run_t3_validation.py --sample           # Phase A: cache 10 cases
python run_t3_validation.py --judge --execute  # Phase B: ~$0.28 USD
```

### Analysis and Figures

```bash
# Original figures (Nature MI / NComms format)
python analysis/regenerate_figures_nature_mi.py

# Revision analysis (per-stack EMR + tables)
python analyze_revision_results.py

# Per-field BERTScore (R1.5 — revision)
python analysis/bertscore_per_field.py
```

### Compiling the Manuscript

```bash
cd article
pdflatex ncomms_main.tex
pdflatex ncomms_main.tex                       # second pass for cross-refs
pdflatex supplementary_nature_mi.tex
pdflatex supplementary_nature_mi.tex
```

## Tests

```bash
python -m pytest tests/ -v   # 102 tests (51 original + 51 revision-batch)
```

Test breakdown:

- `tests/test_core.py` — 51 original protocol/metric/PROV tests
- `tests/test_cost_estimator.py` — 16 revision tests (pricing, alias resolution, BudgetGuard)
- `tests/test_humaneval_loader.py` — 8 tests (load, stratified sample, determinism)
- `tests/test_gsm8k_loader.py` — 17 tests (loader + extractor + grade_runs)
- `tests/test_pass_at_1.py` — 10 tests (correct, wrong, timeout, code-fence handling)

## Companion Paper

The 500-abstract PM2.5/respiratory-health analysis underlying the applied-impact finding is reported in detail in our companion paper:

> Rover, L. & Tadano, Y. *When the same question gets different answers: quantifying LLM non-determinism in evidence synthesis.* Research Synthesis Methods (2026, **submitted**).

This NatComms manuscript does NOT reproduce the companion paper's detailed pairwise reproducibility, silver-standard validation, or meta-analytic propagation tables; it cites them and reports only the aggregate finding (23 study-level effect estimates appear/disappear depending on which run is used) plus an independent in-paper LLM-judge triangulation on 10 new cases distinct from the 23 effects.

## Citation

If you use this protocol or dataset, please cite:

> Rover, L., Siqueira, H. V., Bacalhau, E. T., de Azevedo, A. T. & Tadano, Y. S. Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility. *Nature Communications* (2026, **major revision**).

## License

- **Code:** MIT License (see [`LICENSE`](LICENSE))
- **Data and manuscript:** CC-BY 4.0 (see [`LICENSE-DATA`](LICENSE-DATA); the Figshare deposit, DOI 10.6084/m9.figshare.31653373, is CC-BY 4.0)

This matches the manuscript's Code Availability (MIT) and Data Availability (CC-BY 4.0) statements.
