# GenAI Reproducibility Protocol

**Same Prompt, Different Answer: Exposing the Reproducibility Illusion in Large Language Model APIs**

Submitted to the *Journal of Artificial Intelligence Research* (JAIR), February 2026.

## Overview

This repository contains the reference implementation, experimental data, and analysis scripts for a lightweight protocol for logging, versioning, and provenance tracking of generative AI experiments. The protocol introduces **Prompt Cards** and **Run Cards** as structured documentation artifacts, and adopts the **W3C PROV** data model for machine-readable provenance graphs.

## Repository Structure

```
paper-experiment/
├── article/                  # Manuscript and submission materials
│   ├── main.tex             # JAIR-formatted manuscript (~41 pages)
│   ├── references.bib       # Bibliography
│   ├── cover_letter.tex     # Cover letter for JAIR
│   ├── highlights.tex       # Research highlights
│   ├── jair_significance_statement.tex  # JAIR significance statement
│   ├── reviewer-suggestions.yaml  # Suggested reviewers
│   ├── jair.cls             # JAIR LaTeX class
│   ├── tables/              # LaTeX table files
│   └── figures/             # Publication figures (PDF + PNG)
├── src/                      # Reference implementation
│   ├── protocol/            # Core protocol components
│   │   ├── logger.py        # Run logging and metadata collection
│   │   ├── hasher.py        # SHA-256 cryptographic hashing
│   │   ├── prompt_card.py   # Prompt Card generation
│   │   ├── run_card.py      # Run Card generation
│   │   └── prov_generator.py # W3C PROV-JSON generation
│   ├── models/              # Model inference wrappers
│   │   ├── llama_runner.py  # Ollama inference (LLaMA 3, Mistral, Gemma 2)
│   │   ├── gpt4_runner.py   # OpenAI GPT-4 API inference
│   │   ├── claude_runner.py # Anthropic Claude API inference
│   │   └── gemini_runner.py # Google Gemini API inference
│   ├── metrics/             # Variability and overhead metrics
│   │   ├── variability.py   # EMR, NED, ROUGE-L, BERTScore
│   │   ├── overhead.py      # Logging time and storage measurement
│   │   └── validation.py    # JSON schema validation
│   └── experiments/
│       └── config.py        # Experimental conditions and parameters
├── data/
│   └── inputs/
│       └── abstracts.json   # 30 scientific abstracts (input data)
├── outputs/                  # Experimental outputs (4,104 runs)
│   ├── runs/                # Individual run JSON files
│   ├── run_cards/           # Run Card documents
│   ├── prov/                # W3C PROV-JSON provenance documents
│   ├── prompt_cards/        # Prompt Card documents
│   ├── all_runs.json        # Consolidated run data
│   └── experiment_summary.json  # Experiment metadata
├── analysis/                 # Analysis scripts and results
│   ├── compute_metrics.py   # Compute variability/overhead metrics
│   ├── merge_all_runs.py    # Merge individual runs into one file
│   ├── statistical_tests.py # Paired t-tests, Wilcoxon, power analysis
│   ├── ablation_study.py    # Protocol minimality ablation
│   ├── generate_tables_v2.py # Generate LaTeX tables
│   ├── generate_figures_v2.py # Generate publication figures
│   ├── bootstrap_cis.json   # Bootstrap confidence intervals (10k resamples)
│   └── balanced_subsample.json # 10-abstract robustness check
├── run_experiments.py        # Main experiment runner (5 abstracts)
├── run_expanded_experiments.py  # Expanded runner (30 abstracts)
├── run_claude_multiturn.py   # Claude multi-turn + RAG experiments
├── run_gemini_multiturn.py   # Gemini multi-turn + RAG experiments
├── requirements.txt          # Python dependencies
├── LICENSE                   # CC-BY 4.0
└── README.md                 # This file
```

## Reproducing the Experiments

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) v0.15+ (for local inference: LLaMA 3 8B, Mistral 7B, Gemma 2 9B)
- OpenAI API key (for GPT-4 experiments, optional)
- Anthropic API key (for Claude experiments, optional)
- Google Gemini API key (for Gemini experiments, optional)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull gemma2:9b
```

### Running Experiments

```bash
# Run all experiments (both models)
python run_experiments.py

# Expanded experiments (30 abstracts, all conditions)
python run_expanded_experiments.py

# LLaMA 3 only (no API key needed)
python run_experiments.py --llama-only

# GPT-4 only (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-key-here"
python run_experiments.py --gpt4-only

# Claude multi-turn + RAG (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="your-key-here"
python run_claude_multiturn.py

# Gemini multi-turn + RAG (requires GEMINI_API_KEY)
export GEMINI_API_KEY="your-key-here"
python run_gemini_multiturn.py
```

### Regenerating Analysis

```bash
# Merge all runs
python analysis/merge_all_runs.py

# Compute metrics
python analysis/compute_metrics.py

# Statistical tests
python analysis/statistical_tests.py

# Generate tables and figures
python analysis/generate_tables_v2.py
python analysis/generate_figures_v2.py
```

### Compiling the Manuscript

```bash
cd article
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Key Results (4,104 Runs across 8 Models, 5 Providers, 4 Tasks, 30 Abstracts)

| Model | Source | Task | EMR [95% CI] |
|-------|--------|------|---------------|
| Gemma 2 9B | Local | Extraction | **1.000** [1.00, 1.00] |
| Gemma 2 9B | Local | Summarization | **1.000** [1.00, 1.00] |
| LLaMA 3 8B | Local | Extraction | **0.987** [0.96, 1.00] |
| LLaMA 3 8B | Local | Summarization | 0.947 [0.89, 0.99] |
| Mistral 7B | Local | Extraction | **0.960** [0.88, 1.00] |
| Mistral 7B | Local | Summarization | 0.840 [0.72, 0.96] |
| GPT-4 | API | Extraction | 0.443 [0.32, 0.57] |
| GPT-4 | API | Summarization | 0.230 [0.16, 0.30] |
| Claude Sonnet 4.5 | API | Extraction | 0.190 [0.05, 0.40] |
| Claude Sonnet 4.5 | API | Summarization | 0.020 [0.00, 0.05] |
| Gemini 2.5 Pro | API | Multi-turn | 0.010 [0.00, 0.03] |
| Gemini 2.5 Pro | API | RAG | 0.070 [0.02, 0.13] |

**Headline result:** Local models average EMR = 0.956 vs. API models EMR = 0.221 under greedy decoding (>4x gap).

**Protocol overhead:** 21-30 ms mean (<1% of inference time), ~4 KB per run record.

**Statistical significance:** All primary local-vs-API comparisons survive Holm-Bonferroni correction (51 of 68 individual tests reach significance after correction).

**Note:** GPT-4 experiments used the `gpt-4-0613` snapshot. The GPT-4 C1 condition was severely incomplete (8/300 runs) due to API quota exhaustion; C2 is used as GPT-4's primary greedy condition.

## Tests

51 tests passing across protocol validation, metric computation, and provenance verification.

```bash
python -m pytest tests/ -v
```

## License

CC-BY 4.0
