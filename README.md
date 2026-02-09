# GenAI Reproducibility Protocol

**Hidden Non-Determinism in Large Language Model APIs: A Lightweight Provenance Protocol for Reproducible Generative AI Research**

Submitted to the *Journal of Artificial Intelligence Research* (JAIR), February 2026.

## Overview

This repository contains the reference implementation, experimental data, and analysis scripts for a lightweight protocol for logging, versioning, and provenance tracking of generative AI experiments. The protocol introduces **Prompt Cards** and **Run Cards** as structured documentation artifacts, and adopts the **W3C PROV** data model for machine-readable provenance graphs.

## Repository Structure

```
paper-experiment/
├── article/                  # Manuscript and submission materials
│   ├── main.tex             # JAIR-formatted manuscript
│   ├── references.bib       # Bibliography (36 entries)
│   ├── cover_letter.tex     # Cover letter for JAIR
│   ├── jair_significance_statement.tex  # JAIR significance statement
│   ├── reviewer-suggestions.yaml  # Suggested reviewers
│   ├── jair.cls             # JAIR LaTeX class
│   ├── table2_main_results_v2.tex  # Main results table
│   ├── table3_overhead_v2.tex      # Overhead table
│   ├── table_model_comparison.tex  # Model comparison table
│   └── figures/             # Publication figures (PDF + PNG)
├── src/                      # Reference implementation
│   ├── protocol/            # Core protocol components
│   │   ├── logger.py        # Run logging and metadata collection
│   │   ├── hasher.py        # SHA-256 cryptographic hashing
│   │   ├── prompt_card.py   # Prompt Card generation
│   │   ├── run_card.py      # Run Card generation
│   │   └── prov_generator.py # W3C PROV-JSON generation
│   ├── models/              # Model inference wrappers
│   │   ├── llama_runner.py  # Ollama/LLaMA 3 8B inference
│   │   └── gpt4_runner.py   # OpenAI GPT-4 API inference
│   ├── metrics/             # Variability and overhead metrics
│   │   ├── variability.py   # EMR, NED, ROUGE-L, BERTScore
│   │   ├── overhead.py      # Logging time and storage measurement
│   │   └── validation.py    # JSON schema validation
│   └── experiments/
│       └── config.py        # Experimental conditions and parameters
├── data/
│   └── inputs/
│       └── abstracts.json   # 30 scientific abstracts (input data)
├── outputs/                  # Experimental outputs (1,864 runs)
│   ├── runs/                # 1,864 individual run JSON files
│   ├── run_cards/           # 1,864 Run Card documents
│   ├── prov/                # 331 W3C PROV-JSON provenance documents
│   ├── prompt_cards/        # 2 Prompt Card documents
│   ├── all_runs.json        # Consolidated run data (~8 MB)
│   └── experiment_summary.json  # Experiment metadata
├── analysis/                 # Analysis scripts and results
│   ├── compute_metrics.py   # Compute variability/overhead metrics
│   ├── merge_all_runs.py    # Merge individual runs into one file
│   ├── statistical_tests.py # Paired t-tests, Wilcoxon, power analysis
│   ├── ablation_study.py    # Protocol minimality ablation
│   ├── generate_tables_v2.py # Generate LaTeX tables
│   ├── generate_figures_v2.py # Generate publication figures
│   └── full_analysis.json   # Complete metrics for all 1,864 runs
├── run_experiments.py        # Main experiment runner (5 abstracts)
├── run_expanded_experiments.py  # Expanded runner (30 abstracts)
├── requirements.txt          # Python dependencies
├── LICENSE                   # CC-BY 4.0
└── README.md                 # This file
```

## Reproducing the Experiments

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) (for local LLaMA 3 8B inference)
- OpenAI API key (for GPT-4 experiments, optional)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3:8b
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

## Key Results (1,864 Runs across 30 Abstracts)

| Model | Task | Condition | EMR | NED | ROUGE-L |
|-------|------|-----------|-----|-----|---------|
| LLaMA 3 8B | Extraction | Greedy (t=0) | **0.987** | 0.0031 | 0.9966 |
| LLaMA 3 8B | Summarization | Greedy (t=0) | 0.947 | 0.0050 | 0.9945 |
| GPT-4 | Extraction | Greedy (t=0) | 0.443 | 0.0724 | 0.9384 |
| GPT-4 | Summarization | Greedy (t=0) | 0.230 | 0.1365 | 0.8695 |

**Protocol overhead:** 25.43 ms mean (0.545% of inference time), ~4.1 KB per run record, 19.52 MB total.

**Statistical significance:** All primary comparisons p < 0.0001 (paired t-test), Cohen's d > 1.6, power > 0.999.

**Note:** GPT-4 experiments used the `gpt-4-0613` snapshot. The GPT-4 C1 condition was severely incomplete (8/300 runs) due to API quota exhaustion; C2 and C3 conditions are complete.

## License

CC-BY 4.0
