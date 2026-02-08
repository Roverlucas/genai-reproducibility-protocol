# GenAI Reproducibility Protocol

**Logging, Versioning, and Provenance in Generative AI Studies: A Protocol for Auditability and Scientific Reproducibility**

Submitted to the *Journal of Artificial Intelligence Research* (JAIR), February 2026.

## Overview

This repository contains the reference implementation, experimental data, and analysis scripts for a lightweight protocol for logging, versioning, and provenance tracking of generative AI experiments. The protocol introduces **Prompt Cards** and **Run Cards** as structured documentation artifacts, and adopts the **W3C PROV** data model for machine-readable provenance graphs.

## Repository Structure

```
paper-experiment/
├── article/                  # Manuscript and submission materials
│   ├── main.tex             # JAIR-formatted manuscript
│   ├── references.bib       # Bibliography
│   ├── cover_letter.tex     # Cover letter for JAIR
│   ├── suggested_reviewers.txt
│   ├── jair.cls             # JAIR LaTeX class
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
│   │   ├── variability.py   # EMR, NED, ROUGE-L computation
│   │   └── overhead.py      # Logging time and storage measurement
│   └── experiments/
│       └── config.py        # Experimental conditions and parameters
├── data/
│   └── inputs/
│       └── abstracts.json   # 5 NLP paper abstracts (input data)
├── outputs/                  # Experimental outputs (330 runs)
│   ├── runs/                # Individual run JSON files
│   ├── run_cards/           # Run Card documents
│   ├── prov/                # W3C PROV-JSON provenance documents
│   ├── prompt_cards/        # Prompt Card documents
│   └── all_runs.json        # Consolidated run data
├── analysis/                 # Analysis scripts and results
│   ├── compute_metrics.py   # Compute variability/overhead metrics
│   ├── merge_all_runs.py    # Merge individual runs into one file
│   ├── generate_tables_v2.py # Generate LaTeX tables
│   ├── generate_figures_v2.py # Generate publication figures
│   ├── full_analysis.json   # Complete metrics for all 330 runs
│   ├── latex_tables_v2.tex  # Generated LaTeX table fragments
│   └── figures/             # Generated figures (10 PDF + 10 PNG)
├── run_experiments.py        # Main experiment runner
├── requirements.txt          # Python dependencies
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

## Key Results (330 Runs)

| Model | Task | Condition | EMR | NED | ROUGE-L |
|-------|------|-----------|-----|-----|---------|
| LLaMA 3 8B | Extraction | Greedy (t=0) | **1.000** | 0.0000 | 1.0000 |
| LLaMA 3 8B | Summarization | Greedy (t=0) | 0.840 | 0.0148 | 0.9823 |
| GPT-4 | Extraction | Greedy (t=0) | 0.520 | 0.0343 | 0.9748 |
| GPT-4 | Summarization | Greedy (t=0) | 0.200 | 0.0718 | 0.9295 |

**Protocol overhead:** 33.56 ms mean (0.69% of inference time), 4.17 KB per run.

## License

CC-BY 4.0
