# Same Prompt, Different Answer

**Exposing the Reproducibility Illusion in Large Language Model APIs**

Lucas Rover, Eduardo Tadeu Bacalhau, Anibal Tavares de Azevedo & Yara de Souza Tadano

Submitted to *Nature Machine Intelligence*, February 2026.

## Overview

This repository contains the reference implementation, experimental data, analysis scripts, and manuscript for a study demonstrating that API-served large language models fail to reproduce their own outputs under documented "deterministic" settings. We provide a lightweight provenance protocol grounded in W3C PROV that makes this invisible variation visible, auditable, and attributable.

**Headline finding:** Under temperature-zero greedy decoding, API-served models reproduce their outputs only 22.1% of the time (EMR), while locally deployed models achieve 95.6% — a gap exceeding four-fold.

## Repository Structure

```
├── article/                     # Manuscript and submission materials
│   ├── nature_mi_main.tex       # Main manuscript (Nature MI format)
│   ├── nature_mi_main.pdf       # Compiled manuscript
│   ├── nature_mi_cover_letter.tex  # Cover letter
│   ├── supplementary_nature_mi.tex # Supplementary Information (S1-S10)
│   ├── references.bib           # Bibliography (~50 references)
│   ├── sn-jnl.cls              # Springer Nature template
│   ├── sn-nature.bst           # Nature bibliography style
│   ├── figures/                 # Publication figures (PDF, 600 DPI)
│   └── tables/                  # Extended Data tables (LaTeX)
├── src/                         # Reference implementation
│   ├── protocol/               # Core protocol components
│   │   ├── logger.py           # Run logging and metadata collection
│   │   ├── hasher.py           # SHA-256 cryptographic hashing
│   │   ├── prompt_card.py      # Prompt Card generation
│   │   ├── run_card.py         # Run Card generation
│   │   └── prov_generator.py   # W3C PROV-JSON generation
│   ├── models/                 # Model inference wrappers
│   │   ├── llama_runner.py     # Ollama (LLaMA 3, Mistral, Gemma 2)
│   │   ├── gpt4_runner.py      # OpenAI GPT-4
│   │   ├── claude_runner.py    # Anthropic Claude
│   │   └── gemini_runner.py    # Google Gemini
│   └── metrics/                # Variability and overhead metrics
│       ├── variability.py      # EMR, NED, ROUGE-L, BERTScore
│       ├── overhead.py         # Logging time and storage
│       └── validation.py       # JSON schema validation
├── data/inputs/                 # 30 scientific abstracts
├── outputs/                     # 4,104 experimental run records
│   ├── runs/                   # Individual run JSON files
│   ├── run_cards/              # Run Card documents
│   ├── prov/                   # W3C PROV-JSON provenance graphs
│   └── prompt_cards/           # Prompt Card documents
├── analysis/                    # Analysis scripts and results
│   ├── regenerate_figures_nature_mi.py  # Figure generation (Nature MI style)
│   ├── bootstrap_cis.json      # Bootstrap 95% CIs (10k resamples)
│   ├── expanded_metrics.json   # Full metric results
│   └── ...                     # Statistical tests, ablation, etc.
├── tests/                       # 51 tests (protocol, metrics, provenance)
├── run_experiments.py           # Main experiment runner
├── run_expanded_experiments.py  # Expanded runner (30 abstracts)
├── run_claude_multiturn.py      # Claude multi-turn + RAG experiments
├── run_gemini_multiturn.py      # Gemini multi-turn + RAG experiments
├── requirements.txt             # Python dependencies
└── LICENSE                      # CC-BY 4.0
```

## Key Results

**4,104 experiments across 8 models, 5 API providers, 4 tasks, 30 abstracts.**

| Model | Deployment | Extraction EMR | Summarisation EMR |
|-------|-----------|----------------|-------------------|
| Gemma 2 9B | Local | **1.000** [1.00, 1.00] | **1.000** [1.00, 1.00] |
| LLaMA 3 8B | Local | **0.987** [0.96, 1.00] | 0.947 [0.89, 0.99] |
| Mistral 7B | Local | **0.960** [0.88, 1.00] | 0.840 [0.72, 0.96] |
| DeepSeek Chat | API | 0.800 | 0.760 |
| GPT-4 | API | 0.443 [0.32, 0.57] | 0.230 [0.16, 0.30] |
| Claude Sonnet 4.5 | API | 0.190 [0.05, 0.40] | 0.020 [0.00, 0.05] |
| Perplexity Sonar | API | 0.100 | 0.010 |

All comparisons survive Holm-Bonferroni correction (51/68 tests significant). Cliff's delta: 0.784-0.896.

## Reproducing the Experiments

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) v0.15+ (for local models)
- API keys (optional): OpenAI, Anthropic, Google Gemini

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3:8b && ollama pull mistral:7b && ollama pull gemma2:9b
```

### Running Experiments

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

### Analysis and Figures

```bash
python analysis/regenerate_figures_nature_mi.py
```

### Compiling the Manuscript

```bash
cd article
pdflatex nature_mi_main.tex
pdflatex nature_mi_main.tex   # second pass for cross-references
```

## Tests

```bash
python -m pytest tests/ -v   # 51 tests passing
```

## Citation

If you use this protocol or dataset, please cite:

> Rover, L., Bacalhau, E. T., de Azevedo, A. T. & Tadano, Y. S. Same Prompt, Different Answer: Exposing the Reproducibility Illusion in Large Language Model APIs. *Nature Machine Intelligence* (2026, submitted).

## License

- **Code:** MIT License
- **Data and manuscript:** CC-BY 4.0
