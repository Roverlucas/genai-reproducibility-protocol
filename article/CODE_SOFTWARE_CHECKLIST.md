# Nature Code/Software Submission Checklist — FILLED
## "Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility"

> Reference policy: Nature code & software submission policy (https://www.nature.com/documents/nr-software-policy.pdf).
> Corresponding author: Lucas Rover (ORCID: 0000-0001-6641-9224)
> Coauthors: Hugo Valadares Siqueira (0000-0002-1278-4602); Eduardo Tadeu Bacalhau (0000-0002-3936-0375); Anibal Tavares de Azevedo (0000-0003-1678-7795); Yara de Souza Tadano (0000-0002-3975-3419)
> Manuscript ID: NCOMMS-2026-XXXXX (Major Revision, Nature Communications)
> Revision date: 2026-05-11

---

## 1. Software identification

| Item | Response |
|------|----------|
| Software name | GenAI Reproducibility Protocol (Run Card / Prompt Card / W3C PROV reference implementation) |
| Public repository | https://github.com/Roverlucas/genai-reproducibility-protocol |
| Repository visibility (review) | Private — reviewer link supplied via Editorial Manager |
| Repository visibility (post-publication) | Public on acceptance |
| Versioned release for this manuscript | Tag `v1.1-natcomms-revision1` (created at submission of revised manuscript) |
| Persistent archive (snapshot) | Figshare DOI [10.6084/m9.figshare.31653373](https://doi.org/10.6084/m9.figshare.31653373) (CC-BY 4.0); GitHub release tag `v1.1-natcomms-revision1` is mirrored on Figshare |
| Reviewer data archive (private) | Figshare reviewer link — see Data Availability for the URL provided to the editorial office (excluded from the published version) |
| License (code) | MIT (LICENSE file at repository root) |
| License (data and manuscript artefacts) | CC-BY 4.0 |
| Authors / contributors | Lucas Rover — ORCID [0000-0001-6641-9224](https://orcid.org/0000-0001-6641-9224) (lead developer, corresponding author); Hugo Valadares Siqueira — ORCID [0000-0002-1278-4602](https://orcid.org/0000-0002-1278-4602); Eduardo Tadeu Bacalhau — ORCID [0000-0002-3936-0375](https://orcid.org/0000-0002-3936-0375); Anibal Tavares de Azevedo — ORCID [0000-0003-1678-7795](https://orcid.org/0000-0003-1678-7795); Yara de Souza Tadano — ORCID [0000-0002-3975-3419](https://orcid.org/0000-0002-3975-3419) (manuscript authors) |
| Contact | lucasrover@alunos.utfpr.edu.br |

---

## 2. Documentation

| Item | Status | Location |
|------|--------|----------|
| README at repository root | ☑ | `README.md` (overview, setup, reproducing experiments, citation) |
| Revision plan and decision log | ☑ | `REVISION_PLAN.md` (consolidated response strategy for the Major Revision) |
| Live status / progress tracker | ☑ | `STATUS.md` (per-task status of T1–T17 revision deliverables) |
| Method-level documentation | ☑ | Manuscript Methods §Models and infrastructure, §Protocol design, §Statistical analysis; Supplementary §S1–S10 |
| API payload documentation | ☑ | Supplementary §S4 (exact JSON request bodies for every deployment stack) |
| Run-Card / Prompt-Card schema | ☑ | `src/protocol/run_card.py`, `src/protocol/prompt_card.py`, JSON schemas under `src/protocol/` |
| Provenance documentation (W3C PROV) | ☑ | `src/protocol/prov_generator.py`; PROV-JSON examples in `outputs/prov/` |
| Inline code documentation | ☑ | Module-level and function-level docstrings throughout `src/` |

---

## 3. Installation instructions

| Item | Response |
|------|----------|
| Programming language and version | Python 3.14.3 (tested); minimum supported 3.10 (declared in `README.md`) |
| Package manifest | `requirements.txt` (pinned versions) |
| Recommended environment | Virtual environment via `python -m venv .venv` |
| Reference install commands | `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| Local LLM runtime | Ollama v0.15.5 (https://ollama.com/) — `ollama pull llama3:8b mistral:7b gemma2:9b` |
| Optional dependencies | `datasets` (HumanEval/GSM8K loaders fall back to GitHub-mirror fetch when absent); `tiktoken` (cost estimator falls back to chars/4 heuristic when absent) |
| One-command sanity check | `python -m pytest tests/ -q` (102 tests must pass) |

---

## 4. Dependencies (from `requirements.txt`)

| Package | Pinned version | Purpose |
|---------|---------------|---------|
| `openai` | 2.17.0 | OpenAI API client (GPT-4, GPT-4o stacks) |
| `ollama` | 0.4.7 | Local model serving (LLaMA 3, Mistral, Gemma 2) |
| `numpy` | 2.4.2 | Numerical primitives, bootstrap resampling |
| `python-Levenshtein` | 0.27.3 | Normalized edit distance |
| `pyyaml` | 6.0.3 | Configuration files |
| `tabulate` | 0.9.0 | Tabular reporting in analysis scripts |
| `matplotlib` | 3.10.8 | Figure generation (Fig. 1–4 + Extended Data) |
| `scipy` | 1.17.0 | Fisher's exact, Mann–Whitney U, Wilcoxon signed-rank |
| `bert-score` | 0.3.13 | BERTScore F1 (semantic-equivalence layer) |
| `datasets` (optional) | 2.21.0 | HumanEval / GSM8K loaders |
| `tiktoken` (optional) | 0.7.0 | Cost estimator |

External SDKs called via `urllib` (no extra pip dependency): Anthropic Messages API, Google Gemini API, DeepSeek Chat Completions API, Perplexity Sonar API, Together AI Chat Completions API.

---

## 5. Example data and expected output

| Item | Path | Notes |
|------|------|-------|
| Original input corpus (30 ML/AI abstracts) | `data/inputs/abstracts.json` | Used for Tasks 1–4 in the original submission |
| Revision input corpus (10 PubMed PM2.5 abstracts, T14) | `data/inputs/revision/` | Cross-domain extension introduced in the revision |
| Original run records (4,104 PROV-instrumented runs) | `outputs/runs/` | One JSON per run; aggregated in `outputs/all_runs.json` |
| Revision run records (T1+T4+T14, in progress) | `outputs/revision/runs/` | Currently 808 records; final count documented in revised manuscript Table 1 (TBD pending T1 completion) |
| Run Cards (human-readable provenance) | `outputs/run_cards/`, `outputs/revision/run_cards/` | One per run |
| W3C PROV-JSON provenance graphs | `outputs/prov/` | Validated against W3C PROV-JSON serialization |
| Aggregated metrics | `analysis/expanded_metrics.json`, `analysis/bootstrap_cis.json`, `analysis/bertscore_per_field_results.json`, `analysis/revision/*.json` | Inputs for figure/table generators |
| Expected figure outputs | `article/figures/` | Compiled PDFs at 600 DPI |
| Expected table outputs | `article/tables/`, `analysis/tables/` | LaTeX fragments included via `\input{}` in `ncomms_main.tex` |

---

## 6. Tests

| Item | Response |
|------|----------|
| Test framework | `pytest` |
| Test directory | `tests/` |
| Test file inventory | `test_core.py` (protocol, hashing, provenance, metrics — original 51 tests); `test_cost_estimator.py`, `test_humaneval_loader.py`, `test_gsm8k_loader.py`, `test_pass_at_1.py` (new for revision) |
| Total tests | 102 (51 original + 51 new revision tests) |
| Status | All 102 collected; full suite passes locally on Python 3.14.3 |
| How to run | `python -m pytest tests/ -v` |
| Coverage targets | Run Card / Prompt Card schema validation, SHA-256 hashing, EMR / NED / ROUGE-L / BERTScore, W3C PROV serialization, HumanEval `pass@1` sandbox (timeout, runtime errors, code-fence stripping), GSM8K answer extraction, cost estimator |

---

## 7. Reproducibility instructions

Step-by-step protocol to reproduce all results in the revised manuscript:

1. **Clone and check out the tagged release:**
   ```bash
   git clone https://github.com/Roverlucas/genai-reproducibility-protocol.git
   cd genai-reproducibility-protocol
   git checkout v1.1-natcomms-revision1
   ```
2. **Create the environment and install pinned dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Pull local model weights (open-weight stacks):**
   ```bash
   ollama pull llama3:8b mistral:7b gemma2:9b
   ```
4. **Set API keys for closed stacks** (only if reproducing API-served runs):
   ```bash
   export OPENAI_API_KEY=...
   export ANTHROPIC_API_KEY=...
   export GEMINI_API_KEY=...
   export DEEPSEEK_API_KEY=...
   export TOGETHER_API_KEY=...     # quasi-isolation probe
   export PERPLEXITY_API_KEY=...    # not used in revision (D4)
   ```
5. **Reproduce the original 4,104 runs:**
   ```bash
   python run_experiments.py             # local stacks + GPT-4 single-turn
   python run_claude_multiturn.py        # Claude multi-turn / RAG
   python run_gemini_multiturn.py        # Gemini multi-turn / RAG
   python run_chat_control.py            # Supplementary §S7 chat-format control
   ```
6. **Reproduce the revision experiments (T1 HumanEval, T1 GSM8K, T4 multi-turn extension, T14 PubMed PM2.5):**
   ```bash
   ./run_revision_full.sh                # resumable orchestrator (logs in outputs/revision/logs/)
   ```
   The script honours `--resume` and a per-stack budget cap; partial runs reload from `outputs/revision/checkpoint.json`.
7. **Regenerate metrics, tables, and figures:**
   ```bash
   python run_full_analysis.py
   python analysis/regenerate_figures_nature_mi.py
   python analysis/bertscore_per_field.py        # T6 per-field analysis
   python analyze_revision_results.py            # T1+T4+T14 aggregation
   ```
8. **Compile the manuscript:**
   ```bash
   cd article
   pdflatex ncomms_main.tex
   biber   ncomms_main
   pdflatex ncomms_main.tex
   pdflatex ncomms_main.tex
   ```
9. **Verify the test suite:**
   ```bash
   python -m pytest tests/ -v       # expect 102 passed
   ```

Run-level reproducibility evidence is preserved per run in `outputs/runs/<run_id>.json` and `outputs/prov/<run_id>.prov.json` (Run Card + W3C PROV graph; SHA-256 over canonicalised Prompt Card and Run Card).

---

## 8. Computing environment

| Item | Response |
|------|----------|
| Reference workstation | Apple M4, 24 GB unified memory, macOS 14.6 |
| Python | 3.14.3 (CPython); minimum supported 3.10 |
| LLM runtime (local stacks) | Ollama v0.15.5 with Apple Metal acceleration |
| Container/image | Not required; `requirements.txt` is the canonical environment specification. A Dockerfile is on the post-acceptance roadmap (Maintenance plan, §10) |
| OS dependencies | `pdflatex`, `biber` (TeX Live or TinyTeX) for manuscript compilation only |
| Network access | Required only for API-served stacks (OpenAI, Anthropic, Google, DeepSeek, Perplexity, Together AI). Local stacks run fully offline once weights are pulled |
| Random-seed handling | Per-run seeds (42, 123, 456, 789, 1024) recorded in every Run Card; greedy decoding (temperature 0) used for the deterministic conditions |

---

## 9. Hardware requirements

| Item | Recommended |
|------|-------------|
| RAM | 16 GB minimum; 24 GB recommended for Gemma 2 9B local serving |
| Disk | ~10 GB (model weights) + ~2 GB (run records, PROV graphs, figures) |
| GPU | Optional. Apple Silicon (Metal) used for local stacks on the reference workstation; an NVIDIA GPU with ≥8 GB VRAM achieves equivalent or better throughput. CPU-only execution is supported by Ollama at reduced throughput |
| API access | Only required to reproduce API-served stacks; ~US$50 of credit was sufficient for the full revision experimental matrix |

---

## 10. Maintenance plan

| Item | Response |
|------|----------|
| Long-term hosting | GitHub (https://github.com/Roverlucas/genai-reproducibility-protocol). Repository made public on acceptance |
| Persistent archival | Figshare (DOI: [10.6084/m9.figshare.31653373](https://doi.org/10.6084/m9.figshare.31653373), CC-BY 4.0). The GitHub release tag `v1.1-natcomms-revision1` is mirrored on Figshare. The deposit is available privately to reviewers via the share URL in the Cover Letter and will be made public at acceptance |
| Issue tracking | GitHub Issues. Bug reports and reproduction questions are accepted from the broader community |
| Versioning | Semantic versioning. Future protocol changes will not overwrite the manuscript tag |
| Author commitment | Lucas Rover (corresponding author) commits to maintain the repository for ≥5 years post-publication, in accordance with the funding institution's data-management policy (UTFPR) |
| Backup | Mirror to institutional repository (UTFPR / RIUT) on acceptance |

---

## 11. Items requiring final insertion before submission

| Item | Status | Action |
|------|--------|--------|
| Figshare DOI | ☑ | [10.6084/m9.figshare.31653373](https://doi.org/10.6084/m9.figshare.31653373) — already minted (CC-BY 4.0); private reviewer share URL: https://figshare.com/s/3d17327cef1ae99ed37c |
| Final run count (4,104 + revision additions) | ⚠ TBD | Update Table 1 and §1 (CODE_SOFTWARE_CHECKLIST) once T1 HumanEval + T1 GSM8K complete |
| Pass@1 results (HumanEval) | ⚠ TBD | Pending T1 execution completion |
| Final tagged release `v1.1-natcomms-revision1` | ⚠ TBD | Created at the moment the revised manuscript is uploaded |
| Reviewer figshare link | ☑ Provided to editorial office; not in the public manuscript | https://figshare.com/s/3d17327cef1ae99ed37c (private; for reviewers only) |

---

*Prepared by Lucas Rover (corresponding author). Cross-references: `REVISION_PLAN.md`, `STATUS.md`, `article/REPORTING_SUMMARY_FILLED.md`, `article/ML_CHECKLIST_FILLED.md`.*
