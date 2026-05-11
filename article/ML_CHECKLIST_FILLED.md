# Machine Learning Checklist V 1.1 — FILLED GUIDE
## "Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility"

> Use this guide to fill in the official Nature PDF form.
> Corresponding author: Lucas Rover (ORCID: 0000-0001-6641-9224)
> Coauthors: Hugo Valadares Siqueira (0000-0002-1278-4602); Eduardo Tadeu Bacalhau (0000-0002-3936-0375); Anibal Tavares de Azevedo (0000-0003-1678-7795); Yara de Souza Tadano (0000-0002-3975-3419)
> Date: 11-05-2026

---

## 1. Availability and reproducibility of Code and Data

| Item | Check? | URL / Notes |
|------|--------|-------------|
| Code will be included in a CodeOcean capsule | ☐ | N/A |
| **Source code** is included or available in a public repository | ☑ | https://github.com/Roverlucas/genai-reproducibility-protocol (available to reviewers during review; public upon publication) |
| A **compiled standalone version** is available | ☐ | N/A — Python scripts, no compilation needed |
| A **test dataset** and instructions/scripts for replicating results | ☑ | https://github.com/Roverlucas/genai-reproducibility-protocol (data/inputs/ contains all 30 abstracts; run scripts reproduce all experiments) |
| A **Readme file** with install/run instructions | ☑ | https://github.com/Roverlucas/genai-reproducibility-protocol (README.md in repository root) |
| Code available to reviewers during review | ☑ | Repository access provided via editorial system |
| **Pretrained models** used and accessible | ☑ | Ollama Hub: llama3:8b, mistral:7b, gemma2:9b (open-weight, freely downloadable) |
| **Pretrained models** used but not accessible | ☐ | N/A |
| Paper describes how to obtain code and data after publication | ☑ | Data Availability and Code Availability statements in manuscript |

---

## 2. Datasets

| Item | Answer | Details |
|------|--------|---------|
| **A.** All data sources listed in paper | ☑ Yes | Methods §Input data; Supplementary Section S2 (Table S1 lists all 30 abstracts with citations); Supplementary Section S3 (RAG contexts) |
| **B.** Train/test/validation datasets publicly available with links | ☑ Yes | All 30 input abstracts and RAG contexts available at GitHub repo (data/inputs/abstracts.json, data/inputs/rag_contexts.json). No train/test split — this is an inference reproducibility study, not a training study. |
| **C.** Reported and discussed potential dataset biases; mitigation strategies used | ☑ Yes | Discussion §Limitations: corpus limited to 30 English-language AI/ML abstracts; other domains/languages may show different effects. Balanced subsample analysis (Extended Data Table 4) controls for sample-size differences. |
| **D.** Data cleaning and preprocessing steps clearly described | ☑ Yes | Methods §Input data and §Tasks. No preprocessing applied to abstracts — used verbatim. Supplementary Section S4 documents exact API payloads. |
| **E.** Instances of combining data from multiple sources clearly identified | ☐ No | Not applicable — all data generated from a single experimental protocol. No combining of external datasets. |

---

## 3. Model and training

| Item | Answer | Details |
|------|--------|---------|
| **A.** Model architecture | N/A — We do not propose a new model. We evaluate 8 existing LLMs: LLaMA 3 8B (Transformer decoder), Mistral 7B (Transformer decoder), Gemma 2 9B (Transformer decoder), GPT-4 (proprietary), Claude Sonnet 4.5 (proprietary), Gemini 2.5 Pro (proprietary), DeepSeek Chat (Transformer decoder), Perplexity Sonar (proprietary + retrieval). |
| **B.** A Model Card is provided | ☐ No | We do not train or release a model. We provide a "Run Card" (provenance record) for each of the 4,104 inference runs, which extends the Model Card concept to the inference layer (Methods §Protocol design). |
| **C.** Data split into train/validation/test | ☐ No | Not applicable — this is an inference reproducibility study, not a training or prediction study. No model training was performed. |
| **D.** Method of data splitting clearly stated | ☐ No | Not applicable — no data splitting. All 30 abstracts are used as inputs for all models under identical conditions. |
| **E.** Data splitting mimics real-world applications | ☐ No | Not applicable — no data splitting performed. |
| **F.** Data splitting avoids data leakage | ☐ No | Not applicable — no data splitting performed. |
| **G.** Interpretability studied and validated | ☐ No | Not applicable — we do not develop or train a model. We measure output reproducibility of existing models. Interpretability of the reproducibility gap is addressed through the quasi-isolation experiment (Results §Cloud deployment) and sources-of-non-determinism analysis (Methods §Sources). |

---

## 4. Evaluation

| Item | Answer | Details |
|------|--------|---------|
| **A.** Performance metrics described and justified | ☑ Yes | Methods §Metrics: Exact Match Rate (EMR), Normalized Edit Distance (NED), ROUGE-L F1, BERTScore F1. Three-level framework justified in Results §Semantic analysis. |
| **B.** Cross-validation included | ☐ No | Not applicable — no predictive model. Robustness verified via 10,000-resample bootstrap CIs and balanced 10-abstract subsample analysis (Extended Data Table 4). |
| **C.** Community-accepted benchmark datasets/tasks used | ☐ No | Not applicable in the traditional sense. We designed a novel experimental protocol for inference reproducibility. The 30 abstracts are drawn from highly cited ML papers (Supplementary Section S2). BERTScore and ROUGE-L are community-standard NLP metrics. |
| **D.** Baseline comparisons to simple/trivial models | ☐ No | Not applicable — we do not propose a predictive model. The "baseline" is the local deployment (near-perfect reproducibility), against which API deployments are compared. The Together AI quasi-isolation serves as a controlled baseline (same weights, different infrastructure). |
| **E.** Benchmarks with current state-of-the-art | ☐ No | Not applicable — no predictive task. We compare against prior work on LLM non-determinism (Atil et al. 2024, Yuan et al. 2025) in the Discussion. Our protocol is compared feature-by-feature against 5 existing tools (Supplementary Table S2). |
| **F.** Ablation experiments included | ☑ Yes | Supplementary Section S6: Protocol Minimality ablation — systematically removes each field group from the Run Card and assesses which audit questions become unanswerable (Supplementary Tables S3–S4). Temperature sweep (Extended Data Table 3) ablates the temperature parameter. Chat-format control (Supplementary Section S7) ablates prompt format. |
| **G.** Model tested on fully independent dataset | ☐ No | Not applicable — no predictive model. The study tests inference reproducibility across 8 models on the same input corpus. |

---

## 5. Computational resources

| Item | Answer | Details |
|------|--------|---------|
| **A.** Hardware/computing resources reported | ☑ Yes | Methods §Models and infrastructure: Apple M4 (24 GB RAM) for local models via Ollama v0.15.5. API models accessed via respective cloud endpoints. |
| **B.** Computational costs reported | ☑ Yes | Methods §Protocol overhead: <1% logging overhead (~25 ms per run, ~4 KB storage per run). Total 4,104 runs. Extended Data Table 6 provides detailed overhead breakdown. |

---

## SUMMARY

- **Sections fully answered**: 1 (Code/Data), 2 (Datasets), 4A (Metrics), 4F (Ablation), 5 (Compute)
- **Sections marked N/A with explanation**: 3 (Model/Training — we evaluate existing models, not train new ones), 4B–E,G (Evaluation — no predictive task)
- **Key message for editors**: This is an empirical reproducibility study of LLM inference, not a model development paper. The ML Checklist items about training, data splitting, and model cards do not apply, but we provide equivalent transparency through our provenance protocol (Run Cards) and open experimental records.

---

## Revision additions (2026-05-08, Major Revision response — Nature Communications)

This section documents the experimental and analytical extensions added during the revision and how each item of the ML Checklist remains satisfied for the new material. The original checklist entries above continue to apply unchanged unless explicitly amended below.

### R-1. New experiments added (mapped to revision tasks)

| Task ID | Domain | Benchmark / Source | Items | Reps | Stacks | Notes |
|---------|--------|--------------------|-------|------|--------|-------|
| T1 (code) | Code generation | HumanEval (Chen et al., 2021) | 30 problems | 5 | 8 deployment stacks | Pass@1 via sandboxed execution (`src/tasks/pass_at_1.py`); cross-domain extension requested by Editor and R1/R3 |
| T1 (math) | Mathematical reasoning | GSM8K (Cobbe et al., 2021) | 30 problems | 5 | 8 deployment stacks | Final-answer extraction via numeric-token parser (`src/tasks/gsm8k_extractor.py`) |
| T4 (extension) | Multi-turn / RAG | Original 30 abstracts | 10 abstracts | 5 | gpt-4o-2024-11-20, deepseek-chat | Multi-turn extension to GPT-4o and DeepSeek (closes R3 coverage gap) |
| T14 | Cross-domain (health) | 10 PubMed PM2.5 abstracts | 10 abstracts | 5 | 8 deployment stacks | Light cross-domain probe (epidemiology / air pollution); corpus drawn from sister paper RSM submission |
| D8 (drift) | Stack drift check | Subset of original tasks | subset | 5 | gpt-4o-2024-11-20 | Snapshot-drift control versus original gpt-4-0613 results |

All new runs follow the same Run Card and Prompt Card schema as the original 4,104 runs, the same SHA-256 canonicalisation, and the same W3C PROV-JSON serialization. Records are written to `outputs/revision/runs/` (currently 808 records; final count finalised after T1 completion — flagged TBD in `STATUS.md`).

### R-2. New deployment stacks introduced

| Stack | Provider | Role | Why added |
|-------|----------|------|-----------|
| `gpt-4o-2024-11-20` (OpenAI) | OpenAI | T1 + T4 + D8 drift check | Snapshot-drift control answering R3 item 6: provides a current production stack alongside the legacy `gpt-4-0613` originals |
| Together AI (LLaMA 3 8B INT4) | Together AI | Quasi-isolation probe (already in original) | No change; emphasised under the new "deployment stack" reframe (T5) |

The Methods now define the **unit of analysis as a deployment stack** — the tuple (model weights, provider, serving infrastructure, API layer) — and Table 1 / Table 2 column headings have been changed from `Model` to `Deployment Stack` accordingly. This applies retroactively to the original entries above: every reference to "model" should be read as "deployment stack" in the published version.

### R-3. New analyses added

| Analysis | Script | Output | Justification |
|----------|--------|--------|---------------|
| Per-field reproducibility (BERTScore + EMR per `objective`, `method`, `key_result`) | `analysis/bertscore_per_field.py` | `analysis/bertscore_per_field_results.json`, `analysis/tables/table_per_field_metrics.tex`, `analysis/figures/per_field_radar.pdf` | Responds to R1 by showing BERTScore saturation (≥0.97 across fields) while EMR exposes substantive divergence (Cohen's d = +1.41, conclusion-relevant vs. metadata fields). Validates the three-level reproducibility framework |
| Mechanism × stack mapping table | `article/ncomms_main.tex` (T8 table) | New Methods table linking each candidate mechanism (tensor parallelism, speculative decoding, dynamic batching, KV-cache reuse) to the stacks where it is plausibly active | Responds to R3 mechanism question |
| Revised aggregation across new tasks | `analyze_revision_results.py` | `analysis/revision/emr_per_stack_per_task.json`, `analysis/revision/pass_at_1_humaneval.json`, `analysis/revision/gsm8k_accuracy.json` | Aggregates T1+T4+T14 into the same EMR / NED / ROUGE-L / BERTScore framework as the original tables |

### R-4. Updates to the ML Checklist sections

- **Section 1 (Code and Data):** No change to checks; the same repository now also hosts the revision artefacts under `outputs/revision/`, `data/inputs/revision/`, `analysis/revision/`, and `src/tasks/`. Reviewer access is via the same private link until acceptance, after which the repository becomes public and a Zenodo DOI is minted from the tagged release `v1.1-natcomms-revision1`.
- **Section 2A (Data sources):** Methods §Input data and Supplementary §S2/§S3 updated to list the additional datasets — HumanEval (MIT-licensed), GSM8K (MIT-licensed) and the 10 PubMed PM2.5 abstracts (drawn from the sister paper, Rover & Tadano, RSM under review).
- **Section 2B (Train/test/validation):** Still N/A — this remains an inference-reproducibility study with no model training. HumanEval and GSM8K are used as evaluation prompts only, not as training data.
- **Section 2C (Dataset bias):** Discussion §Limitations now includes an explicit cross-domain caveat — the revision tested coding (HumanEval), mathematical reasoning (GSM8K), and a small health-domain probe (T14). Cross-language generalisation and large-scale clinical applications remain out of scope.
- **Section 4A (Performance metrics):** Pass@1 (HumanEval, sandboxed execution with timeout) and final-answer accuracy (GSM8K) added as task-appropriate metrics alongside EMR, NED, ROUGE-L, and BERTScore. All scripts are in `src/metrics/` and `src/tasks/`.
- **Section 4F (Ablation):** No removal — the original ablations (Supplementary §S6 minimality, Extended Data Table 3 temperature sweep, Supplementary §S7 chat-format control) remain. The drift check (D8, gpt-4o-2024-11-20) functions as an additional ablation over the stack snapshot dimension.
- **Section 5A (Hardware):** Unchanged — Apple M4 / 24 GB / macOS 14.6 / Python 3.14.3 / Ollama v0.15.5. New API-served runs use the same hosted endpoints; total revision API spend ≤ US$50.
- **Section 5B (Computational cost):** Updated to reflect protocol overhead measured on revision runs as well — < 1% logging overhead is preserved across the new task types.

### R-5. Test suite update

Original suite: 51 tests passing (`tests/test_core.py`).
Revision additions: 51 new tests covering HumanEval loader, GSM8K loader, pass@1 sandbox, and cost estimator (`tests/test_humaneval_loader.py`, `tests/test_gsm8k_loader.py`, `tests/test_pass_at_1.py`, `tests/test_cost_estimator.py`).
**Total: 102 tests, all passing on Python 3.14.3** (`python -m pytest tests/ -q`).

### R-6. Items still pending final number insertion

- ⚠ **TBD — Pass@1 per stack (HumanEval):** awaits T1 batch completion (orchestrator `run_revision_full.sh`).
- ⚠ **TBD — Final EMR per stack on T14:** PubMed PM2.5 still in flight at the time of writing this checklist; current runs in `outputs/revision/runs/`.
- ⚠ **TBD — Zenodo DOI:** to be minted at acceptance and inserted into Section 1, the manuscript Data Availability and Code Availability statements, and the corresponding citation block.
