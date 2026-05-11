# Nature Portfolio Reporting Summary — FILLED GUIDE
## "Same Prompt, Different Answer: Exposing the Reproducibility Illusion in Large Language Model APIs"

> Use this guide to fill in the official Nature PDF form (open in Adobe Acrobat Reader).
> Corresponding author: Lucas Rover
> Date: 11-03-2026

---

## 1. Statistics

*For all statistical analyses, confirm that the following items are present in the figure legend, table legend, main text, or Methods section.*

| # | Item | Status | Where in manuscript |
|---|------|--------|-------------------|
| 1 | Exact sample size (n) for each group | **Confirmed** | Table 1 (main text): n per model per task. Supplementary Table S7: full coverage matrix with exact run counts (e.g., 30 abstracts × 5 reps = 150 runs). Methods §Input data. |
| 2 | Statement on distinct vs repeated samples | **Confirmed** | Methods §Experimental conditions: 5 repetitions of the same prompt per abstract (repeated measures). Each abstract is a distinct input. |
| 3 | Statistical test(s) used, one- or two-sided | **Confirmed** | Methods §Statistical analysis: Fisher's exact test (two-sided), Mann–Whitney U (two-sided), Wilcoxon signed-rank (two-sided), bootstrap CIs (10,000 resamples, percentile method). Holm–Bonferroni correction across 68 tests. All tests two-sided. |
| 4 | Description of all covariates tested | **Confirmed** | Methods §Experimental conditions: temperature (0.0, 0.3, 0.7), seed (fixed vs variable), deployment mode (local vs API), task type, prompt format (chat vs completion). **Clarification (Revision T13):** *Deployment mode* refers to whether a model is served locally (single-GPU Ollama on Apple M4) or via a remote cloud API. It is a **stack-level covariate fixed for the lifetime of each experimental group, not a per-run varying parameter.** It is distinct from *decoding mode* (greedy at $t=0$, fixed across deterministic conditions C1–C2) and from *prompt format* (controlled in Supplementary §S7 with the chat-format control experiment). |
| 5 | Assumptions or corrections | **Confirmed** | Methods §Statistical analysis: Holm–Bonferroni correction for 68 multiple comparisons. Non-parametric tests chosen (no normality assumption). Bootstrap percentile method for CIs. Supplementary Section S9 details full correction procedure. |
| 6 | Central tendency and variation/uncertainty | **Confirmed** | All EMR values reported with 95% bootstrap CIs in brackets (e.g., 0.443 [0.32, 0.57]). Table 1, Extended Data Tables 1–5. |
| 7 | Test statistic, CI, effect sizes, df, P values | **Confirmed** | Cohen's d > 1.6 (paired t-test), Cliff's delta 0.784–0.896, Cohen's h 0.40–3.14. P values exact where possible. Supplementary Table S5 (Holm–Bonferroni results for all 68 tests). |
| 8 | Bayesian analysis priors/MCMC | **n/a** | No Bayesian analysis performed. |
| 9 | Hierarchical/complex designs | **n/a** | No hierarchical models. Bootstrap CIs computed at the per-abstract level, then aggregated. |
| 10 | Effect size estimates | **Confirmed** | Cliff's delta (non-parametric effect size) for local vs API: 0.784–0.896. Cohen's d > 1.6. Cohen's h for Fisher comparisons: 0.40–3.14. Reported in Results and Supplementary Section S9. |

---

## 2. Software and code

### Data collection
```
All experiments conducted using custom Python scripts (Python 3.14.3).
Local models: Ollama v0.15.5 serving LLaMA 3 8B, Mistral 7B, Gemma 2 9B.
API models: OpenAI Python SDK v1.59.9 (GPT-4); urllib-based runners for
Anthropic (Claude Sonnet 4.5), Google (Gemini 2.5 Pro), DeepSeek Chat,
Perplexity Sonar, Together AI. All API payloads documented in Supplementary
Section S4. Data collection scripts available in the project repository.
```

### Data analysis
```
Python 3.14.3 with: scipy (statistical tests), scikit-learn (metrics),
rouge-score (ROUGE-L), bert-score (BERTScore), matplotlib and seaborn
(figures), json and hashlib (provenance hashing). Bootstrap CIs computed
with custom script (10,000 resamples). All analysis scripts available in
the project repository at:
https://github.com/Roverlucas/genai-reproducibility-protocol
```

---

## 3. Data

### Data availability statement
```
All 4,104 experimental records, provenance metadata (Run Cards in JSON
format), PROV-JSON provenance graphs, and input abstracts are available
to reviewers during peer review via the project repository at
https://github.com/Roverlucas/genai-reproducibility-protocol. Upon
publication, an archived snapshot with a persistent DOI will be deposited
in Zenodo under a CC-BY 4.0 licence. Source data for all figures and
tables are included in the repository. No restrictions on data
availability apply.
```

---

## 4. Research involving human participants, their data, or biological material

| Field | Response |
|-------|----------|
| Reporting on sex and gender | N/A — This study does not involve human participants. It evaluates the reproducibility of LLM API outputs using published scientific abstracts as input data. |
| Reporting on race, ethnicity, or other socially relevant groupings | N/A — No human participants. |
| Population characteristics | N/A — No human participants. |
| Recruitment | N/A — No human participants. |
| Ethics oversight | No ethics approval was required. The study uses only publicly available scientific abstracts and commercial/open-source LLM APIs. No human subjects, personal data, or sensitive information were involved. |

---

## 5. Field-specific reporting

**Select: NONE of the three options apply directly.**

> Our study is a computational experiment (software engineering / AI systems evaluation). It does not involve life sciences experiments, behavioural/social sciences with human subjects, or ecological/environmental fieldwork. If forced to choose, select "Life sciences" and fill the fields as follows:

### If selecting "Life sciences study design":

| Field | Response |
|-------|----------|
| **Sample size** | 30 scientific abstracts × 8 models × 4 tasks × up to 5 conditions × 5 repetitions = 4,104 total runs (3,904 unique + 200 chat-format controls). Sample size was determined by the need to achieve sufficient statistical power for detecting large effects (Cohen's d > 1.6). The balanced subsample analysis (Extended Data Table 4) confirms robustness with n = 10 abstracts. |
| **Data exclusions** | One Claude API run excluded due to API timeout returning empty output (49 of 50 runs retained). GPT-4 C1 summarization limited to 3 abstracts (8 runs) due to API quota exhaustion. All exclusions documented in Supplementary Table S7 and noted in figure/table legends. No post-hoc exclusions of completed runs. |
| **Replication** | Every experimental condition was replicated 5 times per abstract (or 3 times for temperature sweep conditions C3). Replication is the core subject of the study. All 4,104 runs are available in the repository for independent verification. Bootstrap CIs (10,000 resamples) quantify sampling uncertainty. |
| **Randomization** | Not applicable in the traditional sense. All models received identical inputs in identical order. Seed values for variable-seed condition (C2): {42, 123, 456, 789, 1024}. No randomization of input order was needed as each abstract is processed independently. |
| **Blinding** | Not applicable. The study compares computational outputs across known model identities. Blinding is not meaningful in this context as the analysis is fully automated (metric computation via scripts) with no subjective assessment. |

---

## 6. Reporting for specific materials, systems and methods

### Materials & experimental systems

| Item | Status |
|------|--------|
| Antibodies | **n/a** |
| Eukaryotic cell lines | **n/a** |
| Palaeontology and archaeology | **n/a** |
| Animals and other organisms | **n/a** |
| Clinical data | **n/a** |
| Dual use research of concern | **n/a** |
| Plants | **n/a** |

### Methods

| Item | Status |
|------|--------|
| ChIP-seq | **n/a** |
| Flow cytometry | **n/a** |
| MRI-based neuroimaging | **n/a** |

---

## NOTES FOR FILLING THE PDF

1. Open the PDF in **Adobe Acrobat Reader** (not Preview.app — XFA forms don't render in Preview)
2. For each checkbox item in Section 1: click "Confirmed" for items marked Confirmed above, "n/a" for items marked n/a
3. Copy-paste the text blocks above into the corresponding text fields
4. Save as PDF
5. The Reporting Summary will be **published alongside the paper** if accepted

---

## Revision additions (2026-05-08, Major Revision response — Nature Communications)

The following items extend the entries above with the experiments and analyses introduced during the revision. Original entries remain valid; these additions document the new material the editor and reviewers requested.

### Sample size (revised)

The revised manuscript reports the original 4,104 runs **plus** new revision runs added under tasks T1 (HumanEval and GSM8K), T4 (multi-turn extension to gpt-4o-2024-11-20 and DeepSeek), and T14 (10 PubMed PM2.5 abstracts). At the time of submission of the revised manuscript, the revision-specific runs are stored in `outputs/revision/runs/` and aggregated by `analyze_revision_results.py`.

| Revision task | Items × Reps × Stacks | Approximate runs |
|---------------|-----------------------|------------------|
| T1 HumanEval | 30 × 5 × 8 | up to 1,200 |
| T1 GSM8K | 30 × 5 × 8 | up to 1,200 |
| T4 multi-turn extension | 10 × 5 × 2 | 100 |
| T14 PubMed PM2.5 | 10 × 5 × 8 | 400 |

Final per-task and per-stack run counts are reported in the revised Table 1 and Supplementary Table S7 once the orchestrator completes (final numbers flagged ⚠ TBD in `STATUS.md`).

### Number of replicates (revised)

- New tasks (T1 HumanEval, T1 GSM8K, T4 extension, T14): **5 repetitions per problem per stack per condition**, identical to the original protocol.
- Drift check (D8, gpt-4o-2024-11-20 vs. legacy gpt-4-0613): 5 repetitions on a matched subset of the original tasks.

### Statistical methods (revised)

- All revision metrics (EMR, NED, ROUGE-L, BERTScore F1, Pass@1, GSM8K answer accuracy) are reported with **95 % confidence intervals computed by 10,000-resample percentile bootstrap**, using the same `analysis/bootstrap_cis.py` routine as the original submission.
- The new per-field analysis (T6) computes **per-field EMR and BERTScore F1 over `objective`, `method`, and `key_result`**, with the same 10,000-resample bootstrap (`analysis/bertscore_per_field.py`). Pairwise comparisons (conclusion-relevant vs. metadata fields) report Cohen's d with 95 % bootstrap CIs.
- Holm–Bonferroni correction is extended to cover the additional pairwise comparisons introduced by T1, T4, and T14; the full corrected table appears in Supplementary §S9.
- All significance tests remain two-sided and non-parametric (Fisher's exact, Mann–Whitney U, Wilcoxon signed-rank), consistent with the original submission.

### Software and code (revised)

The Software and code blocks in §2 are extended to include the following revision-specific scripts and modules:

- `src/tasks/humaneval_loader.py`, `src/tasks/gsm8k_loader.py`, `src/tasks/gsm8k_extractor.py`, `src/tasks/pass_at_1.py`, `src/tasks/llm_judge.py`, `src/tasks/pm25_case_loader.py`, `src/tasks/pubmed_loader.py`
- `analysis/bertscore_per_field.py`, `analysis/revision/` aggregation outputs
- `run_revision_experiments.py`, `run_revision_full.sh`, `analyze_revision_results.py`
- New tests (`tests/test_humaneval_loader.py`, `tests/test_gsm8k_loader.py`, `tests/test_pass_at_1.py`, `tests/test_cost_estimator.py`) bringing the suite from 51 to 102 tests, all passing.

### Data availability (revised)

The Data Availability statement in §3 also covers `outputs/revision/runs/` and `data/inputs/revision/`. A Zenodo DOI minted from release tag `v1.1-natcomms-revision1` will be inserted at acceptance.
