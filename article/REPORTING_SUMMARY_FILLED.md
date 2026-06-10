# Nature Portfolio Reporting Summary — content for the official form
## "Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility"

> Content to transcribe into the official Nature Portfolio Reporting Summary PDF (XFA form).
> Corresponding author: Lucas Rover (ORCID: 0000-0001-6641-9224)
> Coauthors: Hugo Valadares Siqueira (0000-0002-1278-4602); Eduardo Tadeu Bacalhau (0000-0002-3936-0375); Anibal Tavares de Azevedo (0000-0003-1678-7795); Yara de Souza Tadano (0000-0002-3975-3419)

---

## 1. Statistics

*For all statistical analyses, confirm that the following items are present in the figure legend, table legend, main text, or Methods section.*

| # | Item | Status | Where in manuscript |
|---|------|--------|-------------------|
| 1 | Exact sample size (n) for each group | **Confirmed** | Table 1 (main text): n per deployment stack per task. Supplementary Table S7: full coverage matrix (e.g., 30 abstracts × 5 reps = 150 runs). Total: 7,004 runs. Methods §Tasks and §Input data. |
| 2 | Statement on distinct vs repeated samples | **Confirmed** | Methods §Experimental conditions: 5 repetitions of the same prompt per item (repeated measures). Each abstract / HumanEval problem / GSM8K problem is a distinct input. |
| 3 | Statistical test(s) used, one- or two-sided | **Confirmed** | Methods §Statistical analysis: Fisher's exact test (two-sided), Mann–Whitney U (two-sided), Wilcoxon signed-rank (two-sided), bootstrap CIs (10,000 resamples, percentile method). Holm–Bonferroni correction across 68 tests. All tests two-sided. |
| 4 | Description of all covariates tested | **Confirmed** | Methods §Experimental conditions: temperature (0.0, 0.3, 0.7), seed (fixed vs variable), deployment mode (local vs API), task type, prompt format (chat vs completion). *Deployment mode* refers to whether a stack is served locally (single-GPU Ollama on Apple M4) or via a remote cloud API; it is a stack-level covariate fixed for the lifetime of each experimental group, not a per-run varying parameter. It is distinct from *decoding mode* (greedy at t=0, fixed across deterministic conditions C1–C2) and from *prompt format* (controlled in Supplementary §S7 with the chat-format control experiment). |
| 5 | Assumptions or corrections | **Confirmed** | Methods §Statistical analysis: Holm–Bonferroni correction for 68 multiple comparisons. Non-parametric tests chosen (no normality assumption). Bootstrap percentile method for CIs. Supplementary §S9 details the full correction procedure. |
| 6 | Central tendency and variation/uncertainty | **Confirmed** | All EMR values reported with 95% bootstrap CIs in brackets (e.g., 0.443 [0.32, 0.57]). Table 1, Extended Data Tables 1–7. |
| 7 | Test statistic, CI, effect sizes, df, P values | **Confirmed** | Cliff's δ 0.784–0.896 (large to very large); Cohen's d > 1.6 (paired); Cohen's h 0.40–3.14. Per-field paired Cohen's d = +1.41 (conclusion-relevant vs metadata fields). P values exact where possible. Supplementary §S9 (Holm–Bonferroni results for all 68 tests). |
| 8 | Bayesian analysis priors/MCMC | **n/a** | No Bayesian analysis performed. |
| 9 | Hierarchical/complex designs | **n/a** | No hierarchical models. Bootstrap CIs computed at the per-item level, then aggregated. |
| 10 | Effect size estimates | **Confirmed** | Cliff's δ (non-parametric) for local vs API: 0.784–0.896. Cohen's d > 1.6 (paired). Cohen's h 0.40–3.14 (Fisher). Per-field Cohen's d = +1.41. Reported in Results and Supplementary §S9. |

---

## 2. Software and code

**Data collection.** All experiments conducted using custom Python scripts (Python 3.14.3). Local stacks: Ollama v0.15.5 serving LLaMA 3 8B, Mistral 7B, Gemma 2 9B. API stacks: OpenAI Python SDK v1.59.9 (GPT-4 gpt-4-0613 and gpt-4o-2024-11-20); urllib-based runners for Anthropic (Claude Sonnet 4.5), Google (Gemini 2.5 Pro), DeepSeek Chat, Perplexity Sonar, Together AI (LLaMA 3 8B INT4 quasi-isolation probe). Loaders for HumanEval and GSM8K via standard benchmark formats. All API payloads documented in Supplementary §S4. Data collection scripts available in the project repository.

**Data analysis.** Python 3.14.3 with: scipy (statistical tests), scikit-learn (metrics), rouge-score (ROUGE-L), bert-score (BERTScore F1, roberta-large), matplotlib and seaborn (figures), json and hashlib (provenance hashing). Bootstrap CIs computed with custom script (10,000 resamples, percentile method). Two-judge LLM-as-judge triangulation via Claude Opus 4.7 and gpt-4o. Per-field analysis via a dedicated script. All analysis scripts available at https://github.com/Roverlucas/genai-reproducibility-protocol

---

## 3. Data

**Data availability statement.** All 7,004 experimental records, provenance metadata (Run Cards in JSON format), W3C PROV-JSON provenance graphs, input abstracts (30 AI/ML + 10 PubMed PM2.5), HumanEval and GSM8K problem identifiers, and two-judge LLM-as-judge verdicts are deposited on Figshare with persistent DOI 10.6084/m9.figshare.31653373 (CC-BY 4.0). The Figshare deposit mirrors the project GitHub repository at https://github.com/Roverlucas/genai-reproducibility-protocol (release tag v1.1). Source data for all figures and tables are included. No restrictions on data availability apply.

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
| **Sample size** | Total 7,004 logged generative-AI runs across 9 deployment stacks and 6 task families: 30 AI/ML abstracts × 8 stacks × up to 4 tasks × up to 5 conditions × 5 reps; 30 HumanEval problems × 5 reps × 8 stacks; 30 GSM8K × 5 × 8; 10 PubMed PM2.5 × 5 × 8; multi-turn extension to gpt-4o and DeepSeek (50 runs each). Sample size determined to achieve power for detecting large effects (Cohen's d > 1.6). Balanced 10-abstract subsample (Extended Data Table 4) confirms robustness. |
| **Data exclusions** | One Claude API run excluded due to API timeout returning empty output (49/50 retained). GPT-4 C1 summarisation limited to 3 abstracts (8 runs) due to API quota exhaustion. All exclusions documented in Supplementary Table S7 and noted in figure/table legends. No post-hoc exclusions of completed runs. |
| **Replication** | Every experimental condition replicated 5 times per item (or 3 times for temperature sweep conditions). Replication is the core subject of the study — reproducibility under fixed deterministic settings. All 7,004 runs available in the repository (Figshare DOI 10.6084/m9.figshare.31653373) for independent verification. Bootstrap CIs (10,000 resamples) quantify sampling uncertainty. |
| **Randomization** | Not applicable in the traditional sense. All stacks received identical inputs in identical order. Seed values for variable-seed condition (C2): {42, 123, 456, 789, 1024}. Two-judge LLM-as-judge sampling: 30 cases stratified across stack and disagreement kind. |
| **Blinding** | The two-judge LLM-as-judge protocol used blinded comparison: each judge scored pair (A, B) with stack/run identifiers withheld, against three pre-registered criteria (direction, magnitude ±20%, CI overlap). For the rest of the study, blinding is not meaningful — automated metric computation across known stack identities. |

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
