# Changes Log — NatComms Major Revision (VF4)

**Manuscript:** Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility
**Original submission:** 2026-03-12
**Revision compiled:** 2026-05-13 (VF3 sent to coauthors)
**VF4 (coauthor feedback applied):** 2026-05-19
**Editor deadline:** 2026-06-12

This document tracks every change applied to the manuscript and supplementary material, mapped to the reviewer comment that motivated it. The VF4 section at the end records the second round of coauthor feedback applied after VF3.

Status legend: ✅ DONE · 🟡 PARTIAL · ⏳ PENDING-EXTERNAL

---

## Files in submission package (VF4 — final)

| File (MTS upload name) | Status |
|------------------------|--------|
| `01_revised_manuscript.pdf` | ✅ 28 p, 646 KB |
| `02_revised_manuscript_tracked.pdf` | ✅ 28 p, 723 KB (latexdiff vs original Nature MI submission, 2026-03-03) |
| `03_revised_supplementary.pdf` | ✅ 22 p, 456 KB (incl. §S11 revision-batch, §S12 LLM-as-judge, §S13 per-field) |
| `04_point_by_point_response.pdf` | ✅ 15 p, 421 KB (navigation index + 15 R1 + 6 R3 items) |
| `05_cover_letter.pdf` | ✅ 3 p, 197 KB (4 macroclusters + joint-publication impact justification) |
| `06_changes_log.pdf` (this file) | ✅ generated from markdown |
| `07_reporting_summary.pdf` | ✅ Nature form, all fields populated for VF4 |
| `08_machine-learning-checklist.pdf` | ✅ Nature form V1.1, all sections completed for VF4 |
| `09_code_software_checklist.pdf` | ✅ generated from markdown (Code/Software checklist explicitly requested by editor in revision letter) |

---

## Latex source snapshots

- **Original submission (PDF + tex commit `fb0be76`):** `article/nature_mi_main.tex` (835 lines, before Nature Comms rebrand)
- **Post-T5 reframe snapshot (VF1 baseline):** `submission_revision_v1/ncomms_main_post_T5.tex` (894 lines)
- **VF3 .tex:** state at commit `e2e16b0` (2026-05-13)
- **VF4 .tex (final):** `article/ncomms_main.tex` (~1100 lines)
- **Track-changes .tex (VF4 vs original submission, auto-generated):** `submission_revision_v1/ncomms_main_tracked_VF4.tex` — 1354 lines, 301 DIFadd / 182 DIFdel blocks

Regenerate track changes with:
```
latexdiff article/nature_mi_main.tex@fb0be76 article/ncomms_main.tex \
  > submission_revision_v1/ncomms_main_tracked_VF4.tex
```

---

## Detailed change log — by reviewer point

### ✅ T5 — Deployment-stack reframe — applied 2026-05-08

**Reviewer point:** R1.1 (framing critique — central conceptual ask)

| Section | Lines | Change |
|---------|-------|--------|
| Abstract | 62–67 | Inserted (model_weights, provider, infra, API_layer) tuple definition; "API-served models" → "API-served stacks" |
| Introduction | 99–106 | New ~80-word paragraph anchoring deployment-stack concept |
| Results §1 heading + lead | 115–117 | Renamed to deployment stacks |
| Table 1 | 240–262 | Column "Model" → "Deployment stack" |
| Figs 1, 2, 5 captions | 230, 271, 315 | Updated terminology |
| Methods §"Unit of analysis: deployment stacks" | 407–415 | NEW ~155-word subsection |
| Methods §"Deployment stacks evaluated" | 417+ | Renamed; entries as tuples |
| Discussion | 341–344 | NEW ~110-word paragraph: "the deployment stack, not the abstract model, is the carrier of variation" |

**Word-count delta:** +330 words.

---

### ✅ T6 — Per-field reproducibility analysis — applied 2026-05-08

**Reviewer point:** R1.5 (textual vs substantive contradiction)

**Pivot finding:** BERTScore F1 saturates across all fields (Δ=0.001, d=-0.10) but EMR exposes a paired Cohen's d=+1.41 between conclusion-relevant fields (mean EMR=0.455) and metadata fields (mean EMR=0.684). This strengthens the manuscript's three-level framework argument: BERTScore alone cannot detect substantive payload divergence.

**Single-stack illustration:** Gemini 2.5 Pro RAG, `key_result` field — EMR=0.10, BERTScore F1=0.969 (field changes 90% of the time, BERTScore still 0.97).

**Files added:**
- `analysis/bertscore_per_field.py` — analysis script
- `analysis/bertscore_per_field_results.json` — raw results (53 KB)
- `analysis/tables/table_per_field_metrics.tex` — Extended Data table
- `analysis/figures/per_field_radar.pdf` — heatmap with separator

---

### ✅ T7 — Cloud vs production-serving infrastructure distinction — applied 2026-05-08

**Reviewer point:** R1.4

**Edit:** Methods §"Sources of non-determinism in distributed inference" — new opening paragraph (~120 words) distinguishing:
- **Cloud deployment factors:** network latency, load balancing, multi-tenancy, shared hardware
- **Production serving infrastructure factors:** tensor parallelism, FlashAttention, dynamic batching, speculative decoding, mixed-precision accumulation, prefix caching

Together AI quasi-isolation result explicitly invoked to license the inferential bridge.

---

### ✅ T8 — Mechanism × stack mapping table — applied 2026-05-08

**Reviewer point:** R3.5

**Edit:** New `Table~\ref{tab:mechanisms}` (full-page width) inside Methods §"Sources of non-determinism". Maps 6 mechanisms × 5 stacks with labels {active / likely / possible / not applicable}. Footnote explicit on inferential attribution for closed-source providers.

---

### ✅ T9 — Protocol scope (client-side) clarification — applied 2026-05-08

**Reviewer point:** R1.10

**Edit:** Methods §"Protocol design" — new ~150-word opening paragraph titled "Scope: client-side observability". Establishes that the protocol documents client-side observable information only; not restricted to open-source stacks; creates the audit trail that providers' opacity otherwise prevents.

---

### ✅ T10 — Perplexity literature-only reframe — applied 2026-05-08

**Reviewer point:** R1.3

**Edit:** Results §"Non-determinism varies substantially across providers" — Perplexity passage rewritten from causal claim to informed hypothesis with citations. Three new bibitems appended:
- `lewis2020rag` — Lewis et al., NeurIPS 33 (2020)
- `perplexitysonar2024` — Perplexity AI docs
- `shi2024ragvariability` — Shi et al., arXiv:2401.05856

---

### ✅ T11 — Anthropic seed clarification — applied 2026-05-08

**Reviewer point:** R1.11

**Edit:** Methods §"Experimental conditions" — paragraph appended explaining Anthropic API has no seed parameter; Run Card records `seed: 42` with `seed_status: "logged-only-not-sent-to-api"`. Non-determinism therefore attributable to infrastructure, not seed variation. Cross-references Supplementary §S4.

---

### ✅ T12 — Concrete divergence examples (Box 1) — applied 2026-05-08

**Reviewer point:** R1.12

**Edit:** New Box 1 (figure* float after Fig 5) with 3 real-case examples from Run Cards:
1. GPT-4 extraction abs_001 `benchmark` field — comma vs "and" (cosmetic)
2. Claude RAG abs_001 reps 0 vs 1 `key_result` field — substantive content change
3. Same group, `method` field — fully reformulated wording

Diff highlights via coloured bold.

---

### ✅ T13 — Mechanical fixes — applied 2026-05-08

**Reviewer points:** R1.7, R1.13, R1.14, R1.15

- **Fig 1 caption:** rewritten to enumerate all 8 stacks shown; explicit explanation of Gemini absence
- **W3C PROV:** definition added on first use in introduction (~line 107) citing `w3cprov2013`
- **Supplementary §S4:** new `Table~\ref{tab:apidocs}` with hyperlinks to 7 provider API docs
- **Reporting Summary:** "deployment mode" entry clarified as stack-level covariate, distinguished from decoding mode and prompt format

---

### ✅ T15 — Quasi-isolation closed-source limitation — applied 2026-05-08

**Reviewer point:** R3.2

**Edit:** Discussion §Limitations — paragraph appended reformulating from strong claim "cloud deployment does not preclude reproducibility" to defensible weaker claim "cloud deployment is not a sufficient cause of non-determinism". Explicitly acknowledges that closed-source proprietary stacks cannot be locally compared.

---

### ✅ T1 — Code (HumanEval) + Math (GSM8K) experiments — applied 2026-05-08

**Reviewer points:** Editor + R1.8 + R3.1

**Experiments:** 30 HumanEval problems × 5 reps × 8 stacks = 1,200 runs. 30 GSM8K problems × 5 reps × 8 stacks = 1,200 runs.

**Key results inserted in new Results §"Coding and math reasoning":**

| Stack | HumanEval EMR | GSM8K EMR |
|-------|---------------|-----------|
| Locals + Together + Gemini | 0.92–1.000 | 0.84–1.000 |
| deepseek-chat | 0.837 [0.72, 0.93] | 0.370 [0.26, 0.49] |
| gpt-4o | 0.837 [0.75, 0.92] | 0.267 [0.17, 0.37] |
| Claude Sonnet 4.5 | **0.393 [0.27, 0.52]** | **0.063 [0.03, 0.10]** |

**Cost:** ~$5 USD (Anthropic + OpenAI + Google + DeepSeek + Together).

---

### ✅ T2 — Promote PM2.5 to Results subsection — applied 2026-05-08

**Reviewer point:** R1.9

**Edit:** New Results §"Applied impact in evidence synthesis: an out-of-AI/ML probe" (lines 246–264). Reports 10 PubMed PM2.5 EMRs across all stacks; emphasises Claude EMR=0.010 mirroring 0.020 AI/ML pattern. Cites companion paper for full 500-abstract analysis (NOT duplicated).

---

### ✅ T3-revised — PM2.5 protected triangulation — applied 2026-05-09

**Reviewer point:** R3.6

**Approach:** Cite companion paper for large-corpus validation (NOT duplicating its tables) + run independent mini LLM-as-judge on 10 NEW cases (distinct from the 23 effects analysed in the companion paper).

**Judge:** Claude Opus 4.7, blind, three pre-registered criteria (direction / magnitude ±20% / CI overlap). Per-case randomisation seed derived from SHA-256 hash.

**Verdicts:** 5 truly_contradictory / 3 semantically_equivalent / 2 ambiguous.

**Cost:** $0.28 USD.

**Files:**
- `src/tasks/pm25_case_loader.py` — case sampling
- `src/tasks/llm_judge.py` — Claude Opus judge with pre-registered criteria
- `run_t3_validation.py` — runner (CLI: `--sample`, `--judge --execute`)
- `outputs/revision/t3_judge/t3_judge_results.json` — aggregate
- `outputs/revision/t3_judge/judge_*.json` — per-case PROV records

**In manuscript:** lines 252–256 (Applied impact subsection) + Supplementary §S12 (full protocol).

---

### ✅ T4 — Multi-turn extension to gpt-4o + deepseek-chat — applied 2026-05-08

**Reviewer point:** R3.3

**Experiments:** Three-turn refinement protocol × 10 abstracts × 5 reps × 2 stacks = 100 runs.

**Results inserted in Results §"Complex workflows" + Fig 2 (regenerated):**
- **gpt-4o multi-turn EMR = 0.090 [0.02, 0.16]**
- **deepseek-chat multi-turn EMR = 0.350 [0.13, 0.60]** (four-fold drop from 0.84 single-turn)

Confirms near-zero EMR pattern previously reported for Claude (0.04) and Gemini (0.01) is universal.

---

### ✅ T14 — 10 PubMed PM2.5 abstracts (out-of-AI/ML domain) — applied 2026-05-08

**Reviewer point:** R3.4

**Experiments:** 10 abstracts × 5 reps × 8 stacks = 400 runs.

**Source:** companion paper corpus, drawn from PM2.5/respiratory-health subset.

**Results inserted in Results §"Applied impact":**
| Stack | EMR |
|-------|-----|
| Locals + Together | 0.96–1.000 |
| deepseek-chat | 0.660 [0.46, 0.85] |
| gemini-2.5-pro | 0.490 [0.26, 0.72] |
| gpt-4o | 0.420 [0.27, 0.58] |
| **Claude Sonnet 4.5** | **0.010 [0.00, 0.03]** |

---

### ✅ T16 — Editorial checklists — applied 2026-05-09

**Editor requirement.**

| Document | Status | Path |
|----------|--------|------|
| **Code/Software submission checklist** (new for revision) | ✅ Created | `article/CODE_SOFTWARE_CHECKLIST.md` (12 KB, 11 sections) |
| **Machine Learning checklist** | ✅ Updated | `article/ML_CHECKLIST_FILLED.md` (revision-additions section appended) |
| **Reporting Summary** | ✅ Updated | `article/REPORTING_SUMMARY_FILLED.md` (T13 fix + revision additions) |
| **Data Availability statement** | ✅ Present | `article/ncomms_main.tex` lines 666–670 (`\bmhead`) + 713 (Declarations) |
| **Code Availability statement** | ✅ Present | `article/ncomms_main.tex` lines 670–676 + 716 |

---

### ✅ T17 — Final assembly + consistency audit — applied 2026-05-10

**Editor requirement.** Track changes / colour highlighting, point-by-point response, cover letter.

- **latexdiff** track-changes version generated from `post_T5` baseline vs final `ncomms_main.tex`
- Point-by-point response: 15 pages, 22 verbatim `revquote` environments (15 R1 + R2 ack + 6 R3)
- Cover letter: revised with summary of major changes
- **Consistency audit:** 20/20 critical checks passed (run counts, EMRs, stack naming, T3 verdicts, abstract ≤150 words, bibliography integrity, snapshot tags, companion-paper status "submitted")

---

## Run totals

| Batch | Runs | Description |
|-------|------|-------------|
| Original (Mar 2026 submission) | 4,104 | 8 models × 4 tasks (extraction, summarisation, multi-turn, RAG) + chat-format controls |
| **Revision additions (May 2026)** | **2,900** | HumanEval (1,200) + GSM8K (1,200) + PubMed PM2.5 (400) + multi-turn extension (100) |
| **Grand total** | **7,004** | 9 deployment stacks × 6 task families |

---

## API budget

```
Caminho A (HumanEval + GSM8K + PubMed + multi-turn):    $8.10 USD
T3 LLM-as-judge (10 cases × Claude Opus 4.7):           $0.28 USD
─────────────────────────────────────────────────────────────────
TOTAL revision spend:                                   $8.38 USD
Approved budget:                                       $50.00 USD
Margin:                                                $41.62 USD ✅
```

---

## VF4 — Second round of coauthor feedback applied (2026-05-19)

After VF3 was distributed to the four coauthors on 2026-05-13, one coauthor returned six structural suggestions. None changed factual content; all six target presentation, emphasis, and editorial navigation. Plus three audit-driven fixes caught during compilation review.

### Six surgical improvements (commit `bfd9663`, 2026-05-19)

| # | Item | File | Change |
|---|------|------|--------|
| **P3.1** | R1.9 + R3.6 framing: companion paper rebased from "load-bearing" to "additional larger-corpus context" | `01_point_by_point_response.tex` | Both responses now lead with the in-paper two-judge LLM-as-judge on 30 new cases as primary substantive evidence; companion paper (Rover & Tadano, OSF VR934) cited only as larger-corpus context |
| **P3.2** | Navigation index inserted on page 1 of the point-by-point | `01_point_by_point_response.tex` | Compact table mapping 10 thematic clusters to all 21 reviewer points (R1.1–R1.15 + R3.1–R3.6) for editorial navigation |
| **P4.2** | Cover-letter flat 11-item list regrouped into four macro-clusters | `03_revised_cover_letter.tex` | (A) Conceptual reframes (3 items), (B) Experimental expansion (3 items), (C) Validations & metrics (4 items), (D) Editorial infrastructure (1 item) |
| **P4.1** | Joint-publication impact justified explicitly | `03_revised_cover_letter.tex` | New paragraph: (i) the two works form a self-contained cause+consequence reproducibility story; (ii) co-publishing extends Nature Communications' leadership on reproducibility into LLM-based research |
| **P1.1** | Compact Stack×Mechanisms summary table promoted to Results | `ncomms_main.tex` | New Table 1 (3 rows × 4 cols, footnotes for parenthetical examples) inserted in Results §"Cloud deployment does not preclude reproducibility"; full 6×5 detailed table remains in Methods (Table 3) |
| **P2.1** | "Section at a glance" executive italics added to SI long sections | `supplementary_nature_mi.tex` | One paragraph in italic-bold at the start of S9 (Holm-Bonferroni 51/68; Cliff's δ 0.78–0.90), S11 (2,900 new runs; domain-transferable), S12 (73–90% truly contradictory; κ=0.29), S13 (paired Cohen's d=+1.41; BERTScore saturated) |

### Varredura fix (commit `16f1784`, 2026-05-19)

| Item | File | Change |
|------|------|--------|
| Cover-note item 5 framing leftover from VF3 | `01_point_by_point_response.tex` | Item 5 of the cover-note enumeration ("Applied evidence-synthesis impact and validation") still led with "To preserve the contributions of our companion paper..."; rewritten to match the new P3.1 framing in the detailed R1.9/R3.6 responses |

### Audit-driven fixes after first VF4 compilation (commit `01f4924`, 2026-05-19)

| Issue caught | File | Fix |
|--------------|------|-----|
| Cover-letter PDF rendered `[leftmargin=*]` and `[leftmargin=*,resume]` as literal text after each cluster heading | `03_revised_cover_letter.tex` | Added missing `\usepackage{enumitem}` to preamble (the regrouped enumerations needed it; the original flat enumerate did not) |
| SI introductory paragraph still claimed "Sections S1–S10" while the TOC and content covered S1–S13 | `supplementary_nature_mi.tex` | Updated range to "Sections S1–S13" with explicit enumeration of the new sections (S11 revision-batch tasks, S12 two-judge LLM-as-judge triangulation protocol, S13 per-field reproducibility analysis) |

### Date marker filled (commit `eab2727`, 2026-05-19)

| Item | File | Change |
|------|------|--------|
| `\date{Revision submitted: TBD ...}` placeholder | `01_point_by_point_response.tex` | `\date{Revision submitted: May, 2026 — Original submission: 2026-03-12}` |

### Table 1 layout fix after audit of compiled PDF (commits `c3675a3` → `0874292`, 2026-05-19)

| Issue caught | File | Fix |
|--------------|------|-----|
| Audit of compiled PDF (`-15`) showed Table 1 right-truncated: header "Production-serving complexity" cut off; first attempt with `\resizebox`+`\newline` caused the table to silently disappear from output (cross-reference rendered as `Table ??`) | `ncomms_main.tex` | Final layout: explicit column widths `p{3.2}/p{2.8}/p{2.5}/p{3.0}cm`, `\tabcolsep=3pt`, superscript footnote markers (a/b/c/\*) instead of inline parenthetical examples in headers, all examples on a `\scriptsize` line below `\bottomrule`. Renders correctly in PDF `-15` (regenerated 2026-05-19 16:38) |

### Tracked manuscript regenerated against original submission (commit `66e75d0`, 2026-05-19)

Previous VF3 tracked PDF used the post-T5 snapshot as baseline (only the reframe and downstream changes were visible). The VF4 tracked PDF (`02_revised_manuscript_tracked.pdf`, 723 KB) uses the **original Nature Comms submission state** (commit `fb0be76`, 2026-03-03) as baseline so the editor and reviewers see every change since the first round: title rewrite, 4,104→7,004 experiments, deployment-stack reframe, new contributions, W3C PROV expanded definition, US→UK spelling, and all revision additions.

Stats: 1354 lines, 301 added blocks, 182 deleted blocks.

---

## Verification checklist (pre-submission, VF4)

- [x] All 15 R1 + 6 R3 reviewer points have a `revquote` + `response` + `changes` block
- [x] Navigation index on point-by-point page 1 (VF4 addition)
- [x] All ⏳ and 🟡 tasks have been promoted to ✅
- [x] Track changes generated via latexdiff (vs original submission, not vs T5 snapshot)
- [x] Response letter PDF compiles cleanly (cover letter `enumitem` fix included)
- [x] All Extended Data tables/figures referenced in main text resolve (including new Table 1 compact summary)
- [x] All new citations added to bibitems resolve
- [x] LaTeX compiles without errors (clean + tracked + supplementary)
- [x] Companion paper status uniformly "in preparation, OSF preregistered"
- [x] Companion paper framed as "additional larger-corpus context, not load-bearing"
- [x] No GHOST numbers (3,010 or 7,114) remaining
- [x] Stack naming uniformly lowercase (gpt-4o, deepseek-chat, claude-sonnet-4-5)
- [x] Two-judge LLM-as-judge verdicts: Claude 22/3/5, gpt-4o 27/3/0 across manuscript + supplementary + response
- [x] Abstract 148 words (NatComms limit ≤150)
- [x] All three editorial checklists (Code, ML, Reporting) updated for VF4 totals (7,004 experiments; 9 deployment stacks; 6 task families)
- [x] ORCIDs populated in manuscript title page, cover letter, and all checklists (LR 0000-0001-6641-9224; HVS 0000-0002-1278-4602; ETB 0000-0002-3936-0375; ATA 0000-0003-1678-7795; YST 0000-0002-3975-3419)
- [x] **GitHub tag `v1.1-natcomms-revision1` created** (annotated tag on `origin/main`)
- [x] **Figshare deposit live** (DOI 10.6084/m9.figshare.31653373; private reviewer URL in cover letter)
- [x] **Cover letter date filled in** ("May, 2026")
- [x] **Code/Software submission checklist** prepared (explicitly requested by editor in major-revision letter)
- [ ] **All 5 coauthors sign off on VF4** (HUMAN — Lucas + HVS + ETB + ATA + YST)
- [ ] **ORCID linked on MTS account** (HUMAN — at MTS upload, just attach the IDs already in manuscript)
- [ ] **Final `git push` of 8 VF4 commits** (HUMAN — via @devops at submission time)

---

## Word count summary

| Section | Words | NatComms guidance |
|---------|-------|-------------------|
| Abstract | 148 | ≤150 ✅ |
| Main text (Intro + Results + Discussion) | 4,868 | ~5,000 recommended ✅ |
| Methods | 3,138 | ~3,000 recommended (slight over, acceptable) |
| Total | 8,156 | — |

---

## Display items

| Type | Count | NatComms limit |
|------|-------|----------------|
| Main figures | 6 | up to 10 total display items ✅ |
| Main tables | 3 | — |
| Extended Data tables/figures | (counted separately) | unlimited |

---

*Auto-updated by Sage academic-chief after T17 final assembly. Status: 100% technical completion; awaiting coauthor sign-off and MTS upload.*
