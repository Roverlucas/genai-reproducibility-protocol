# Revision Status — VF4 Coautor-Feedback Checkpoint

**Last update:** 2026-05-19
**Manuscript:** *Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility*
**Target journal:** *Nature Communications* (NCOMMS-2026-XXXXX, Major Revision)
**Editor:** Dr. Marcel Bigorajski (Associate Editor)
**Decision deadline:** ~2026-06-12 (≈24 days margin)

---

## VF4 — Coautor feedback applied (2026-05-19)

Following one coautor's structural feedback on the VF3 bundle (sent 2026-05-12), six surgical improvements were applied. No factual claims changed; only emphasis, organization, and reader-navigation:

| Point | File | Change |
|-------|------|--------|
| P3.1 | `01_point_by_point_response.tex` | R1.9 and R3.6 rewritten: in-paper two-judge LLM-as-judge on 30 new cases now leads as primary evidence; companion paper rebased as "additional larger-corpus context, not load-bearing" |
| P3.2 | `01_point_by_point_response.tex` | New navigation index (10 thematic clusters × 21 reviewer points) inserted after cover note |
| P4.2 | `03_revised_cover_letter.tex` | Flat 11-item list regrouped into 4 macro-clusters: (A) Conceptual reframes, (B) Experimental expansion, (C) Validations & metrics, (D) Editorial infrastructure |
| P4.1 | `03_revised_cover_letter.tex` | New paragraph justifying joint-publication impact: self-contained cause+consequence story + extension of NC reproducibility leadership to LLM research |
| P1.1 | `ncomms_main.tex` | New compact Stack×Mechanisms summary table (`tab:mech_summary`, 3×3) inserted in Results §"Cloud deployment does not preclude reproducibility"; cross-ref to Methods Table 2 detailed |
| P2.1 | `supplementary_nature_mi.tex` | "Section at a glance" italic-bold executive summaries at the start of S9 (Holm-Bonferroni 51/68; Cliff's δ 0.78–0.90), S11 (2,900 new runs, domain-transferable), S12 (73–90% truly contradictory, κ=0.29), S13 (paired Cohen's d=+1.41, BERTScore saturated) |

**Compilation status:** `pdflatex` not available locally — VF4 PDFs need to be regenerated via Overleaf upload of `~/Desktop/overleaf_complete_VF4.zip` before MTS submission.

---

## VF3 — Initial coautor-review snapshot (2026-05-13)

---

## ✅ Technical work complete

### Manuscript (28 pages, 148-word abstract)
- Deployment-stack reframe applied throughout (Abstract → Methods → Discussion)
- 7,004 controlled experiments documented: 4,104 original + 2,900 revision
- 9 deployment stacks, 6 task families
- Bibliography 56/56, no orphan citations
- Figure 5 (visual abstract) regenerated with updated stats (7,004 / 9 / 6 / 3.1×)

### Supplementary Information (23 pages, S1–S13)
- S11 Revision-batch tasks (HumanEval, GSM8K, PubMed PM2.5)
- S12 Two-judge LLM-as-judge protocol (N=30, Claude Opus 4.7 + gpt-4o, κ=0.29)
- S13 Per-field reproducibility analysis (paired Cohen's d=+1.41)
- `longtable` per-field metrics table (auto-paginates across multiple pages)
- `\IfFileExists` wrapper for cross-environment include path

### Response materials
- Point-by-point response: 22 verbatim revquote blocks (R1.1–R1.15 + R3.1–R3.6), 15 pages
- Cover letter: 3 pages with companion-paper offer for parallel NC consideration, dated May 2026
- Tracked manuscript: latexdiff vs T5-snapshot (30 pages with diff markup)

### Checklists (markdown drafts — need transcription to Nature PDF forms)
- `article/ML_CHECKLIST_FILLED.md` — Machine Learning checklist
- `article/REPORTING_SUMMARY_FILLED.md` — Reporting Summary
- `article/CODE_SOFTWARE_CHECKLIST.md` — Code/Software Submission

### Companion paper (OSF preregistered 2026-05-12)
- DOI: [10.17605/OSF.IO/VR934](https://doi.org/10.17605/OSF.IO/VR934)
- Status: manuscript in final preparation (not yet submitted to any journal)
- Cited in this manuscript as `\bibitem{rover2026evidence}` (entry [53])

### Data + code archival
- Figshare DOI: [10.6084/m9.figshare.31653373](https://doi.org/10.6084/m9.figshare.31653373) (CC-BY 4.0)
- Reviewer-private Figshare share URL provided in cover letter
- GitHub tag: `v1.1-natcomms-revision1` (annotated, points at the commit of the revised submission)

### Audit fixes applied (2026-05-13, from advisor Yara Tadano)
- ✓ 3 critical (C-1 Figure 5 stats; C-2 4× CI scope; C-3 0.221 average scope)
- ✓ 3 important (I-1 S10b→S13; I-2 S2-bis→S11; I-3 S4→S6)
- ✓ 9 minor polish items (cover letter Box vs ExtFig; abstract bound; HumanEval Cliff's δ caveat; Cohen's d/h distinction; Q4_0/INT4 alignment; "8 models" clarification; 80× endpoints; thakkar bibitem)

---

## ⏳ Pending (human-only, fora do meu alcance técnico)

1. **Coauthor sign-off** — HVS, ETB, ATA, YST (3–7 days lead time)
2. **Transcribe 3 markdown checklists** to official Nature PDF forms via Adobe Acrobat Reader (~60–90 min)
3. **ORCID profile linking on MTS** — confirm coauthor profiles in Editorial Manager match the manuscript ORCIDs
4. **Final upload via Editorial Manager** — 8 documents (5 PDFs + 3 PDF checklists)

---

## 📦 Deliverable bundles on `~/Desktop/`

### VF4 (current, 2026-05-19, .tex only — needs Overleaf compile)

| File | Size | Purpose |
|------|------|---------|
| `manuscript_VF4.zip` | 204 KB | Overleaf source upload (manuscript + supplementary + figures + cls/bst) |
| `response_letter_VF4.zip` | 23 KB | Overleaf source for point-by-point + cover |
| `overleaf_complete_VF4.zip` | 228 KB | Both above combined — **upload this single ZIP** |
| `submission_mts_VF4.zip` | — | **Not yet generated** — needs PDFs from Overleaf first |

**Next step:** upload `overleaf_complete_VF4.zip` to Overleaf, compile all 4 .tex files (2 passes each for main and SI), download PDFs, then build `submission_mts_VF4.zip` with files renamed `_VF4.pdf`.

### VF3 (previous, 2026-05-13, included PDFs)

| File | Size | Purpose |
|------|------|---------|
| `manuscript_VF3.zip` | 1.77 MB | Overleaf source upload (manuscript + supplementary + figures) |
| `response_letter_VF3.zip` | 611 KB | Overleaf source for point-by-point + cover |
| `overleaf_complete_VF3.zip` | 2.38 MB | Both above combined |
| `submission_mts_VF3.zip` | 2.19 MB | 5 final PDFs (renamed `_VF3.pdf`) for Editorial Manager |

ZIPs are also archived in `overleaf_upload/` and mirrored to Figshare.

---

## Historical archive

- `REVISION_PLAN.md` — 8-decision strategy plan compiled 2026-05-08 with advisor Yara Tadano
- Git tag `v1.1-natcomms-revision1` — submission commit snapshot
- Git history preserves all earlier drafts (pre-revision `nature_mi_main.tex`, Portuguese translation, intermediate `ncomms_cover_letter.tex`, etc.) — removed from working tree to reduce clutter but recoverable via `git log` if needed
