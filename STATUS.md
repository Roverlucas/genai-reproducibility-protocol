# Revision Status — Submission-Ready Checkpoint

**Last update:** 2026-05-13
**Manuscript:** *Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility*
**Target journal:** *Nature Communications* (NCOMMS-2026-XXXXX, Major Revision)
**Editor:** Dr. Marcel Bigorajski (Associate Editor)
**Decision deadline:** ~2026-06-12 (≈30 days margin)

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
