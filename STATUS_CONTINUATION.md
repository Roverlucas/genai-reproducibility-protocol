# STATUS & CONTINUATION — NatComms resubmission (saved 2026-06-11)

Resume point after co-author review. Everything below is committed and pushed.

## Identity
- **Manuscript:** "Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility"
- **Journal:** Nature Communications (single-blind — NO anonymisation; real author names)
- **Manuscript number:** NCOMMS-26-021731A
- **Editor:** Dr. Marcel Bigorajski · screening email from Kajal Bhadale
- **Repo:** github.com/Roverlucas/genai-reproducibility-protocol · **HEAD commit `0974890`** · **tag `v1.1`**
- **Figshare DOI:** 10.6084/m9.figshare.31653373 · private reviewer URL: figshare.com/s/3d17327cef1ae99ed37c

## Where the final files are
- **For co-authors (PDFs):** `/Users/lucasrover/Downloads/NatComms_revisao_final_autores/` (01–08 + LEIA-ME) and `.zip` alongside.
- **Submission staging (compile + upload):** `OVERLEAF_FINAL_SUBMISSION/` — `ready_pdfs/` (01–05), `02_compile_tracked/` (latexdiff), `checklist_content_for_official_forms/` (07/08 content + `OFFICIAL_FORMS_FILL_GUIDE.md` with the real number), `SUBMISSION_README.md`.
- **Canonical sources:** `article/ncomms_main.tex` (29pp), `article/supplementary_nature_mi.tex` (25pp), `response_letter/01_point_by_point_response.tex` (20pp), `response_letter/03_revised_cover_letter.tex` (3pp). All compile with 0 errors / 0 undefined refs.

## What is DONE
- **Editor's 3 problems:** (1) editor correspondence removed from point-by-point; (2) tables renumbered 1–4, consecutive, in order of appearance; (3) checklist content cleaned (forms still to be filled — see PENDING).
- **Profa. Yara's comments:** all applied (refs in citation order, table-title cleanup, etc.).
- **Zero revision/reviewer mentions** in manuscript + supplementary (validated word-by-word).
- **Multi-agent critical panel (37 agents):** 7/7 MUST-FIX (M1–M7) + 10/10 SHOULD-FIX (S1–S10, incl. S3/S6/S7 with real repo data). Full report: `CRITICAL_REVIEW_PANEL.md`.
- **AI-writing + formatting pass:** tells reduced, all overfulls resolved (one pre-existing 46pt field-list left, renders fine).
- **Manuscript number** inserted (cover letter Re-line, point-by-point header, checklists).

## PENDING — do AFTER co-author review
1. **🔴 Official XFA forms (REQUIRED by editor, manual in Adobe Acrobat):** Reporting Summary + Code & Software Submission Checklist. Download from MTS, fill with `checklist_content_for_official_forms/OFFICIAL_FORMS_FILL_GUIDE.md`, upload (a "Please wait" on upload is expected). The ML Checklist is already filled (`ML_CHECKLIST_official_FILLED.pdf`).
2. **🟠 Co-author approvals** — Profa. Yara explicitly asked to review before submission; cover letter asserts "all authors have approved."
3. **🟡 Optional polish (non-blocking):** degenerate bootstrap CIs [1.00,1.00] → Clopper-Pearson/Wilson; power analysis a-priori instead of post-hoc; one-line reconciliation of 3,904 / 4,104 / 7,004 run counts; refine 2–3 citations (Cliff 1993 + Vargha–Delaney for δ thresholds; correct Claude/Gemini model cards); confirm the 10-display-item limit (6 figs + 4 tables + Box 1/Fig 6 ≈ 11 — consider moving Methods Table 3 to Extended Data).

## How to RESUME / rebuild
- Recompile each canonical with `pdflatex` ×2 (sn-jnl.cls is local in the staging folders).
- **Bibliography is in citation order** via `/tmp/reorder_bib.py` + `/tmp/apply_bib.py` (re-run after adding any `\cite`). If `/tmp` is gone, the logic: sort `\bibitem` blocks by first-citation order; 57 refs currently.
- **Tracked manuscript:** `latexdiff --math-markup=off --append-safecmd="msinserted,response,changes" 02_compile_tracked/ncomms_main_ORIGINAL_presubmission.tex <current> > tracked.tex`, then swap the mangled bib for the clean one (`/tmp/fix_tracked_bib.py`), then pdflatex ×2.
- After editing the manuscript, **re-sync** any `\msinserted` verbatim quotes in the point-by-point that changed.

## GOTCHAS
- Repo goes PUBLIC on acceptance → do NOT commit `FEEDBACK_YARA_CONSOLIDADO.md` (contains Yara's WhatsApp + reviewer comments) or the stale `submission_revision_v1/READY_FOR_REVIEW/` PDFs.
- Official Nature checklists are XFA forms (show "Please wait" outside Acrobat) — cannot be filled programmatically; the `.md` files are content sources only.
- "gpt-4.1" in the manuscript is the companion-paper snapshot (external context), NOT one of the 9 study stacks.
- Validation rule: word-by-word, never grep-only.
