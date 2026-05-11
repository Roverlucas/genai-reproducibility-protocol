# Response Letter Package — NatComms Major Revision

**Manuscript:** Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility
**Status:** In preparation (started 2026-05-08)

---

## Documents in this package

| File | Purpose | Status |
|------|---------|--------|
| `01_point_by_point_response.tex` | Verbatim reviewer comments + responses + manuscript changes for each point | 🟡 T5+T6 populated; T7-T17 placeholders ready to fill |
| `02_changes_log.md` | Granular change log: every edit by section/line, mapped to reviewer point | 🟡 T5+T6 populated; T7-T17 status tracked |
| `03_revised_cover_letter.tex` | Cover letter to the editor summarising the revision | 🟢 Drafted (placeholders for T1-T4 results) |

---

## How to compile

```bash
cd /Users/lucasrover/paper-experiment/response_letter

# Render point-by-point response (uses standard LaTeX packages)
pdflatex 01_point_by_point_response.tex
pdflatex 01_point_by_point_response.tex  # second pass for refs

# Render cover letter
pdflatex 03_revised_cover_letter.tex
pdflatex 03_revised_cover_letter.tex
```

Both should compile cleanly with TinyTeX (already installed at `~/Library/TinyTeX/bin/universal-darwin/`).

---

## Submission package layout (final)

When all revisions complete, the resubmission package will be:

```
submission_revision_v1/
├── ncomms_main_clean.tex              # Final clean revised manuscript
├── ncomms_main_clean.pdf              # Compiled
├── ncomms_main_tracked.tex            # latexdiff of post-T5 vs final
├── ncomms_main_tracked.pdf            # Track-changes version (Editor sees this)
├── supplementary_revised.tex          # Revised supplementary
├── supplementary_revised.pdf
├── 01_point_by_point_response.pdf     # Compiled response letter
├── 03_revised_cover_letter.pdf        # Cover letter
├── ML_checklist_v2.pdf                # Updated ML checklist
├── code_software_checklist.pdf        # New (T16)
├── reporting_summary_v2.pdf           # Updated reporting summary
└── figshare_data_DOI.txt              # Updated figshare DOI reference
```

---

## How responses are organised in `01_point_by_point_response.tex`

Each reviewer point has three blocks:

1. **Reviewer quote** (verbatim, in italic grey, in `revquote` environment)
2. **Authors' response** (green label, plain text)
3. **Changes in manuscript** (blue label, with explicit line/section references)

Plus a **status marker**: `\done`, `\inprogress`, or `\pending` for tracking.

When the revision is complete, all `\inprogress` and `\pending` markers should be replaced with `\done` and the corresponding response/changes blocks should be filled with concrete content.

---

## Coautores: how to review

1. Open `01_point_by_point_response.pdf` (or .tex)
2. Reviewer comments are in italic grey
3. Our responses are in green
4. Changes references are in blue
5. Status markers tell you what's done vs in-progress
6. Cross-reference with `02_changes_log.md` for granular line-by-line edits
7. The actual revised manuscript is at `../article/ncomms_main.tex` (clean) or `../submission_revision_v1/ncomms_main_tracked.tex` (track changes, generated last)

---

## Update protocol

When a task (T1-T17) completes:

1. Update `02_changes_log.md` row from ⏳/🔄 to ✅
2. Replace the `\pending` or `\inprogress` macro in `01_point_by_point_response.tex` with `\done`
3. Fill in the concrete numbers/text in the corresponding response and changes blocks
4. Recompile both PDFs

When **all** tasks complete:

1. Update cover letter (`03_revised_cover_letter.tex`) with final result numbers
2. Run `latexdiff submission_revision_v1/ncomms_main_post_T5.tex article/ncomms_main.tex > submission_revision_v1/ncomms_main_tracked.tex`
3. Compile track-changes version
4. Final coauthor sign-off
5. Submit via MTS

---

*Maintained by Sage academic-chief.*
