# Human Validation Protocol — PM2.5 Disagreement Cases

**Manuscript:** Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility
**Submission:** Nature Communications — Major Revision (Item B1)
**Protocol version:** 1.0 — drafted 2026-05-11

---

## 1. Background and purpose

Our in-paper triangulation (Supplementary §S12) classifies divergences between LLM-generated extractions as **truly contradictory**, **semantically equivalent**, or **ambiguous** using two independent LLM judges (Claude Opus 4.7 and gpt-4o). The cover letter and response to Reviewer 3 (R3.6) commit to a complementary **human-rater validation** on a 10-case subsample with inter-rater Cohen's κ reported.

This protocol describes:
- the 10-case subsample selection,
- the criteria and rating procedure (identical to the LLM-judge protocol),
- how the resulting κ values feed back into the manuscript / supplementary / response letter.

**Goal:** report Cohen's κ for (i) human–human agreement and (ii) human-consensus vs each LLM judge, demonstrating that the LLM-judge verdicts are not an artefact of LLM-on-LLM evaluation.

---

## 2. Raters

| Rater | Role | Expertise |
|-------|------|-----------|
| **Rater A** | Profa. Yara de Souza Tadano (coauthor) | Environmental epidemiology, PM2.5 health effects |
| **Rater B** | [to be identified by Lucas] — independent epidemiologist | Environmental / respiratory epidemiology |

Both raters must be **independent** of the LLM-judge results and **blind to the model identity** of the extractions during rating. Rater A is a coauthor; rater B should be external (not on the author list) to satisfy the "two independent domain experts" claim in the cover letter.

---

## 3. Materials provided to each rater

For each of the 10 cases:

1. **Source abstract** (full PubMed text, ~250–400 words).
2. **Extraction X** and **Extraction Y** — two structured extractions of effect-estimate data produced by the same LLM on the same abstract under temperature=0 fixed-seed conditions. The X/Y labels are randomised per case (independent of which was run A vs run B in the source data), and the rater is blind to which extraction came from which LLM.
3. **Rating form** (`01_rating_form_rater_A.md` or `_B.md`) listing the three pre-registered criteria and verdict options.

The 10 selected cases are deterministically sampled (seed=42) from the 30-case pool used in the LLM-judge analysis, stratified to cover all four disagreement kinds (direction, magnitude, CI overlap, mixed). See §6 for the selection algorithm.

---

## 4. Pre-registered judgment criteria

For each case, the rater answers three yes/no questions and then issues one verdict.

### Criterion (a) — Direction
Do the two extractions report the same direction of effect?
- For ratio measures (RR, OR, HR): "positive" means >1.0; "negative" means <1.0; "null" means ~1.0 with CI overlapping 1.0.
- For risk differences: "positive" means >0; "negative" means <0.

Response options: **same / different / ambiguous**

### Criterion (b) — Magnitude
Do the effect estimates agree within ±20% relative to their average?

Computational shortcut: `|A−B| ÷ ((|A|+|B|)/2) ≤ 0.20` → "same"; otherwise "different"; if either value is missing → "ambiguous".

Response options: **same / different / ambiguous**

### Criterion (c) — CI overlap
If both 95% CIs are reported, do they share any range (overlap)?

Computational shortcut: intervals `[L1, U1]` and `[L2, U2]` overlap iff `max(L1, L2) ≤ min(U1, U2)`. If either CI is missing → "ambiguous".

Response options: **overlap / disjoint / ambiguous**

### Verdict
Based on the three criteria, choose one of:

- **truly_contradictory** — at least one of (a), (b), (c) materially fails (different direction, magnitude diverges by >20%, or CIs disjoint).
- **semantically_equivalent** — all three criteria hold: same direction, same magnitude within 20%, CIs overlap.
- **ambiguous** — data missing or judgment uncertain.

Also write a **brief rationale** (1–2 sentences) explaining the verdict.

---

## 5. Procedure

1. The rater receives the materials package (10 case files + 1 rating form).
2. The rater works through the 10 cases **in the order presented** (do not skip ahead).
3. For each case, the rater:
   - Reads the source abstract.
   - Reads Extraction X and Extraction Y.
   - Records the three criteria responses (a, b, c).
   - Issues the verdict.
   - Writes 1–2 sentence rationale.
4. The rater **does not consult the other rater** during rating (independence requirement for valid Cohen's κ).
5. Estimated time: **45–90 minutes** total (5–10 min per case).
6. The rater returns the completed form to Lucas Rover (lucasrover@alunos.utfpr.edu.br) as either:
   - Markdown file (preferred — easy to parse),
   - PDF / scanned printout (Lucas transcribes to JSON), or
   - Spreadsheet (CSV template available on request).

Both raters complete their forms independently. Lucas then runs the analysis script (§7) to compute κ.

---

## 6. Sample selection (10 of 30 cases)

The 10 cases are pre-selected by `select_10_for_human_validation.py` using:

- **Stratification by disagreement kind:** balanced across direction / magnitude / CI / mixed (target ≈2–3 each).
- **Coverage of judge-agreement and judge-disagreement:** include both cases where Claude and gpt-4o agreed and cases where they disagreed, so the human rating informs whether the LLM-judge disagreements track human ambiguity or LLM artefact.
- **Seed = 42** (deterministic, reproducible).

Run the selector once before distributing the materials:

```bash
cd /Users/lucasrover/paper-experiment
.venv/bin/python human_validation/select_10_for_human_validation.py
```

This produces:
- `human_validation/selected_10_cases.json` — the 10 cases with anonymised X/Y order
- `human_validation/02_case_packages/case_01.md` … `case_10.md` — one printable file per case

---

## 7. Computing Cohen's κ

After both raters return their completed forms:

1. Lucas transcribes the verdicts to JSON in `human_validation/rater_A_verdicts.json` and `rater_B_verdicts.json` using the schema:
   ```json
   {
     "rater": "A",
     "case_verdicts": {
       "case_01": "truly_contradictory",
       "case_02": "semantically_equivalent",
       ...
     }
   }
   ```
2. Run the analysis script:
   ```bash
   .venv/bin/python human_validation/compute_human_kappa.py
   ```
3. The script outputs `human_validation/human_kappa_results.json` with:
   - **Human–human Cohen's κ** (Rater A vs Rater B)
   - **Human consensus** (majority vote per case, or "ambiguous" if split)
   - **Human consensus vs Claude Opus 4.7** Cohen's κ
   - **Human consensus vs gpt-4o** Cohen's κ
   - **Fleiss' κ** across {Rater A, Rater B, Claude Opus, gpt-4o} on the 10 cases

---

## 8. Reporting

Once κ is computed, the following inserts will be made (Lucas):

### Manuscript §2.7 — append one sentence
> *"An independent human-rater validation on a 10-case subsample by two domain experts (one coauthor, one external) yielded human–human Cohen's κ = [VALUE] and human-consensus vs Claude Opus 4.7 κ = [VALUE], confirming that the truly-contradictory classification is not an LLM-on-LLM artefact."*

### Supplementary §S12 — new subsection
> *"§S12.4 Human-rater validation: protocol, 10-case subsample composition, per-case verdicts, and Cohen's κ vs each LLM judge."*

### Response letter R3.6 — append paragraph
> *"To address the concern that the LLM-as-judge classification may itself reflect an LLM-on-LLM bias, we conducted a complementary human-rater validation on a 10-case subsample. Two independent domain experts (Profa. Yara de Souza Tadano, environmental epidemiologist and coauthor; and [external rater name], environmental epidemiologist independent of the author list) rated the cases blind, against the same three pre-registered criteria. Human–human κ = [VALUE]; human-consensus vs Claude Opus 4.7 κ = [VALUE]; vs gpt-4o κ = [VALUE]. The human rating [confirms / refines] the LLM-judge verdict pattern."*

### Cover letter item 5 — update
> *"…a complementary human-rater validation by two independent epidemiologists yielded κ = [VALUE], confirming the LLM-judge majority finding."*

---

## 9. Quality control

- Both raters must complete the form independently. Do not discuss verdicts before both submissions.
- If a rater is uncertain on a case, the correct answer is "ambiguous" — do not guess.
- Rationale fields are useful for adjudication if Lucas needs to flag any case for re-review (e.g., a rater misread a CI).
- Disagreements between raters are expected; Cohen's κ is the formal measure of agreement net of chance.

---

## 10. Timeline

| Step | Owner | Time |
|------|-------|------|
| Identify Rater B (independent epidemiologist) | Lucas + Profa. Yara | 1–3 days |
| Run `select_10_for_human_validation.py` and prepare case packages | Lucas | 15 min |
| Distribute materials to both raters | Lucas | 10 min |
| Rater A rates (Profa. Yara) | Yara | 45–90 min |
| Rater B rates (external) | Rater B | 45–90 min |
| Transcribe verdicts to JSON | Lucas | 15 min |
| Run `compute_human_kappa.py` | Lucas | 1 min |
| Apply the four insertions (§8) and recompile PDFs | Lucas + Sage | 30 min |
| Update READY_FOR_REVIEW package + push to GitHub | Lucas + Sage | 10 min |
| **Total elapsed** (counting parallel rater work) | | **2–5 days** |

---

## 11. Files in this protocol

```
human_validation/
├── 00_PROTOCOL.md                       ← this document
├── STEP_BY_STEP_EXECUTION.md            ← quick-start guide for Lucas
├── select_10_for_human_validation.py    ← case sampler
├── compute_human_kappa.py               ← analysis script
├── selected_10_cases.json               ← (auto-generated)
├── 01_rating_form_rater_A.md            ← (auto-generated)
├── 01_rating_form_rater_B.md            ← (auto-generated)
├── 02_case_packages/                    ← (auto-generated, 10 files)
│   ├── case_01.md
│   ├── ...
│   └── case_10.md
├── rater_A_verdicts.json                ← Lucas creates after Yara returns form
├── rater_B_verdicts.json                ← Lucas creates after external returns form
└── human_kappa_results.json             ← (auto-generated by analysis script)
```

---

*Drafted by Sage (academic-chief) following the 6-agent audit recommendation that LLM-on-LLM validation should be triangulated with human raters before final submission. The protocol is consistent with the pre-registered criteria in `outputs/revision/t3_judge/` and uses the same case format as the existing LLM-judge pipeline.*
