# Step-by-Step Execution — Human Validation

**Quick reference for Lucas.** Full protocol in `00_PROTOCOL.md`.

---

## 🎯 Big picture (5 steps over ~2-5 days)

```
1. Identify Rater B (external epidemiologist)         ← Lucas + Yara, ~1-3 days
2. Generate case packages                              ← 1 command, ~1 min
3. Distribute to Rater A (Yara) + Rater B              ← Lucas, ~10 min
4. Wait for ratings (work in parallel)                 ← Raters, ~45-90 min each
5. Transcribe + compute κ + insert in manuscript       ← Lucas + Sage, ~30-60 min
```

---

## ▶ Step 1 — Identify Rater B

**Who:** External epidemiologist (NOT on author list).

**Candidates to ask Profa. Yara about:**
- Outro pesquisador do grupo UTFPR/PPGSAU
- Colaborador da Unicamp/USP em epidemiologia ambiental
- Coautor anterior de Yara em paper de PM2.5 / saúde respiratória
- Pesquisador externo conhecido em SBE / Brazilian Society of Epidemiology

**Email template for inviting Rater B:**

```
Subject: Invitation: 60-90 minute blinded rating task for Nature Communications revision

Dear [Name],

We are preparing the resubmission of a Major Revision at Nature Communications
on hidden non-determinism in large language model APIs. One of the reviewer
concerns relates to whether divergences in LLM-extracted effect estimates
from PM2.5 / respiratory health abstracts are substantively contradictory or
merely cosmetic textual variation.

To validate this beyond LLM-on-LLM evaluation, we are running a blinded human-
rater task with two independent domain experts in environmental epidemiology.
Profa. Yara Tadano (UTFPR, coauthor) is one rater; we would be grateful if
you could serve as the second, external rater.

The task: rate 10 cases (each case = 1 abstract + 2 anonymised extractions)
against 3 pre-registered criteria (effect direction, magnitude ±20%, CI
overlap), then issue one of three verdicts per case. Estimated time: 45-90
minutes total. The materials package is self-contained and we provide a
fillable form. Cohen's κ between raters will be reported in the supplementary;
your contribution will be acknowledged in the manuscript Acknowledgements.

Could you confirm whether you are able to participate, and if so a target
deadline (we hope to resubmit within 2-3 weeks)?

Best regards,
Lucas Rover
```

---

## ▶ Step 2 — Generate case packages

```bash
cd /Users/lucasrover/paper-experiment
.venv/bin/python human_validation/select_10_for_human_validation.py
```

This produces:
- `human_validation/selected_10_cases.json` — the 10 cases with anonymised X/Y order
- `human_validation/01_rating_form_rater_A.md`
- `human_validation/01_rating_form_rater_B.md`
- `human_validation/02_case_packages/case_01.md` … `case_10.md`

**Sanity check:** each `case_XX.md` should contain the abstract + Extraction X + Extraction Y, with NO model identifier or run number visible to the rater.

---

## ▶ Step 3 — Distribute to raters

**Rater A — Profa. Yara:**
- Email Yara with subject: "Human-rater task — 10 cases for NatComms revision (45-90 min)"
- Attach: `01_rating_form_rater_A.md` + 10 case files from `02_case_packages/`
- Ask her to fill the form and reply within X days
- Optionally provide as a single ZIP

**Rater B — external:**
- Same email + materials, with form `_rater_B.md`

**Important:** Do NOT share the LLM-judge verdicts with either rater. Do NOT discuss specific cases with the raters until both have submitted.

Pre-package the materials as ZIP for convenience:

```bash
cd /Users/lucasrover/paper-experiment
zip -r human_validation/rater_A_package.zip \
  human_validation/00_PROTOCOL.md \
  human_validation/01_rating_form_rater_A.md \
  human_validation/02_case_packages/

zip -r human_validation/rater_B_package.zip \
  human_validation/00_PROTOCOL.md \
  human_validation/01_rating_form_rater_B.md \
  human_validation/02_case_packages/
```

---

## ▶ Step 4 — Wait for completed forms

Each rater returns their completed `01_rating_form_rater_X.md` (or PDF) to Lucas.

**If they prefer paper or PDF:** transcribe their verdicts manually into JSON:

```bash
# Edit by hand
nano human_validation/rater_A_verdicts.json
```

Schema:

```json
{
  "rater": "A",
  "rater_name": "Yara de Souza Tadano",
  "rater_affiliation": "UTFPR — Programa de Pós-Graduação em Sustentabilidade Ambiental Urbana",
  "rating_date": "2026-05-XX",
  "case_verdicts": {
    "case_01": "truly_contradictory",
    "case_02": "semantically_equivalent",
    "case_03": "ambiguous",
    "case_04": "truly_contradictory",
    "case_05": "truly_contradictory",
    "case_06": "semantically_equivalent",
    "case_07": "truly_contradictory",
    "case_08": "ambiguous",
    "case_09": "truly_contradictory",
    "case_10": "truly_contradictory"
  },
  "rationales": {
    "case_01": "Direction differs (one positive, one null with overlapping CI).",
    "case_02": "...",
    "case_03": "..."
  }
}
```

(Same schema for `rater_B_verdicts.json`.)

---

## ▶ Step 5 — Compute κ + insert results

```bash
cd /Users/lucasrover/paper-experiment
.venv/bin/python human_validation/compute_human_kappa.py
```

This prints, e.g.:

```
==================================================
Human-rater validation results (n=10 cases)
==================================================
Rater A (Yara) verdicts:    7 contradictory, 2 equivalent, 1 ambiguous
Rater B (external) verdicts: 8 contradictory, 1 equivalent, 1 ambiguous

Cohen's kappa:
  Human-Human (A vs B):              0.78  (substantial)
  Human consensus vs Claude Opus 4.7: 0.65  (substantial)
  Human consensus vs gpt-4o:          0.71  (substantial)

Fleiss' kappa across 4 raters:       0.68  (substantial)

Agreement rate:
  Human-Human (A vs B):              90%
  Human consensus vs Claude Opus:    80%
  Human consensus vs gpt-4o:         85%
```

Then update the three documents (Sage can do this automatically once you give the go-ahead):
- `article/ncomms_main.tex` §2.7 — append 1 sentence with κ values
- `article/supplementary_nature_mi.tex` §S12 — add new §S12.4 subsection
- `response_letter/01_point_by_point_response.tex` R3.6 — append validation paragraph
- `response_letter/03_revised_cover_letter.tex` item 5 — update "will accompany" → "yielded κ=X"

Recompile PDFs, update READY_FOR_REVIEW package, commit + push to GitHub.

---

## ⚠️ Quality gates before submission

- [ ] Both raters submitted independently (no consultation during rating).
- [ ] Cohen's κ (human-human) ≥ 0.40 — anything lower suggests the criteria need refinement (not expected with pre-registered criteria, but check).
- [ ] κ (human-consensus vs Claude Opus) ≥ 0.40 — confirms LLM judge not arbitrary.
- [ ] κ (human-consensus vs gpt-4o) ≥ 0.40 — confirms LLM judge not arbitrary.
- [ ] If any κ < 0.40, schedule a 30-min adjudication call with both raters to discuss the disagreements before reporting.

---

## 📋 Materials checklist

Before you send the raters anything, confirm:

- [ ] `human_validation/00_PROTOCOL.md` is in the ZIP
- [ ] The right rating form (`_A.md` for Yara, `_B.md` for external)
- [ ] All 10 case files in `02_case_packages/`
- [ ] No LLM-judge verdicts leaked anywhere in the materials
- [ ] No model identifiers (Claude/GPT-4.1) visible in the case files (the X/Y labels are blind)

---

## 🆘 Troubleshooting

**Rater B declined or unavailable?**
- Ask Profa. Yara for 2–3 backup names
- If truly impossible to find a second external rater, fall back to Yara + Lucas as co-raters with explicit acknowledgement of the non-independence (still gives a κ, just weaker defense)

**A rater wants to skip a case (no opinion)?**
- That's "ambiguous" — record as such

**Raters disagree on > 4 of 10 cases?**
- Adjudication call: walk through each disagreement, identify whether it's a criterion-interpretation issue (resolvable) or a genuine epistemic disagreement (report as is)

**Want to add a 3rd human rater (e.g., a co-author with clinical expertise)?**
- Sure — add `rater_C_verdicts.json`, the analysis script handles N≥2 raters automatically (it computes Fleiss' κ across all human raters + Cohen's κ pairwise)
