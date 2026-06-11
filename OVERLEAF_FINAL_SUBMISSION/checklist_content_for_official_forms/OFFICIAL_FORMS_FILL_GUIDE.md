# Fill guide for the official Nature XFA forms

> Single-blind submission to **Nature Communications** → use the real corresponding-author name (NOT "DBPR"/"DAPR").
> Manuscript number: NCOMMS-26-021731A.
> Open each official form in **Adobe Acrobat Reader** (XFA forms show "Please wait" elsewhere), type the content below, save, upload.

---

## A. Code and Software Submission Checklist (official XFA form)

| Field | What to type |
|-------|--------------|
| Corresponding author(s) | Lucas Rover |
| Required content — single zip OR access link | Provide link: GitHub `https://github.com/Roverlucas/genai-reproducibility-protocol` (tag `v1.1`); permanent archive Figshare DOI `10.6084/m9.figshare.31653373` (CC-BY 4.0); private reviewer access `https://figshare.com/s/3d17327cef1ae99ed37c` |
| ☑ Compiled software / source code | Python source in the GitHub repo |
| ☑ Small dataset to demo | `data/inputs/abstracts.json` (30 abstracts) + `data/inputs/revision/` (HumanEval, GSM8K, PubMed) |
| ☑ README — 1. System requirements | Python 3.14.3 (min 3.10); deps in `requirements.txt`; Ollama v0.15.5; tested on macOS 14.6 / Apple M4 (24 GB); no non-standard hardware required |
| README — 2. Installation guide | `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`; typical install < 5 min |
| README — 3. Demo | run `run_experiments.py`; expected output: per-run JSON Run Cards + EMR metrics; demo run time minutes |
| README — 4. Instructions for use | run the `run_*.py` orchestrators on your own abstracts; reproduction steps in §7 of the repo README |
| License | MIT (OSI-approved) |
| Open-source repo link | https://github.com/Roverlucas/genai-reproducibility-protocol |
| Where is the code's functionality described (pseudocode) | ☑ Methods section (§Protocol design) |

## B. Reporting Summary (official Nature Portfolio XFA form)

Transcribe from `REPORTING_SUMMARY_FILLED.md` (already cleaned). Key entries:
- §1 Statistics: items 1–10 — Confirmed/n-a as in the .md (sample size = 7,004 runs; tests two-sided + Holm–Bonferroni; effect sizes Cliff's δ 0.784–0.896, Cohen's d > 1.6).
- §2 Software and code — Data collection / Data analysis blocks from the .md.
- §3 Data availability — the statement from the .md (Figshare DOI 10.6084/m9.figshare.31653373; GitHub tag v1.1).
- §4 Human participants — all N/A.
- §5 Field-specific (Life sciences design): sample size / exclusions / replication / randomization / blinding from the .md.
- §6 Materials & methods — all n/a.

## C. Machine Learning Checklist V1.1 (only if the journal requests it — not in the editor's 2-form list)

Transcribe from `ML_CHECKLIST_FILLED.md`:
- §1 Code/Data availability — tick repo + reviewer access + pretrained models (Ollama Hub).
- §2 Datasets A–E — A/B/C/D Yes, E No (single protocol).
- §3 Model & training A–G — "We evaluate existing LLMs, not a new model"; B/C/D… mostly No/N-A; we provide Run Cards instead of Model Cards.
- §4 Evaluation A–G — A Yes (EMR/NED/ROUGE-L/BERTScore + Pass@1/accuracy), F Yes (ablations), rest N/A.
- §5 Compute A/B — Yes (Apple M4; < 1% overhead).
