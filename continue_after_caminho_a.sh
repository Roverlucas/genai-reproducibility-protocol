#!/bin/bash
# Autonomous continuation script — fires when Caminho A finishes.
# Authorised by Lucas Rover 2026-05-08: "tem autorização para seguir após terminar caminho A".
#
# Phases:
#   1. Wait for run_revision_full.sh + run_revision_experiments.py to exit
#   2. Run T3 Phase B (Claude Opus LLM-judge, ~$0.24)
#   3. Run analyze_revision_results.py (EMR/NED/pass@1/GSM8K accuracy, no API)
#   4. Generate latexdiff (track changes vs post-T5 snapshot)
#   5. Compile all PDFs (manuscript clean + tracked, response, cover, supplementary)
#   6. Build "READY_FOR_PROFESSOR_REVIEW" package
#
# All phases are idempotent: re-running the script is safe.
# Output: outputs/revision/logs/continuation.log (fully timestamped).

set +e  # Don't abort on any single phase error — log and continue
cd /Users/lucasrover/paper-experiment

LOG=outputs/revision/logs/continuation.log
mkdir -p outputs/revision/logs

# Redirect all stdout+stderr to the log
exec >> "$LOG" 2>&1

set -a
source /Users/lucasrover/.env 2>/dev/null
set +a

PYTHON=.venv/bin/python
PDFLATEX=$HOME/Library/TinyTeX/bin/universal-darwin/pdflatex
# latexdiff installed via homebrew (TinyTeX 2025 is older than TeX Live 2026 repo)
LATEXDIFF=/opt/homebrew/bin/latexdiff

echo ""
echo "================================================================"
echo "  CONTINUATION SCRIPT STARTED: $(date)"
echo "================================================================"

# ---------- PHASE 1 — Wait for Caminho A ----------
echo ""
echo "[PHASE 1] Waiting for Caminho A to finish..."
WAIT_INTERVAL=120  # 2 minutes
ELAPSED=0
MAX_WAIT=18000  # 5 hours hard cap
while pgrep -f "run_revision_full.sh" > /dev/null || pgrep -f "run_revision_experiments.py" > /dev/null; do
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "[PHASE 1] HARD-CAP reached ($MAX_WAIT s). Aborting wait."
    break
  fi
  RUNS=$(ls outputs/revision/runs/ 2>/dev/null | wc -l | tr -d ' ')
  echo "  [$(date +%H:%M:%S)] Caminho A still running. Runs so far: $RUNS"
  sleep $WAIT_INTERVAL
  ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done
echo "[PHASE 1] Caminho A finished at $(date)"
RUNS=$(ls outputs/revision/runs/ 2>/dev/null | wc -l | tr -d ' ')
echo "[PHASE 1] Total runs generated: $RUNS"

# ---------- PHASE 2 — T3 Phase B (Claude Opus LLM-judge) ----------
echo ""
echo "[PHASE 2] T3 Phase B: Claude Opus LLM-as-judge starting at $(date)"
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "[PHASE 2] WARN: ANTHROPIC_API_KEY not set. Skipping."
else
  $PYTHON run_t3_validation.py --judge --execute 2>&1 | tee outputs/revision/logs/t3_judge.log
fi
echo "[PHASE 2] T3 finished at $(date)"

# ---------- PHASE 3 — Analyse Caminho A results ----------
echo ""
echo "[PHASE 3] Analysing Caminho A results at $(date)"
$PYTHON analyze_revision_results.py 2>&1 | tee outputs/revision/logs/analysis.log
echo "[PHASE 3] Analysis finished at $(date)"

# ---------- PHASE 4 — latexdiff ----------
echo ""
echo "[PHASE 4] Generating latexdiff at $(date)"
mkdir -p submission_revision_v1
if [ -f "submission_revision_v1/ncomms_main_post_T5.tex" ] && [ -f "article/ncomms_main.tex" ]; then
  if [ -x "$LATEXDIFF" ]; then
    $LATEXDIFF \
      submission_revision_v1/ncomms_main_post_T5.tex \
      article/ncomms_main.tex \
      > submission_revision_v1/ncomms_main_tracked.tex 2>> outputs/revision/logs/latexdiff.log
    if [ -s submission_revision_v1/ncomms_main_tracked.tex ]; then
      echo "[PHASE 4] latexdiff OK -> submission_revision_v1/ncomms_main_tracked.tex"
    else
      echo "[PHASE 4] latexdiff produced empty file — check log"
    fi
  else
    echo "[PHASE 4] latexdiff binary not found at $LATEXDIFF"
  fi
else
  echo "[PHASE 4] Snapshot or current .tex missing"
fi

# ---------- PHASE 5 — Compile all PDFs ----------
echo ""
echo "[PHASE 5] Compiling final PDFs at $(date)"

compile() {
  local file=$1
  local dir=$(dirname "$file")
  local base=$(basename "$file" .tex)
  ( cd "$dir" && $PDFLATEX -interaction=nonstopmode -halt-on-error "$base.tex" > /tmp/_pdf_${base}.log 2>&1 )
  if [ -f "$dir/$base.pdf" ]; then
    echo "  ✓ $dir/$base.pdf"
  else
    echo "  ✗ $dir/$base.tex — see /tmp/_pdf_${base}.log"
  fi
}

compile article/ncomms_main.tex
compile article/supplementary_nature_mi.tex
compile response_letter/01_point_by_point_response.tex
compile response_letter/03_revised_cover_letter.tex
if [ -f submission_revision_v1/ncomms_main_tracked.tex ]; then
  compile submission_revision_v1/ncomms_main_tracked.tex
fi

# ---------- PHASE 6 — Build review package ----------
echo ""
echo "[PHASE 6] Building professor-review package at $(date)"
mkdir -p submission_revision_v1/READY_FOR_REVIEW
cp -f article/ncomms_main.pdf submission_revision_v1/READY_FOR_REVIEW/01_revised_manuscript_clean.pdf 2>/dev/null
cp -f submission_revision_v1/ncomms_main_tracked.pdf submission_revision_v1/READY_FOR_REVIEW/02_revised_manuscript_tracked.pdf 2>/dev/null
cp -f article/supplementary_nature_mi.pdf submission_revision_v1/READY_FOR_REVIEW/03_supplementary.pdf 2>/dev/null
cp -f response_letter/01_point_by_point_response.pdf submission_revision_v1/READY_FOR_REVIEW/04_point_by_point_response.pdf 2>/dev/null
cp -f response_letter/03_revised_cover_letter.pdf submission_revision_v1/READY_FOR_REVIEW/05_cover_letter.pdf 2>/dev/null
cp -f response_letter/02_changes_log.md submission_revision_v1/READY_FOR_REVIEW/06_changes_log.md 2>/dev/null
cp -rf article/ML_CHECKLIST_FILLED.md submission_revision_v1/READY_FOR_REVIEW/07_ml_checklist.md 2>/dev/null
cp -rf article/REPORTING_SUMMARY_FILLED.md submission_revision_v1/READY_FOR_REVIEW/08_reporting_summary.md 2>/dev/null
[ -f article/CODE_SOFTWARE_CHECKLIST.md ] && cp -f article/CODE_SOFTWARE_CHECKLIST.md submission_revision_v1/READY_FOR_REVIEW/09_code_software_checklist.md

# Generate review summary
cat > submission_revision_v1/READY_FOR_REVIEW/00_README.md <<'README_EOF'
# Pacote de Revisão para Coautores — NatComms Major Revision

**Data de geração:** AUTO_TIMESTAMP
**Manuscrito:** Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility
**Status:** Pronto para revisão dos coautores antes da resubmissão

## Documentos no pacote

| # | Documento | O que revisar |
|---|-----------|---------------|
| 01 | revised_manuscript_clean.pdf | Manuscrito revisado limpo (sem track changes) |
| 02 | revised_manuscript_tracked.pdf | Manuscrito com track changes via latexdiff |
| 03 | supplementary.pdf | Supplementary information atualizada |
| 04 | point_by_point_response.pdf | Resposta verbatim a cada comentário dos reviewers |
| 05 | cover_letter.pdf | Carta ao editor com sumário das mudanças |
| 06 | changes_log.md | Log granular de cada edição |
| 07 | ml_checklist.md | Machine Learning checklist |
| 08 | reporting_summary.md | Reporting Summary |
| 09 | code_software_checklist.md | Code/Software submission checklist |

## Como revisar

1. Ler **05_cover_letter.pdf** para overview das mudanças
2. Ler **04_point_by_point_response.pdf** para entender a lógica de cada resposta
3. Ler **02_revised_manuscript_tracked.pdf** para ver edições destacadas
4. Validar **03_supplementary.pdf** — especialmente §S4 (API docs hyperlinks) e §S6 (per-field metrics)
5. Verificar checklists 07-09

## Coautores e foco sugerido

- **Lucas Rover (LR):** todos os documentos
- **Hugo Valadares Siqueira (HVS):** orientação geral, validação metodológica
- **Eduardo Tadeu Bacalhau (ETB):** análises estatísticas (T6 BERTScore por campo, bootstrap CIs)
- **Anibal Tavares de Azevedo (ATA):** design experimental (novos T1 + T4 + T14)
- **Yara de Souza Tadano (YST):** seção PM2.5 + cite paper-irmão (D6) + supervisão geral

## Sign-off

Cada coautor confirmar via email/WhatsApp:
- [ ] Lucas Rover
- [ ] Hugo Valadares Siqueira
- [ ] Eduardo Tadeu Bacalhau
- [ ] Anibal Tavares de Azevedo
- [ ] Yara de Souza Tadano

Após sign-off → submeter via MTS (link no `05_cover_letter.pdf`).
README_EOF

# Substitute timestamp
sed -i.bak "s/AUTO_TIMESTAMP/$(date '+%Y-%m-%d %H:%M:%S')/" submission_revision_v1/READY_FOR_REVIEW/00_README.md
rm -f submission_revision_v1/READY_FOR_REVIEW/00_README.md.bak

# ---------- PHASE 7 — Final summary ----------
echo ""
echo "================================================================"
echo "  CONTINUATION COMPLETE: $(date)"
echo "================================================================"
echo ""
echo "Artefatos gerados:"
ls -la submission_revision_v1/READY_FOR_REVIEW/ 2>/dev/null
echo ""
echo "Total de runs Caminho A:"
ls outputs/revision/runs/ 2>/dev/null | wc -l
echo ""
echo "T3 verdicts:"
[ -f outputs/revision/t3_judge/t3_judge_results.json ] && \
  $PYTHON -c "import json; d=json.load(open('outputs/revision/t3_judge/t3_judge_results.json')); print(d.get('verdict_counts'))" 2>/dev/null
echo ""
echo "Custo total (estimado):"
[ -f outputs/revision/checkpoint.json ] && \
  $PYTHON -c "import json; d=json.load(open('outputs/revision/checkpoint.json')); print(f'Caminho A: \${d.get(\"total_spent_usd\", 0):.4f}')"
[ -f outputs/revision/t3_judge/t3_judge_results.json ] && \
  $PYTHON -c "import json; d=json.load(open('outputs/revision/t3_judge/t3_judge_results.json')); print(f'T3 judge: \${d.get(\"total_cost_usd\", 0):.4f}')"
echo ""
echo "Próximos passos manuais (Lucas):"
echo "  1. Revisar PDFs em submission_revision_v1/READY_FOR_REVIEW/"
echo "  2. Enviar pacote para coautores (HVS, ETB, ATA, YST)"
echo "  3. Após sign-off → submeter via MTS"
echo ""
echo "================================================================"
