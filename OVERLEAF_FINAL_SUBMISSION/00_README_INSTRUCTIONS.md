# Pacote de Submissão Revisão 1 — Nature Communications

**Manuscript:** "Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility"
**Gerado em:** 2026-05-25
**Versão:** Após correção dos 3 issues do editor (21-Mai) + fix counter appendix (25-Mai)

---

## Mapeamento dos 9 entregáveis

| # | Arquivo final | Como obter | Status |
|---|---------------|-----------|--------|
| 01 | `01_revised_manuscript_clean.pdf` | Compilar `01_compile_manuscript/` no Overleaf | RECOMPILAR |
| 02 | `02_revised_manuscript_tracked.pdf` | Rodar latexdiff em `02_compile_tracked/`, depois compilar | RECOMPILAR |
| 03 | `03_supplementary.pdf` | Compilar `03_compile_supplementary/` no Overleaf | RECOMPILAR |
| 04 | `04_point_by_point_response.pdf` | Compilar `04_compile_response_letter/` no Overleaf | RECOMPILAR |
| 05 | `05_cover_letter.pdf` | Compilar `05_compile_cover_letter/` no Overleaf | RECOMPILAR |
| 06 | `06_changes_log.md` | Arquivo já pronto | OK |
| 07 | `07_ml_checklist.pdf` | `ready_pdfs/07_ml_checklist.pdf` | PRONTO |
| 08 | `08_reporting_summary.pdf` | `ready_pdfs/08_reporting_summary.pdf` | PRONTO |
| 09 | `09_code_software_checklist.pdf` | `ready_pdfs/09_code_software_checklist.pdf` | PRONTO |

---

## Passo a passo no Overleaf

### Para cada uma das 5 pastas de compilação (01 a 05):

1. Faça ZIP da pasta (já estão prontas em `*.zip` no diretório raiz)
2. No Overleaf: **New Project → Upload Project → selecionar o ZIP**
3. **Settings → Main document**:
   - `01_compile_manuscript`: `ncomms_main.tex`
   - `02_compile_tracked`: `ncomms_main_tracked.tex` (gerado por latexdiff, ver seção abaixo)
   - `03_compile_supplementary`: `supplementary_nature_mi.tex`
   - `04_compile_response_letter`: `01_point_by_point_response.tex`
   - `05_compile_cover_letter`: `03_revised_cover_letter.tex`
4. Compile (**Recompile** ou Ctrl/Cmd+Enter)
5. Baixe o PDF (**Download PDF** no canto superior direito)
6. Renomeie conforme tabela acima

---

## Como gerar o tracked manuscript (item 02)

A pasta `02_compile_tracked/` contém 3 arquivos `.tex`:

- `ncomms_main_ORIGINAL_presubmission.tex` — versão submetida originalmente (10-Mai)
- `ncomms_main_NEW_current.tex` — versão revisada atual (25-Mai)
- `ncomms_main_tracked_OLD_12mai.tex` — tracked anterior (12-Mai, **desatualizado**)

**Opção A — latexdiff no Overleaf (Premium):**

Adicione ao seu projeto Overleaf um `latexdiff.sh` ou use o comando integrado se você tiver plano Pro.

**Opção B — latexdiff online (recomendada para conta Free):**

1. Acesse https://3142.nl/latex-diff/ (ou similar)
2. Upload:
   - **Old:** `ncomms_main_ORIGINAL_presubmission.tex`
   - **New:** `ncomms_main_NEW_current.tex`
3. Baixe o `diff.tex` gerado
4. Renomeie para `ncomms_main_tracked.tex`
5. Faça upload junto com as classes/bib/figures no Overleaf
6. Compile

**Opção C — latexdiff local (se você instalar TeX):**

```bash
brew install --cask mactex-no-gui   # instala latexdiff
cd 02_compile_tracked/
latexdiff ncomms_main_ORIGINAL_presubmission.tex ncomms_main_NEW_current.tex > ncomms_main_tracked.tex
```

---

## Mudanças aplicadas nesta sessão (25-Mai)

### Issue 1 (editor) — Cover note removida do response letter
- **Status:** Já estava aplicado antes desta sessão.

### Issue 2 (editor) — Numeração de tabelas corrigida
- `\setcounter{table}{7}` já tinha sido removido.
- **Adicionado** `\renewcommand{\thetable}{\arabic{table}}` + `\setcounter{table}{3}` antes do `\begin{table*}` em `ncomms_main.tex` linha 786 — força a tab:revision_emr render como **"Table 4"** em vez de "Table A1" (que era o resultado natural dentro de `\begin{appendices}` com sn-jnl.cls).
- **Corrigido** `01_point_by_point_response.tex`:
  - Linha 100: "Table~1" → "Table~2" (referência ao tab:main)
  - Linhas 399, 401: "Methods Table~2" → "Methods Table~3" (referência ao tab:mechanisms)
- **Corrigido** `supplementary_nature_mi.tex`:
  - Linha 196: "main text, Table~5" → "main text, Table~2"

### Issue 3 (editor) — Checklists
- PDFs 07, 08, 09 já refletem o conteúdo correto (verificado via pdftotext).

---

## Numeração final das tabelas no manuscript

| # | Label | Localização | Caption (short) |
|---|-------|-------------|-----------------|
| **Table 1** | `tab:mech_summary` | §Results, linha 166 | At-a-glance mechanism activation by deployment-stack class |
| **Table 2** | `tab:main` | §Results, linha 306 | Exact match rate per deployment stack under greedy decoding |
| **Table 3** | `tab:mechanisms` | §Methods §§Sources of non-determinism | Mapping of non-determinism mechanisms to deployment stacks |
| **Table 4** | `tab:revision_emr` | §Extended Data | EMR with 95% bootstrap CI per deployment stack on revision tasks |

(Sequência limpa **1–2–3–4**, sem o salto para Table 8 que motivou o feedback do editor.)

---

## Pacotes ZIP prontos

Após criar os ZIPs (próximo passo), você terá:

```
OVERLEAF_FINAL_SUBMISSION/
├── 00_README_INSTRUCTIONS.md         (este arquivo)
├── 01_compile_manuscript.zip         → upload ao Overleaf → 01_revised_manuscript_clean.pdf
├── 02_compile_tracked.zip            → latexdiff + upload → 02_revised_manuscript_tracked.pdf
├── 03_compile_supplementary.zip      → upload ao Overleaf → 03_supplementary.pdf
├── 04_compile_response_letter.zip    → upload ao Overleaf → 04_point_by_point_response.pdf
├── 05_compile_cover_letter.zip       → upload ao Overleaf → 05_cover_letter.pdf
├── 06_changes_log.md                 → submeter direto
└── ready_pdfs/
    ├── 07_ml_checklist.pdf           → submeter direto
    ├── 08_reporting_summary.pdf      → submeter direto
    └── 09_code_software_checklist.pdf → submeter direto
```

---

## Checklist final pré-submissão

- [ ] Compilei 01 no Overleaf — confere numeração das Tables 1-4
- [ ] Gerei 02 via latexdiff — todas as mudanças destacadas
- [ ] Compilei 03 supplementary — referências cruzadas OK
- [ ] Compilei 04 response letter — começa com "Navigation index", sem cover note
- [ ] Compilei 05 cover letter — contém a nota ao editor (movida do response)
- [ ] Upload de todos os 9 arquivos no sistema editorial (incluindo 07, 08, 09)
- [ ] Confirmei email de "submission received" da Nature Communications
