# Pacote para Upload no Overleaf

## Conteúdo

```
overleaf_upload/
├── manuscript/                          ← Upload este como projeto principal
│   ├── ncomms_main.tex                  Main manuscript
│   ├── supplementary_nature_mi.tex      Supplementary Information (S1-S12)
│   ├── references.bib                   Bibliography
│   ├── sn-jnl.cls                       Springer Nature class file
│   ├── sn-nature.bst                    Nature bibliography style
│   └── figures/                         All publication figures (PDF)
├── response_letter/                     ← Upload como projeto separado
│   ├── 01_point_by_point_response.tex   Response to reviewers (verbatim)
│   └── 03_revised_cover_letter.tex      Cover letter
└── reference_pdfs/                      ← Para comparar (não upload)
    ├── ncomms_main.pdf                  PDF compilado local
    ├── supplementary_nature_mi.pdf
    ├── 01_point_by_point_response.pdf
    └── 03_revised_cover_letter.pdf
```

## Como subir no Overleaf

### Opção A — Novo projeto Overleaf (recomendado se quiser começar limpo)

1. Crie um ZIP do diretório `manuscript/`:
   ```bash
   cd overleaf_upload
   zip -r manuscript.zip manuscript/
   ```
2. No Overleaf: **New Project → Upload Project → manuscript.zip**
3. Defina `ncomms_main.tex` como main document (Menu → Settings → Main document)
4. Compile (deve gerar 27 páginas, ~590 KB)

### Opção B — Atualizar projeto Overleaf existente

Se você já tem o projeto Overleaf com a versão pré-revisão:

1. No Overleaf, faça **upload** dos seguintes arquivos (substituem os existentes):
   - `manuscript/ncomms_main.tex` → substitui o `.tex` principal
   - `manuscript/supplementary_nature_mi.tex` → substitui o supplementary
   - `manuscript/figures/*.pdf` → substitui figuras (especialmente `fig_multiturn_comparison.pdf`)
   - `manuscript/references.bib` → substitui a bibliografia

2. Compile e verifique se o número de páginas é 27 (manuscript) + 18 (supplementary).

### Opção C — Response letter como projeto separado

1. Crie ZIP de `response_letter/`:
   ```bash
   cd overleaf_upload
   zip -r response_letter.zip response_letter/
   ```
2. No Overleaf: **New Project → Upload Project → response_letter.zip**
3. Defina `01_point_by_point_response.tex` como main document

## Verificações pós-upload

- [ ] Manuscript compila sem erros → 27 páginas
- [ ] Abstract = 150 palavras
- [ ] Section "Coding and math reasoning" presente
- [ ] Section "Applied impact in evidence synthesis" presente
- [ ] Bibitem `rover2026evidence` diz "submitted"
- [ ] Supplementary tem §S11 + §S12
- [ ] Response letter compila → 15 páginas

## Pacotes LaTeX necessários (Overleaf provê todos)

`amsmath`, `amssymb`, `xcolor`, `soul`, `hyperref`, `enumitem`, `parskip`, `geometry`, `inputenc`, `fontenc`, `booktabs`, `longtable`, `float`, `listings`, `multirow`, `array`, `caption`, `subcaption`

