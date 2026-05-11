# Plano Consolidado de Resposta — Major Revision Nature Communications

**Manuscript:** Same Prompt, Different Answer: Hidden Non-Determinism in LLM APIs Undermines Scientific Reproducibility
**Editor:** Dr. Marcel Bigorajski (Associate Editor, Nature Communications)
**Decisão:** Major Revision (12-mar-2026) — prazo sugerido 3 meses (até ~12-jun-2026)
**Plano consolidado em:** 2026-05-08 por Lucas Rover + Profa. Yara Tadano + Sage (academic-chief)

---

## 1. Sumário Executivo

Decisão editorial **favorável**: três reviewers (R2 é co-reviewer training de R1) convergem que o problema é importante. R1: *"strong and timely contribution"*. R3: *"primary novelty lies in the quasi-isolation probe"*. **Editor explicita** que sem expansão experimental para coding/reasoning **não devolverá ao review**.

**Estimativa de aceitação após resposta adequada:** 50–65%.
**Risco principal:** resposta apenas argumentativa sem trabalho experimental novo → desk-reject na resubmissão.

---

## 2. Decisões Metodológicas Consolidadas

| # | Decisão | Justificativa |
|---|---------|---------------|
| **D1** | Escopo experimental: HumanEval (code) + GSM8K (math) + 10 PubMed abstracts (T14 light) | Convergência editor + R1 + R3; Pareto 80/20 (Yara) |
| **D2** | Validação PM2.5: **TRIANGULAÇÃO PROTEGIDA** — cite paper-irmão (Rover & Tadano, RSM, under review) para validação completa de 500 abstracts + mini-LLM-as-judge in-paper (10 NOVOS casos, distintos dos 23 do RSM) com Claude Opus 4.7. **NÃO duplicar tabelas do RSM** (pairwise disagreement, silver standards, random-effects per run, small-literature simulation, blindage analyses) | R3 item 6; **proteger RSM** (under review, risco scoop); resposta mais forte (validação independente em 2 frentes) |
| **D3** | Reframe profundo: deployment stack como unidade primária de análise | R1 (crítica conceitual central); transforma fraqueza em contribuição |
| **D4** | Perplexity (T10): argumento literatura, sem novos experimentos | Otimização Yara; economia API + tempo |
| **D5** | Quasi-isolation closed-source: limitação justificada | Yara/Sage convergência; R3 item 2 reconhece impossibilidade técnica |
| **D6** | Paper-irmão (RSM): sumário agregado no NatComms, sem duplicar tabelas | Mitigação de duplicate publication; mantém RSM com primazia dos detalhes |
| **D7** | Validação humana: Profa. Yara + 2º rater independente (Cohen's kappa) | Padrão-ouro; defesa contra qualquer ranger novo |
| **D8** | Snapshots: manter originais + 1 snapshot atual em subset | Resposta cirúrgica a R3 item 6; ~50 runs adicionais |

---

## 3. Reframe Profundo — Deployment Stack como Unidade de Análise

### 3.1 Conceito-âncora a introduzir

> **A unidade de análise neste trabalho não é um modelo, mas uma deployment stack: tupla (model_weights, provider, infrastructure, API_layer). O mesmo conjunto de pesos servido por duas stacks distintas constitui dois objetos experimentais distintos para fins de reprodutibilidade. Esta é a contribuição mecanística central — e é exatamente o que o resultado Together AI demonstra.**

### 3.2 Mudanças textuais pervasivas

1. **Título** — manter (já endereça "LLM APIs"); reforçar no abstract.
2. **Abstract** — substituir "API-served models" → "API-served deployment stacks"; introduzir tupla.
3. **Introdução** — novo parágrafo (~80 palavras) introduzindo deployment stack como construto.
4. **Methods** — nova subseção curta **"Unit of Analysis: Deployment Stacks"** (~150 palavras).
5. **Tabelas** — coluna `Model` → `Deployment Stack` (e.g., "GPT-4 / OpenAI / production / chat completions API").
6. **Figuras** — legends atualizadas; eixo de Fig 1 explicita stack.
7. **Discussion** — parágrafo dedicado conectando Together AI → "stack, não modelo, é o objeto de variação".

### 3.3 Nova tabela proposta (Methods ou SI)

| Stack ID | Model Weights | Provider | Infrastructure | Probe Role |
|----------|--------------|----------|----------------|------------|
| L-LLaMA3 | LLaMA 3 8B Q4 | Local Ollama | Single-GPU M4 | Reference local |
| L-Mistral | Mistral 7B Q4 | Local Ollama | Single-GPU M4 | Reference local |
| L-Gemma2 | Gemma 2 9B Q4 | Local Ollama | Single-GPU M4 | Reference local |
| C-LLaMA3 | LLaMA 3 8B INT4 | Together AI | Multi-GPU cloud | **Quasi-isolation probe** |
| API-GPT4 | GPT-4 (proprietary) | OpenAI | Production cluster | Closed; full stack opaque |
| API-Claude | Claude Sonnet 4.5 | Anthropic | Production cluster | Closed; full stack opaque |
| API-Gemini | Gemini 2.5 Pro | Google | Production cluster | Closed; full stack opaque |
| API-DeepSeek | DeepSeek Chat | DeepSeek | Production cluster | Closed; full stack opaque |
| API-Perplexity | Sonar | Perplexity | Production + retrieval | Search-augmented |

---

## 4. Plano de Tarefas (mapeado de Yara T1-T17)

### Prioridade P1 — Bloqueantes

| ID | Tarefa | Caminho | Esforço |
|----|--------|---------|---------|
| T1 | Expansão coding+math (HumanEval+GSM8K) | A | ~600-800 runs novos; 3 sem |
| T2 | Promover PM2.5 a posição central | B | depende T3; 1 sem reescrita |
| T3 | Validação semântica PM2.5 (LLM-judge + 10 humano) | B | ~2 sem (incl. 2º rater) |
| T4 | Estender multi-turn/RAG para GPT-4 + DeepSeek | A | ~200-400 runs; 1-2 sem |

### Prioridade P2 — Críticas substantivas

| ID | Tarefa | Caminho | Esforço |
|----|--------|---------|---------|
| T5 | Reframe deployment stack pervasivo | C | 5 dias |
| T6 | BERTScore por campo (objective, method, key_result) | C | 3-5 dias (dados existentes) |
| T7 | Distinção cloud vs production serving infra | C | 2 dias |
| T8 | Tabela mecanismos por stack | C | 2 dias |

### Prioridade P3 — Clarificações

| ID | Tarefa | Caminho | Esforço |
|----|--------|---------|---------|
| T9 | Esclarecer escopo do protocolo (cliente vs provider) | C | 1 dia |
| T10 | Perplexity: argumento literatura (sem experimentos) | C | 0.5 dia |
| T11 | Esclarecer Anthropic seed (logged-only) | C | 0.5 dia |
| T12 | Adicionar exemplos concretos de divergências | C | 1 dia |
| T13 | Fix Fig 1 caption + W3C PROV def + links S4 | C | 0.5 dia |
| T14 | 10 PubMed abstracts em domínio diferente (light) | A | ~50 runs; 2 dias |
| T15 | Reconhecer limitação quasi-isolation closed-source | C | 0.5 dia |

### Prioridade P4 — Editorial

| ID | Tarefa | Esforço |
|----|--------|---------|
| T16 | Code/Software checklist + ML checklist + Zenodo + ORCID | 1 sem |
| T17 | Track changes + point-by-point + cover letter | 1 sem |

---

## 5. Cronograma — 9 Semanas (PERT-CPM)

```
SEMANA  │ CAMINHO A (experimentos)        │ CAMINHO B (PM2.5)            │ CAMINHO C (reescrita)
────────┼─────────────────────────────────┼──────────────────────────────┼──────────────────────────────
   1    │ Pipeline unificado (T1+T4+T14)  │ LLM-as-judge code (T3)       │ Reframe deployment (T5)
   2    │ Configurar HumanEval/GSM8K      │ Run LLM-judge nos 23 casos   │ Reescrita tabelas + legendas
   3    │ EXECUÇÃO BATCH (noturna, APIs)  │ Yara + 2º rater humano (T3)  │ T7 cloud vs infra; T8 mecanismos
   4    │ Análise pass@1, EMR, NED        │ Validação cruzada (kappa)    │ T6 BERTScore por campo
   5    │ Resultados consolidados         │ Subseção "Applied impact"    │ T15 limitação; T9 escopo protocolo
   6    │ ────────── CONVERGÊNCIA ──────────  │ Integração + T11, T12
   7    │ Quick wins finais T10, T13                                      │
   8    │ ⚠️ QUALITY GATE (6-agent + AI-detection) + T16 checklists      │
   9    │ T17 track changes + point-by-point + cover letter → SUBMISSION │
```

**Margem:** ~3 semanas vs prazo editorial de 12 semanas. Em caso de slip, pedir extensão formal (prática NatComms).

---

## 6. Dependências Externas e Restrições

### Orçamento API (aprovado pelo Lucas: ≤$50 USD — escopo reduzido)
- **OpenAI:** subset mínimo (~10 problemas HumanEval + ~10 GSM8K) com gpt-4-turbo OU gpt-4o
- **Anthropic:** Claude Sonnet 4.5 full (T1+T4+T14) + Claude Opus 4.7 para T3 LLM-judge
- **Google Gemini:** full coverage (mais barato)
- **DeepSeek:** ⏳ Lucas vai liberar API key após outros experimentos rodarem
- **Together AI:** já usado quasi-isolation (sem custo novo)
- **Perplexity:** D4 — sem rodar (argumento literatura)

### T14 PubMed abstracts (D-T14 confirmado)
- **Fonte:** corpus PubMed do paper-irmão `/Users/lucasrover/llm-evidence-synthesis-reproducibility/`
- **Tema:** PM2.5/respiratório (alinha narrativa com paper-irmão)
- **N:** 10 abstracts amostrados aleatoriamente

### Bloqueios externos (você precisa resolver)
- **D6 — paper-irmão:** confirmar com Profa. Yara overlap aceitável; **APENAS** sumário agregado + citação. **NÃO incluir** tabelas do RSM (pairwise disagreement, silver standards, random-effects, blindage) no NatComms — risco de invalidar RSM (under review).
- **D7 — REMOVIDA:** validação humana **não necessária** — RSM já tem silver-standard validation via DeepSeek-R1 (modelo independente). Preservar para RSM.
- **D7-NEW — Mini-LLM-judge protegido:** rodar Claude Opus 4.7 em **10 NOVOS casos** (distintos dos 23 efeitos do RSM) como triangulação in-paper. Custo ~$3-5 USD.

---

## 7. Riscos e Mitigações

| # | Risco | Mitigação |
|---|-------|-----------|
| R1 | Custo API alto em coding/math | Subsets pequenos (~30 itens); modelos baratos onde aplicável; batch noturno (rate limits) |
| R2 | Drift gpt-4-0613 deprecated | Manter originais + 1 snapshot atual subset (gpt-4-turbo/4o); declarar nas Limitations |
| R3 | Slip de prazo | Margem de 3 sem; pedir extensão formal se necessário |
| R4 | AI-detection na escrita | Quality gate obrigatório semana 8: sa-research-marketer + sa-ai-writing-detector |
| R5 | 2º rater indisponível | Backup: aceitar Lucas+Yara co-rating com declaração de viés |

---

## 8. Quality Gates

### Semana 6 (mid-revision)
- [ ] Resultados experimentais T1+T4+T14 consolidados
- [ ] Validação T3 completa com Cohen's kappa reportado
- [ ] Reframe T5 cobrindo abstract, intro, methods, discussion

### Semana 8 (pre-submission)
- [ ] 6-agent academic audit: gap-analyst, peer-review-defender, citation-sentinel, scientific-writer, statistician, language-editor
- [ ] AI-detection check passou
- [ ] All NatComms checklists completos (Code/Software + ML + ORCID)
- [ ] Zenodo release tagged
- [ ] Figshare data updated

### Semana 9 (submission-ready)
- [ ] Track changes versão completa
- [ ] Point-by-point response: cada crítica → response → manuscript change
- [ ] Cover letter destacando expansão coding/math + reframe + applied impact
- [ ] Coautores aprovaram (L.R., H.V.S., E.T.B., A.T.A., Y.S.T.)

---

## 9. Próximas Ações (Semana 1)

1. **Lucas:** email para Profa. Yara confirmando D6 (paper-irmão coordination)
2. **Lucas:** identificar 2º rater independente para D7 (epidemiologista)
3. **Lucas:** estimar custo API para T1 (HumanEval+GSM8K) — go/no-go orçamentário
4. **Sage (academic-chief):** delegar @experiment-runner para preparar pipeline unificado T1+T4+T14
5. **Sage:** delegar @scientific-writer para iniciar reframe T5 (caminho C paralelo)

---

*Documento vivo — atualizar status semanal. Coautores: Lucas Rover, Hugo Valadares Siqueira, Eduardo Tadeu Bacalhau, Anibal Tavares de Azevedo, Yara de Souza Tadano.*
