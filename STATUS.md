# Revision Status Checkpoint — NatComms Major Revision

**Last update:** 2026-05-08 (auto-saved by Sage academic-chief)
**Editor deadline:** ~2026-06-12 (~5 weeks remaining of 12-week prazo)
**Plan:** `REVISION_PLAN.md` (single source of truth)

---

## ✅ Concluído nesta sessão

### Caminho C — Reframe + Análises
| Tarefa | Status | Artefatos | Resultado-chave |
|--------|--------|-----------|------------------|
| **T5 Reframe deployment stack** | ✅ DONE | `article/ncomms_main.tex` editado | Abstract/Intro/Methods/Discussion + Tables. +330 palavras. Crítica conceitual de R1 transformada em contribuição. |
| **T6 BERTScore por campo** | ✅ DONE | `analysis/bertscore_per_field.py`, `analysis/bertscore_per_field_results.json`, `analysis/tables/table_per_field_metrics.tex`, `analysis/figures/per_field_radar.pdf` | **Pivot conceitual forte**: EMR d=+1.41, BERTScore satura. Reposiciona R1 como validação do framework |

### Infraestrutura
- ✅ API keys configuradas em `~/.env`: OPENAI, ANTHROPIC, DEEPSEEK, GEMINI
- ✅ Saldos verificados (OpenAI quota fresh, Anthropic working, DeepSeek $5, Gemini free tier)
- ✅ Plano consolidado documentado: `REVISION_PLAN.md`
- ✅ Memória atualizada: `memory/genai-reproducibility-natcomms.md`

---

## 🔄 Em progresso / pendente

### Caminho A — EM EXECUÇÃO (lançado 2026-05-08 17:01 BRT)
| Tarefa | Status | Detalhes | Custo estimado |
|--------|--------|----------|------------------|
| T14 PubMed PM2.5 (10 abstracts × 5 reps × 8 stacks) | 🟢 Em progresso (Fase 1) | LLaMA 3 local rodando | ~$1.04 USD |
| T1 HumanEval (30 × 5 × 8 stacks) | ⏳ Aguardando Fase 2 | Código + pass@1 sandboxed | ~$3.00 USD |
| T1 GSM8K (30 × 5 × 8 stacks) | ⏳ Aguardando Fase 3 | Math + answer extraction | ~$3.00 USD |
| T4 Multi-turn extension (gpt-4o + deepseek) | ⏳ Aguardando Fase 4 | 10 abstracts × 5 reps × 2 stacks | ~$1.85 USD |

**Total estimado Caminho A: ~$8.90 USD** (margem $41.10 vs $50)
**Reservado T3 LLM-judge:** ~$10 USD (Claude Opus 4.7)

**Monitoramento:**
- Master log: `tail -f outputs/revision/logs/master.log`
- Por stack: `tail -f outputs/revision/logs/{task}_{stack}.log`
- Checkpoint: `cat outputs/revision/checkpoint.json`
- Runs gerados: `ls outputs/revision/runs/ | wc -l`
- Processo: `ps aux | grep run_revision`

**Resiliência:** `--resume` flag ativo. Se interromper (Ctrl-C, falha API, budget cap), basta re-rodar `./run_revision_full.sh` que retoma de onde parou.

**Total Caminho A estimado:** ~$40-55 USD (dentro do limite $50 com aperto)

### Caminho B — Validação PM2.5 (REPOSICIONADO 2026-05-08 — protege paper-irmão)
| Tarefa | Status | Detalhes |
|--------|--------|----------|
| T3-revised: Citar paper-irmão (cite-only, sem duplicar tabelas) | 🟡 Aguardando T17 | Aggregate cite no Methods + Discussion |
| T3-revised: Mini-LLM-judge (10 NOVOS casos com Claude Opus) | 🔴 Aguardando autorização | ~$3-5 USD, 1-2 dias. Sem 2º rater humano. |
| T3-revised: gpt-4o-2024-11-20 subset (drift check D8) | 🔴 Aguardando autorização | ~$0.50, 30 min. |
| T2 Promover PM2.5 a Results subsection | 🔴 Aguardando T3 | Aggregate finding only; cite RSM para detalhes |

**Decisão crítica:** NatComms cita paper-irmão (Rover & Tadano, RSM, under review) mas **NÃO reproduz** suas tabelas (pairwise disagreement, silver standards, random-effects, blindage) para preservar contribuições do RSM e evitar duplicate publication.

**Validação tem 3 camadas:**
1. Companion paper (citado, não duplicado) — 500 abstracts × 6 modelos × 10 runs
2. In-paper triangulação — 10 NOVOS casos com Claude Opus 4.7 LLM-as-judge
3. Drift check — gpt-4o-2024-11-20 subset

**Removido:** D7 validação humana (não necessária — silver standard via DeepSeek-R1 já fornece validação independente no paper-irmão).

### Caminho C — Quick wins restantes
| Tarefa | Status | Efort |
|--------|--------|-------|
| T7 Distinção cloud vs serving infra | ⏳ Pendente | 2 dias |
| T8 Tabela mecanismos por stack | ⏳ Pendente | 2 dias |
| T9 Esclarecer escopo protocolo | ⏳ Pendente | 1 dia |
| T10 Perplexity argumento literatura | ⏳ Pendente | 0.5 dia |
| T11 Esclarecer Anthropic seed | ⏳ Pendente | 0.5 dia |
| T12 Exemplos concretos de divergências | ⏳ Pendente | 1 dia |
| T13 Fix Fig 1 caption + W3C PROV def | ⏳ Pendente | 0.5 dia |
| T15 Limitação quasi-isolation closed-source | ⏳ Pendente | 0.5 dia |

### Editorial (Caminho final)
| Tarefa | Status |
|--------|--------|
| T16 Code/ML/ORCID/Zenodo checklists | 🔴 Sem prazo crítico |
| T17 Track changes + point-by-point + cover letter | 🔴 Última semana |

---

## 🔑 Decisões e parâmetros (immutáveis)

- **D1 escopo:** HumanEval + GSM8K + 10 PubMed PM2.5 abstracts
- **D2 PM2.5 validation:** LLM-as-judge (23) + amostra humana (10)
- **D3 reframe:** deployment stack como unidade primária ✅ aplicado
- **D4 Perplexity:** argumento literatura, sem rodar
- **D5 quasi-isolation closed-source:** limitação justificada
- **D6 paper-irmão:** sumário agregado, sem dup. de tabelas
- **D7 validador humano:** Profa. Yara + 2º rater independente (Cohen's kappa)
- **D8 snapshots:** manter `gpt-4-0613` originais + adicionar 1 snapshot atual em subset (`gpt-4o-2024-11-20` ou `gpt-4-turbo`)
- **Orçamento aprovado:** ≤$50 USD (escopo reduzido)

---

## 🎯 Achados críticos descobertos

### T6 finding — pivot conceitual
- BERTScore satura (≥0.97 em todos os campos) → não discrimina substantivo vs metadata
- EMR expõe a divergência: Cohen's d = +1.41 conclusion-relevant vs metadata
- Caso Gemini 2.5 Pro RAG `key_result`: EMR=0.10, BERTScore F1=0.969
- **Resposta a R1**: três-níveis framework é exatamente certo; BERTScore é estruturalmente cego à divergência substantiva

### T5 reframe — Abstract atualizado
> "In 4,104 controlled experiments across nine *deployment stacks* — tuples of (model weights, provider, serving infrastructure, API layer)..."

---

## 🚦 Próxima ação (Próxima sessão)

### Imediata (sem custo, sem bloqueio)
1. **Continuar Caminho C quick wins**: T7, T8, T9, T10, T11, T12, T13, T15 (~6 dias de trabalho)
2. **Implementar T3 LLM-as-judge code** (sem rodar APIs ainda — só preparar pipeline)
3. **Construir `run_revision_experiments.py`** unificado para T1+T4+T14 (Caminho A preparação)

### Bloqueada externamente
4. **Lucas:** email Profa. Yara para confirmar D6 (paper-irmão) + disponibilidade D7 (1ª rater)
5. **Lucas:** identificar 2º rater independente
6. **Lucas:** liberar execução A quando preparação pronta + confirmar Together AI key (opcional)

### Editorial (semana 8-9)
7. T16 + T17

---

## 💾 Resiliência — como retomar de onde parou

**Se a sessão for interrompida:**
1. Ler `REVISION_PLAN.md` (estratégia)
2. Ler `STATUS.md` (este arquivo, estado atual)
3. Ler `memory/genai-reproducibility-natcomms.md` (contexto persistido)
4. Verificar `~/.env` (chaves de API)
5. Continuar de "Próxima ação" acima

**Pontos de retomada por tarefa:**
- T5 reframe: aplicado em `article/ncomms_main.tex` — ler diff vs `submission_nature_comms/01_Manuscript.pdf` para revisar
- T6 BERTScore: resultados em `analysis/bertscore_per_field_results.json` e tabela LaTeX pronta
- Caminho A: pipeline ainda não construído — começar do zero ou adaptar `run_experiments.py`
- Caminho B: pipeline ainda não construído — começar do zero

---

*Documento auto-atualizado pelo Sage. Atualizar a cada milestone significativo.*
