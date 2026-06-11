# Painel crítico multiagente — NatComms (37 agentes, 27 crítico/maior confirmados, 25 menores)
# Veredito: GO-COM-RESSALVAS — fechar os 7 MUST-FIX antes de resubmeter

## 🔴 MUST-FIX (bloqueantes)
- **M1.** Citar/engajar **He et al. 2025** ("Defeating Nondeterminism in LLM Inference" / batch-invariance, Thinking Machines) — ausente; ataca a tese mecanística. Posicionar como complementar (eles = provider-side em stacks abertos; nós = quantificação cross-provider em APIs fechadas + protocolo client-side).
- **M2.** Calibrar atribuição causal "implicates production infrastructure, **not** cloud" → "**is consistent with**" (Abstract L61, Intro L101, Discussion L447/L488). Confound escala×serving num único par L-/C-LLaMA3.
- **M3.** Contradição BERTScore: "**>0.97 across all models/conditions**" (main L210) é falsificado pela própria S13 (vários <0.97; legenda L926 diz >0.94). Qualificar: "aggregate (whole-output) >0.97; per-field >0.94".
- **M4.** **Erro de dado:** supp S3 L196 reporta "GPT-4 EMR=0.230 on RAG" — mas **GPT-4 nunca rodou RAG**. É a EMR de summarisation, mal-rotulada. Trocar por Claude RAG=0.000 / Gemini RAG=0.070.
- **M5.** **Aritmeticamente impossível:** texto diz que as 7 discordâncias inter-juiz caem todas numa célula, mas Claude só tem 5 "ambiguous" (supp L871, main L258). Suavizar p/ "5 de 7" + matriz de confusão 3×3.
- **M6.** **Cover letter L55** promete "pass@1 for code and final-answer accuracy for math" — o manuscrito reporta **só EMR**. Adicionar a linha de pass@1/accuracy (dados já existem) OU reescrever a frase.
- **M7.** "**GPT-4.1**" (main L255, L483) não é um dos 9 stacks — é o snapshot do paper companheiro, importado, sem sinalizar. Clarificar inline.

## 🟠 SHOULD-FIX (fortalece materialmente)
- **S1.** Honestidade do "**four-fold gap**": 4× vem só de GPT-4+Claude; 4-stack é ~3×. Liderar com o 3.1× [2.48–3.61] do subsample balanceado. Em code/math o δ agregado é +0.266 (small, dirigido pelo pior stack).
- **S2.** Atenuar dependência do LLM-as-judge (**κ=0.29 "fair"**, n=30, sem ground-truth humano, juízes não-repetidos). Mover peso retórico para o **per-field EMR (Cohen d=+1.41)**, que não depende de juiz. Nomear paradoxo de kappa de alta prevalência + reportar PABAK.
- **S3.** **Janela de coleta ausente** — por tese própria, resultado de API só é reproduzível relativo à época. Adicionar datas de medição por provedor (já estão nos Run Cards).
- **S4.** Tabela 1 mistura C1/C2 sem rótulo; main L123 diz "with fixed seeds GPT-4=0.443" mas esse número é C2. Corrigir.
- **S5.** Cross-ref quebrada: backmatter L713 promete "(S10) Environment/provenance" mas §S10 é outra coisa; environment_hash/versões de client-libs nunca documentados.
- **S6/S7.** IDs HumanEval/GSM8K e passagens RAG não enumerados (cross-ref "full lists in S11" quebrada); prompt verbatim do juiz e parâmetros de decoding dos juízes ausentes em S12.
- **S8.** Não contar "deployment stack" (proposto pelo R1) nem a contribuição #1 (sobrepõe atil2024) como novidade — rebaixar.
- **S9.** Companion paper (OSF, não-revisado) sustenta "23 effects" em destaque — flag inline "preregistered, not yet peer-reviewed"; deixar a §Applied impact carregar a significância nos dados deste paper.
- **S10.** FWER cobre só Tasks 1-2; justificativa em S9 é non-sequitur; claims de code/math são descritivos mas usam "widens/confirms" (inferencial). Adicionar 2ª família corrigida OU declarar descritivo no main.

## 🟡 OPCIONAL (polimento)
CIs degenerados [1.00,1.00]→Clopper-Pearson/Wilson · power post-hoc→a-priori/MDE · reconciliar 3.904/4.104/7.004 · seed do bootstrap · Box 1 "seed=42" p/ Claude · critério "eight stacks" (Perplexity também tem Tasks 1-2) · refs (Cliff 1993 + Vargha-Delaney; model cards corretos) · abstract domain sweep > dados testados · tells de IA ("not X but Y" ≥5×, "per se/itself" ~12×, duplicata "local-vs-API local--API" L797) · display items (~11 main — considerar mover Tabela 3 p/ Extended Data).

## Veredito
**GO-COM-RESSALVAS.** Nenhum MUST-FIX invalida resultados — são inconsistências internas reconstruíveis por um revisor (M3/M4/M5/M7), descalibração de claims (M2), gap de citação que ataca a tese (M1), e mismatch cover-letter↔paper (M6). O risco real não é rejeição por mérito, é **erosão de confiança/desk-scrutiny**: um paper cuja tese é "inconsistências não-documentadas minam reprodutibilidade" não pode conter um EMR de uma célula que nunca rodou. ≈1-2 dias de edição, **sem novos experimentos**.
