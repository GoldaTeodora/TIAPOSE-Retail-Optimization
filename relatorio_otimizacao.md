# 4. Tarefa de Otimização

## 4.1 Formulação do Problema

O problema de otimização consiste em determinar, para cada loja e para cada dia da semana seguinte (horizonte de 7 dias), o número de trabalhadores Juniores ($J_{s,d}$), de Experts ($X_{s,d}$) e a percentagem de promoção ($PR_{s,d}$) a aplicar, de forma a maximizar o lucro semanal líquido.

Foram considerados três objetivos distintos, com crescente complexidade:

- **O1 – Maximização Local:** Maximizar o lucro semanal líquido de cada loja individualmente, sem restrições globais. Cada loja é otimizada de forma independente.
- **O2 – Maximização Global com Restrição:** Maximizar o lucro combinado das quatro lojas, sujeito à restrição de que o número total de unidades vendidas em todas as lojas e todos os dias não ultrapasse as **10 000 unidades**.
- **O3 – Multiobjetivo:** Maximizar o lucro global (O2) e, simultaneamente, minimizar o total de recursos humanos mobilizados. Este objetivo foi implementado de duas formas: O3_WEIGHTED (função escalarizante com pesos) e O3_NS (algoritmo NSGA-II, produzindo a frente de Pareto).

---

## 4.2 Representação da Solução

Cada solução $s$ é representada como um vetor numérico de dimensão $4 \times 7 \times 3 = 84$ parâmetros:

$$s = [J_{1,1}, \ldots, J_{4,7},\ X_{1,1}, \ldots, X_{4,7},\ PR_{1,1}, \ldots, PR_{4,7}]$$

Onde:
- $J_{s,d} \in \mathbb{Z}^+$ — número de Juniores na loja $s$ no dia $d$;
- $X_{s,d} \in \mathbb{Z}^+$ — número de Experts na loja $s$ no dia $d$;
- $PR_{s,d} \in \{0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30\}$ — percentagem de promoção.

Como os métodos de otimização real devolvem valores contínuos, os valores de $J$ e $X$ são arredondados dentro da função de avaliação. As previsões do número de clientes para os 7 dias seguintes, obtidas pelo melhor modelo de previsão, são usadas como entrada para definir os limites superiores de cada parâmetro.

---

## 4.3 Função de Avaliação

A função de avaliação calcula o lucro diário de cada loja segundo as fórmulas do enunciado:

1. **Clientes atendidos:** $A_{s,d} = \min(7 \cdot X_{s,d} + 6 \cdot J_{s,d},\ C_{s,d})$, onde os Experts são usados primeiro;
2. **Unidades por cliente (Expert):** $U_X = \text{round}(F_X \cdot 10 / \ln(2 - PR_{s,d}))$;
3. **Unidades por cliente (Junior):** $U_J = \text{round}(F_J \cdot 10 / \ln(2 - PR_{s,d}))$;
4. **Lucro por cliente:** $P = \text{round}(U \cdot (1 - PR_{s,d}) \cdot 1.07)$;
5. **Lucro diário:** $R_{s,d} = \sum_{c=1}^{A_{s,d}} P_{s,d,c} - J_{s,d} \cdot \text{custo}_J - X_{s,d} \cdot \text{custo}_X$;
6. **Lucro semanal líquido:** $R_s = \sum_{d=1}^{7} R_{s,d} - W_s$.

Os fatores de produtividade e custos fixos semanais por loja são os definidos no enunciado:

| Loja | $F_J$ | $F_X$ | $W_s$ |
|------|--------|--------|--------|
| Baltimore | 1.00 | 1.15 | $700 |
| Lancaster | 1.05 | 1.20 | $730 |
| Philadelphia | 1.10 | 1.15 | $760 |
| Richmond | 1.15 | 1.25 | $800 |

Para O2, soluções que violam a restrição de 10 000 unidades globais recebem uma **penalidade de morte** (retorno de $-\infty$), garantindo que só soluções válidas são aceites. A função de avaliação foi verificada com os dados dos slides 17 e 18 do enunciado antes da execução dos métodos.

---

## 4.4 Métodos de Otimização Implementados

Foram implementados seis métodos de otimização, aplicados a todos os objetivos (O1, O2, O3):

| Método | Tipo | Parâmetros principais |
|--------|------|-----------------------|
| **Random Search** | Pesquisa cega | $n\_iter = 100$ |
| **Hill Climbing** | Pesquisa local | $max\_iter = 300$ |
| **Simulated Annealing** | Pesquisa local estocástica | $max\_iter = 300$, $T_0 = 100$ |
| **Genetic Algorithm** | Evolutivo global | $pop = 30$, $gen = 50$ |
| **Particle Swarm Optimization (PSO)** | Enxame global | $particles = 30$, $iter = 50$ |
| **Differential Evolution (DE)** | Evolutivo global | $pop = 30$, $gen = 50$ |

Para O3_NS foi utilizado exclusivamente o **NSGA-II** ($pop = 60$, $gen = 100$), por se tratar do único método multiobjetivo não-escalarizante implementado.

A metodologia de avaliação consistiu em **20 execuções independentes** (runs) de cada algoritmo, usando as previsões do melhor modelo de forecasting para a semana em análise (janela crescente / rolling window). O lucro médio, desvio padrão, melhor e pior resultado entre as 20 runs são reportados.

---

## 4.5 Resultados — Objetivo O1 (Otimização Local por Loja)

O O1 maximiza o lucro semanal líquido de cada loja de forma independente, sem restrições globais de unidades.

### 4.5.1 Comparação de Algoritmos — O1

**Tabela 1 — Lucro médio semanal ($) por algoritmo e por loja (20 execuções, O1)**

| Algoritmo | Baltimore | Lancaster | Philadelphia | Richmond |
|-----------|-----------|-----------|--------------|----------|
| **PSO** | **$3 197.68** (±$31.78) | $4 502.48 (±$33.34) | **$7 372.60** (±$7.92) | **$2 093.95** (±$0.21) |
| Genetic | $2 977.95 (±$57.20) | **$4 524.00** (±$93.13) | $6 991.08 (±$172.85) | $1 991.85 (±$18.88) |
| DE | $2 649.25 (±$25.60) | $3 962.13 (±$15.87) | $6 426.03 (±$77.32) | $1 773.13 (±$12.27) |
| Random | $2 676.08 (±$68.20) | $4 099.40 (±$27.51) | $6 296.08 (±$114.37) | $1 795.95 (±$3.61) |
| Hill Climbing | $2 406.02 (±$0.60) | $3 450.05 (±$3.75) | $5 810.35 (±$0.00) | $1 611.68 (±$0.32) |
| Simulated Annealing | $2 203.33 (±$24.22) | $3 317.80 (±$30.33) | $5 452.08 (±$71.74) | $1 450.60 (±$24.82) |

**Conclusões O1:**
- O **PSO** é o melhor algoritmo global, vencendo em Baltimore, Philadelphia e Richmond.
- O **Genetic Algorithm** supera ligeiramente o PSO em Lancaster ($4 524 vs $4 502).
- Os métodos de pesquisa local (Hill Climbing, Simulated Annealing) convergem rapidamente mas para soluções de menor qualidade.
- O **Hill Climbing** apresenta desvio padrão próximo de zero, indicando que converge deterministicamente para o mesmo ótimo local em todas as runs.
- A loja com maior lucro absoluto é **Philadelphia** ($7 373/semana com PSO), refletindo o maior volume de clientes previstos.

### 4.5.2 Plano Ótimo — O1 (Baltimore, PSO)

O plano semanal ótimo obtido pelo PSO para Baltimore (O1) é apresentado na Tabela 2.

**Tabela 2 — Plano semanal ótimo O1, Baltimore (melhor run PSO)**

| Dia | Nome | Clientes | Atendidos | Juniores | Experts | Unidades | Custo RH ($) | Lucro ($) |
|-----|------|----------|-----------|----------|---------|----------|--------------|----------|
| 1 | Domingo | 93 | 91 | 7 | 7 | 1 421 | 1 155 | 365 |
| 2 | Segunda | 77 | 77 | 0 | 11 | 1 309 | 880 | 521 |
| 3 | Terça | 87 | 84 | 0 | 12 | 1 428 | 960 | 568 |
| 4 | Quarta | 113 | 111 | 1 | 15 | 1 869 | 1 260 | 740 |
| 5 | Quinta | 99 | 98 | 0 | 14 | 1 666 | 1 120 | 663 |
| 6 | Sexta | 107 | 105 | 0 | 15 | 1 785 | 1 200 | 710 |
| 7 | Sábado | 119 | 119 | 7 | 11 | 1 897 | 1 535 | 495 |
| **TOTAL** | | **695** | **685** | **15** | **85** | **11 375** | **8 110** | **4 062** |
| **Líquido** ($-W_s$) | | | | | | | | **$3 362** |

**Observações:** A loja opera todos os 7 dias. Sem restrição de unidades, o otimizador maximiza o número de Experts (que têm maior fator de produtividade $F_X = 1.15$). A promoção é sempre 0%, confirmando matematicamente que PR = 0 maximiza o lucro por cliente. A taxa de atendimento é de 98.6%.

---

## 4.6 Resultados — Objetivo O2 (Otimização Global com Restrição de Unidades)

O O2 maximiza o lucro conjunto das quatro lojas sujeito à restrição de ≤ 10 000 unidades totais vendidas em toda a semana e em todas as lojas.

### 4.6.1 Comparação de Algoritmos — O2

**Tabela 3 — Lucro total global ($) por algoritmo (20 execuções, O2)**

| Algoritmo | Média ($) | Desvio Padrão ($) | Melhor Run ($) | Pior Run ($) |
|-----------|-----------|-------------------|----------------|--------------|
| **PSO** | **$1 056.05** | ±$74.93 | **$1 140** | $901 |
| DE | $783.00 | ±$51.03 | $888 | $685 |
| Genetic | $459.90 | ±$74.90 | $589 | $285 |
| Simulated Annealing | $416.25 | ±$80.09 | $615 | $276 |
| Hill Climbing | $333.25 | ±$100.96 | $558 | $158 |
| Random | $81.30 | ±$33.07 | $122 | $27 |

**Conclusões O2:**
- O **PSO** mantém a supremacia no contexto global, com lucro médio de $1 056/semana — **35% superior ao segundo melhor (DE: $783)**.
- O Random Search apresenta resultados muito fracos, evidenciando a dificuldade do espaço de pesquisa com restrições ativas.
- O elevado desvio padrão do Hill Climbing ($100.96) indica grande sensibilidade às condições iniciais quando a restrição de unidades força soluções próximas da fronteira.
- O PSO alcança consistentemente o melhor resultado: mesmo na pior run ($901) supera a média do DE ($783).

### 4.6.2 Plano Ótimo — O2 (Melhor Run PSO, $1 140 global)

Sob a restrição de 10 000 unidades, o otimizador distribui os recursos de forma assimétrica pelas lojas, priorizando as de maior margem de lucro. O plano do PSO na melhor run utiliza **9 989 das 10 000 unidades permitidas** (99.9% da capacidade).

**Tabela 4 — Resumo do plano O2 por loja (melhor run PSO)**

| Loja | Dias Abertos | Unidades | Lucro Operacional ($) | $W_s$ ($) | Lucro Líquido ($) |
|------|-------------|----------|----------------------|-----------|------------------|
| Baltimore | 2 (Seg, Sex) | 1 547 | 616 | 700 | −84 |
| Lancaster | 2 (Ter, Qui) | 1 428 | 568 | 730 | −162 |
| Philadelphia | 3 (Seg, Qua, Sex) | 2 856 | 1 137 | 760 | +377 |
| Richmond | 4 (Ter, Qua, Qui, Sex) | 4 158 | 1 809 | 800 | +1 009 |
| **Global** | **11** | **9 989** | **4 130** | **2 990** | **+1 140** |

**Observações:** O otimizador fecha algumas lojas em determinados dias porque o custo fixo semanal $W_s$ é pago independentemente dos dias de operação. As lojas Baltimore e Lancaster apresentam lucro líquido negativo individualmente, mas a restrição global de unidades obriga a concentrar recursos em Philadelphia e Richmond (maior margem). Richmond opera 4 dias e contribui com $1 009 do lucro total. Os fins de semana são evitados por terem custos de RH superiores (Expert: $95 vs $80 em dia útil).

---

## 4.7 Resultados — Objetivo O3 (Multiobjetivo: Lucro vs. RH)

O O3 estende o O2 com um segundo objetivo: minimizar o total de recursos humanos mobilizados. Foram implementadas duas abordagens.

### 4.7.1 O3_WEIGHTED — Função Escalarizante com Pesos

A função de avaliação combina lucro normalizado e custo de RH normalizado numa escala [0, 1]:

$$f = w \cdot \frac{\text{Lucro}}{\text{Lucro}_{max}} + (1-w) \cdot \left(1 - \frac{\text{HR}}{\text{HR}_{max}}\right)$$

Com peso $w = 0.7$ (maior ênfase no lucro). Os valores são normalizados pelos máximos teóricos calculados a partir das previsões de clientes.

**Tabela 5 — Score normalizado médio por algoritmo (20 execuções, O3_WEIGHTED)**

| Algoritmo | Score Médio | Desvio Padrão | Melhor Run | Pior Run |
|-----------|-------------|---------------|------------|----------|
| **PSO** | **0.04923** | ±0.00370 | 0.05372 | 0.04143 |
| DE | 0.03635 | ±0.00248 | 0.04053 | 0.03235 |
| Genetic | 0.02212 | ±0.00396 | 0.02882 | 0.01301 |

*Nota: Apenas PSO, DE e Genetic foram executados em O3_WEIGHTED por limitações computacionais. O score é adimensional (normalizado).*

O PSO obtém o melhor equilíbrio entre lucro e minimização de RH, com score de 0.049 na média.

### 4.7.2 O3_NS — NSGA-II e Frente de Pareto

O NSGA-II realiza uma otimização genuinamente multiobjetivo, gerando um conjunto de soluções não-dominadas (frente de Pareto) que representam diferentes compromissos entre maximizar o lucro e minimizar o RH.

**Tabela 6 — Resultados NSGA-II (20 execuções, O3_NS)**

| Algoritmo | Lucro Médio ($) | Desvio Padrão | Melhor Run ($) | Pior Run ($) |
|-----------|----------------|---------------|----------------|--------------|
| NSGA-II | $73.55 | ±$38.53 | $122 | −$3 |

O lucro médio do O3_NS é significativamente inferior ao O2 ($73 vs $1 056), o que é esperado: ao tentar simultaneamente minimizar RH, o algoritmo gera soluções com menor número de trabalhadores e, consequentemente, menor lucro.

**Frente de Pareto (soluções representativas):**

| Lucro ($) | Total RH |
|-----------|----------|
| 122 | 75 |
| −16 | 74 |
| −74 | 73 |
| −161 | 71 |
| −244 | 70 |
| −329 | 69 |
| −525 | 66 |

A frente de Pareto evidencia o trade-off: reduzir o RH de 75 para 66 trabalhadores implica uma queda de lucro de $122 para −$525. A solução com Lucro = $122 e HR = 75 constitui o ponto de melhor compromisso para um gestor que valorize simultaneamente ambos os objetivos.

---

## 4.8 Análise de Convergência

A análise de convergência permite avaliar a eficiência de cada algoritmo ao longo das iterações. Os gráficos de convergência foram produzidos para cada loja (O1, visão geral e detalhe) e globalmente (O2 e O3_NS).

**Principais observações:**

- **PSO** converge rapidamente nas primeiras iterações e atinge valores superiores, confirmando a sua adequação ao problema.
- **Genetic Algorithm** tem uma convergência mais lenta mas consistente, atingindo valores competitivos com o PSO em Lancaster.
- **Hill Climbing** e **Simulated Annealing** convergem muito rapidamente (em menos de 50 iterações) mas estabilizam em soluções subótimas — típico de métodos de pesquisa local sem mecanismos de exploração global.
- **DE** apresenta convergência gradual e resultados intermédios, sendo o segundo melhor método em O2.
- **Random Search** melhora lentamente ao longo de todas as iterações sem nunca estabilizar, refletindo a ausência de memória entre iterações.

Os gráficos de convergência por loja (visão geral e primeiras iterações) estão disponíveis em `reports/convergence_{loja}_O1_coarse.png` e `reports/convergence_{loja}_O1_fine.png`. A convergência global de O2 está em `reports/global_convergence_O2.png`.

---

## 4.9 Síntese Comparativa dos Três Objetivos

**Tabela 7 — Comparação resumida entre objetivos (melhor algoritmo, melhor run)**

| Objetivo | Melhor Algoritmo | Lucro Global ($) | Restrição | RH Total |
|----------|-----------------|-----------------|-----------|----------|
| O1 (Global estimado) | PSO + Genetic | ~$17 062* | Nenhuma | Elevado |
| O2 | PSO | $1 140 | ≤ 10 000 unidades | Moderado |
| O3_WEIGHTED | PSO | — (score: 0.054) | ≤ 10 000 unidades | Otimizado |
| O3_NS (Pareto) | NSGA-II | $122 (melhor Pareto) | ≤ 10 000 unidades | 75 |

*\*Soma das melhores runs de O1 por loja: Baltimore ($3 362) + Lancaster ($4 524) + Philadelphia ($7 373) + Richmond ($2 094) = $17 353, sem restrição global.*

**O impacto da restrição de unidades (O2 vs O1)** é muito significativo: o lucro global cai de ~$17 000 para $1 140, uma redução de ~93%. Isto ocorre porque a restrição de 10 000 unidades é muito restritiva face ao volume de clientes previsto (~695 clientes/dia × 4 lojas × 7 dias).

---

## 4.10 Discussão

**Promoção (PR):** Em todas as soluções ótimas encontradas, o valor de PR nos dias de operação é 0%. Esta é a solução matematicamente correta: o aumento de PR incrementa as unidades por cliente mas reduz a receita por unidade, resultando sempre num lucro por cliente inferior. Qualquer valor de PR ≠ 0 que apareça nos dados refere-se exclusivamente a dias encerrados, onde não tem impacto no lucro.

**PSO como melhor método:** O PSO destacou-se consistentemente em O1 e O2. A sua natureza de enxame com memória individual e social permite explorar eficientemente espaços de grande dimensão (84 parâmetros), ao contrário de métodos de pesquisa local que ficam presos em ótimos locais.

**Restrição global O2:** A penalidade de morte aplicada a soluções que violam as 10 000 unidades cria uma barreira abrupta no espaço de soluções. O PSO lida melhor com esta descontinuidade graças à sua exploração distribuída, enquanto o Hill Climbing e o SA frequentemente ficam presos em regiões infeasíveis ou adjacentes à fronteira.

**Previsões vs. valores reais:** A utilização das previsões do melhor modelo (XGBoost/ARIMA) introduz alguma incerteza nos planos. As 20 runs de otimização capturam esta variabilidade, permitindo obter estimativas robustas do lucro esperado.

---

*Os resultados apresentados neste capítulo foram obtidos com 20 execuções independentes por algoritmo. O código de otimização encontra-se em `src/optimization/` e os resultados em `reports/`. A visualização interativa dos planos e comparações está disponível no dashboard Streamlit desenvolvido (Tab "Análise Técnica" e Tab "Análise de Lojas").*
