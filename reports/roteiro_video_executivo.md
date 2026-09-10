# Roteiro — Vídeo Executivo (até 5 minutos)
## Tech Challenge Fase 3 · Predição e Inteligência Analítica para Alfabetização no Brasil

**Formato:** simulação de reunião executiva com gestores públicos, lideranças e stakeholders.
**Duração-alvo:** 4min50s. **Ritmo:** ~140 palavras por minuto — falar com calma, sem correr.

**Como usar:** cada bloco traz o slide correspondente, o tempo acumulado e o texto falado. O texto
está escrito para ser dito em voz alta, não lido de tela. Marcas de pausa: `//` respiração curta,
`///` pausa de efeito.

**Divisão sugerida entre os quatro integrantes:**

| Integrante | Blocos | Tempo |
|---|---|---|
| 1 | Abertura e o problema | 0:00 – 1:05 |
| 2 | Solução, dados e rigor metodológico | 1:05 – 2:20 |
| 3 | Resultados e o que o modelo enxerga | 2:20 – 3:30 |
| 4 | Insights, aplicação e fechamento | 3:30 – 4:50 |

---

## Bloco 1 — Abertura (0:00 – 0:20) · Slide 1

> Bom dia a todos. // Nós somos a equipe de ciência de dados desta organização, e viemos apresentar
> um trabalho que responde a uma pergunta simples de fazer e difícil de responder: /// **dá para
> saber, com antecedência, quais crianças brasileiras correm risco de não se alfabetizar — e onde
> elas estão?**

---

## Bloco 2 — O problema (0:20 – 1:05) · Slide 2

> Começo pela boa notícia. // O Indicador Criança Alfabetizada saiu de **55,9% em 2023** para
> **66,0% em 2025** na rede pública. Dez pontos em dois anos. É avanço real.
>
> Mas essa média esconde o problema. /// Entre a melhor e a pior unidade da federação há **quase 35
> pontos percentuais** de diferença. E dentro de um mesmo município, a variação entre escolas chega
> a um desvio-padrão de **quase 15 pontos**.
>
> Tem outra questão, e essa é a que nos trouxe aqui: // o indicador só fica pronto **depois** que a
> avaliação acontece. Quando o número chega à mesa do gestor, aquela turma já terminou o segundo
> ano. /// Nós queríamos saber antes.

---

## Bloco 3 — A solução (1:05 – 1:30) · Slide 3

> Construímos dois modelos, em dois grãos diferentes, porque são duas decisões diferentes.
>
> No **grão aluno**, o modelo prevê se a criança será considerada alfabetizada — proficiência acima
> de 743 pontos na escala Saeb. Isso serve para **dimensionar**: quantas crianças precisam de apoio.
>
> No **grão município**, ele prevê se o município vai ficar abaixo da meta do ano. Isso serve para
> **priorizar**: onde o apoio entra primeiro.

---

## Bloco 4 — Dados e rigor (1:30 – 2:20) · Slides 4, 5 e 6

> A base são os microdados do INEP de 2023 a 2025 — **seis milhões de registros de alunos** —
> integrados às metas do Compromisso Nacional que herdamos da nossa própria fase anterior, e
> enriquecidos com dados territoriais e socioeconômicos do IBGE.
>
> Quero destacar dois cuidados metodológicos, porque é neles que um projeto desses vive ou morre.
>
> O primeiro é **vazamento de dados**. // Os microdados trazem a nota da prova junto — e a nota
> *define* o rótulo. Se usássemos, teríamos 100% de acurácia e valor zero. Mapeamos três camadas de
> vazamento e bloqueamos as três: a regra é que **todo atributo de contexto vem do ano anterior**.
>
> O segundo foi uma armadilha nos próprios dados. /// A variável mais promissora seria o histórico
> **por escola**. Quando testamos, a correlação com o alvo deu **0,06** — ruído puro. Investigando,
> descobrimos que o INEP **reanonimiza o identificador da escola a cada edição**: de quarenta e dois
> mil identificadores presentes em 2024 e 2025, apenas **2%** estão no mesmo município nos dois
> anos. Estávamos colando o passado de uma escola no aluno de outra. // Removemos tudo e passamos o
> contexto para o nível de município, que usa código IBGE e é estável.

---

## Bloco 5 — Resultados (2:20 – 3:05) · Slides 8 e 9

> Os resultados. // No grão aluno, AUC de **0,64** prevendo 2025 com um modelo treinado em 2024 —
> dados que ele nunca viu, de um ano que ainda não existia no treino. A diferença para o treino é
> pequena: o modelo **generaliza**.
>
> Mas quero ser direto sobre esse número. /// Ele é modesto, e o motivo é estrutural: estamos
> prevendo um resultado **individual** usando só variáveis de **contexto**. Duas crianças da mesma
> escola recebem a mesma predição, porque os microdados públicos não trazem nada sobre a criança.
> Isso não invalida o modelo — **redefine o uso dele**. Ele não rotula uma criança; ele estima risco
> agregado.
>
> E é exatamente isso que aparece no grão município: **AUC de 0,806**. // No agregado, o ruído
> individual se cancela.
>
> Aqui vai a parte que a gente faz questão de mostrar. /// Nós comparamos o modelo com uma regra de
> uma linha: "ordene os municípios por quantos pontos faltam para a meta". Uma conta de subtração.
> Ela sozinha dá 0,744. Nosso modelo dá 0,806 — **ganho real**. Mas quando pedimos para ele prever
> **estados inteiros** que nunca viu, ele cai para 0,72 e **perde** para a conta de subtração. // O
> ganho do modelo é intra-estadual, e a gente diz isso no relatório.

---

## Bloco 6 — O que o modelo enxerga (3:05 – 3:30) · Slide 10

> Usando SHAP e importância de variáveis, o que pesa é o **desempenho educacional recente do
> território**: a proficiência média do município no ano anterior sozinha responde por **37% do
> ganho** do modelo. A proficiência do estado vem em seguida.
>
> As variáveis socioeconômicas aparecem — mas em segundo plano. E isso nos leva ao achado mais
> importante do trabalho.

---

## Bloco 7 — Insight: renda não é destino (3:30 – 4:00) · Slide 11

> A correlação entre PIB per capita e taxa de alfabetização, entre municípios, é **0,044**. ///
> Praticamente nula.
>
> O **Ceará** lidera o país com 84% das crianças alfabetizadas — à frente de estados muito mais
> ricos. O **Rio Grande do Sul** está entre as cinco piores unidades da federação, com 53,7%.
>
> A leitura é essa: /// **renda não determina alfabetização. Política educacional local, sim.** E
> isso é uma boa notícia para quem está nesta sala, porque significa que o resultado é **acionável**
> — não é destino econômico.

---

## Bloco 8 — Insight: o risco é concentrado (4:00 – 4:20) · Slide 12

> Somando as probabilidades do modelo, estimamos cerca de **731 mil crianças em risco** em 2025. //
> E **228 municípios** — de 5.556 — concentram **metade** delas.
>
> Isso muda a natureza do problema. /// Uma política que alcance **4% dos municípios** já cobre
> metade do desafio. Deixa de ser uma questão de cobrir o país inteiro e passa a ser uma questão de
> **priorizar bem**.

---

## Bloco 9 — Aplicação (4:20 – 4:40) · Slides 13 e 14

> Aplicando o modelo ao que observamos em 2025, contra a meta de 2026: **1.691 municípios** entram
> na faixa de alto risco, cobrindo **858 mil crianças**, com um esforço médio de quase **10 pontos
> percentuais** para alcançar a meta.
>
> A entrega prática é um **ranking priorizado**, com a probabilidade de risco, a taxa atual, a meta
> e quantos pontos faltam — filtrável por estado, região ou porte. É uma fila de atendimento pronta
> para virar decisão orçamentária.

---

## Bloco 10 — Fechamento (4:40 – 4:50) · Slide 16

> Encerro com a frase que resume o trabalho. /// **Alfabetização, do ponto de vista de dados, é um
> problema de inteligência territorial — não de rotulagem individual.**
>
> O modelo não diz quem é a criança. Ele diz **onde** ela está, **quantas** são, e **quanto** falta.
> // E isso é o suficiente para decidir. /// Obrigado.

---

## Checklist de gravação

- [ ] Falar em ritmo de reunião, não de apresentação decorada — pausar nos `///`
- [ ] Números redondos na fala (`0,64`, `0,81`, `731 mil`), precisão fica nos slides
- [ ] Não ler o slide: o slide sustenta, a fala conduz
- [ ] Manter o bloco de honestidade (o modelo perde para a regra em estados novos) — é diferencial
- [ ] Duração total: cronometrar em ensaio; se passar de 5 min, cortar o Bloco 6, não os insights
- [ ] Encerrar olhando para a câmera, sem "é isso" ou "acho que era isso"
