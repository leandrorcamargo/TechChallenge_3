# Tech Challenge – Fase 3
## Predição e Inteligência Analítica para Alfabetização no Brasil

Projeto integrador da Fase 3 da Pós-Tech. Onde a Fase 2 entregou a **pipeline de engenharia de
dados** do Indicador Criança Alfabetizada, esta fase parte da camada analítica construída lá e
desenvolve **modelos supervisionados de Machine Learning** capazes de antecipar risco educacional e
transformar dado público em decisão de política pública.

A implementação é toda em Python, com `scikit-learn` como espinha dorsal: pré-processamento,
transformação de variáveis e modelo vivem dentro de um único `Pipeline`, que é o que garante
reprodutibilidade e impede vazamento de dados entre treino e teste.

---

## 1. Contexto do problema

A alfabetização nos primeiros anos escolares é a base do desenvolvimento educacional e social do
país. O **Compromisso Nacional Criança Alfabetizada** reúne União, estados, DF e municípios em torno
de uma meta: toda criança alfabetizada até o fim do 2º ano do ensino fundamental, com 80% de
alfabetização até 2030.

A partir da Pesquisa Alfabetiza Brasil (INEP, 2023), o Saeb passou a usar **743 pontos** como ponto
de corte na escala de Língua Portuguesa do 2º ano: acima disso, a criança é considerada
alfabetizada. Esse parâmetro deu origem ao Indicador Criança Alfabetizada.

O indicador nacional saiu de **55,9%** (2023) para **66,0%** (2025) na rede pública. É um avanço
real — e insuficiente, porque a média esconde o problema. Vale registrar que a rede avaliada é
essencialmente pública: em 2025, 87,6% dos alunos avaliados estão na rede municipal e 12,4% na
estadual. Entre a melhor e a pior unidade da
federação há **34,8 pontos percentuais** de diferença, e dentro de um mesmo município a variação
entre escolas chega a um desvio-padrão médio de **14,6 p.p.**

Olhar o indicador depois que ele sai é tarde demais para agir. O gestor público precisa saber
**antes** onde o risco está se formando.

## 2. Objetivo analítico

Desenvolver modelos supervisionados que respondam a duas perguntas, em dois grãos diferentes:

| Grão | Pergunta | Alvo |
|---|---|---|
| **Aluno** | Esta criança será considerada alfabetizada? | `proficiência ≥ 743` |
| **Município** | Este município vai ficar abaixo da meta do ano? | `taxa observada < meta do ano` |

E, a partir deles, responder às perguntas de negócio do desafio: quais fatores mais impactam a
alfabetização, quais municípios apresentam maior risco educacional, quais regiões têm padrões
semelhantes, como antecipar quem não vai atingir as metas futuras e quais variáveis pesam mais.

O foco não é maximizar métrica. É produzir **inteligência aplicável** — e, quando o modelo não for
melhor que uma conta simples, dizer isso com todas as letras (spoiler: em um dos cenários, é
exatamente o que acontece).

## 3. Base de dados

### Fonte principal — microdados do INEP

| Tabela | 2023 | 2024 | 2025 |
|---|---|---|---|
| `TS_ALUNO` (registros) | 1.747.439 | 2.120.560 | 2.222.792 |
| Municípios | 4.873 | 5.519 | 5.556 |
| Escolas | 36.776 | 42.497 | 43.644 |
| Base de modelagem (presentes) | 1.502.809 | 1.851.852 | 1.966.605 |
| Alfabetizados | 58,4% | 59,8% | 66,3% |

### Integrações herdadas da camada Gold da Fase 2

Metas nacionais, estaduais e municipais do Compromisso Nacional (trajetória 2024–2030), indicadores
agregados por município e UF da Base dos Dados, e o percentual de participação de cada município na
avaliação.

### Enriquecimento externo (IBGE)

Coletado por `src/preprocessing/fetch_externo.py` direto das APIs públicas e materializado em
`data/external/`: hierarquia territorial (município, UF, região, mesorregião), **população
residente**, **PIB per capita** e **participação da administração pública no valor adicionado**
— um proxy de dependência econômica do setor público.

### Validação cruzada de fontes

Como a Fase 2 fez entre camadas, comparamos a taxa calculada dos microdados com a taxa publicada na
Base dos Dados: **92,9% dos municípios ficam dentro de 1 ponto percentual**, com diferença absoluta
média de **0,33 p.p.** No nível nacional, a série que calculamos (58,4 / 59,8 / 66,3) acompanha o
indicador publicado (55,9 / 59,2 / 66,0), com aderência quase exata em 2024 e 2025 e diferença de
2,5 p.p. em 2023 — ano de cobertura parcial da avaliação (4.873 municípios contra 5.556 em 2025), em
que os critérios de participação mínima aplicados na publicação oficial pesam mais no agregado.

---

## 4. Etapas de modelagem

### 4.1 Definição do universo

O INEP registra `IN_ALFABETIZADO = 0` para alunos **ausentes** — não porque foram avaliados e
ficaram abaixo do corte, mas porque não há proficiência calculada. Manter esses registros faria o
modelo aprender a prever **falta escolar**, não alfabetização. O universo de modelagem são os alunos
presentes e com proficiência calculada.

### 4.2 Tratamento de data leakage

Mapeamos três camadas de vazamento e bloqueamos as três:

1. **Derivação direta do alvo** — `VL_PROFICIENCIA_LP` *define* o rótulo. Usá-la daria 100% de
   acurácia e valor zero.
2. **Artefatos da aplicação** — respostas item a item, gabaritos, caderno aplicado e peso amostral
   só existem depois da prova.
3. **Agregados do próprio ano** — a taxa do município em *t* é calculada com o resultado dos colegas
   do próprio aluno. É o vazamento sutil, o que passa despercebido e infla a métrica.

A regra que resolve as três: **todo atributo de contexto vem do ano anterior** (sufixo `_lag`). A
lista explícita de colunas proibidas está em `src/config.py::COLUNAS_LEAKAGE`.

### 4.3 A armadilha do `ID_ESCOLA`

O atributo mais promissor seria o histórico **por escola**. Ao testá-lo, a correlação com o alvo
ficou em 0,06 — ruído. Investigando: o INEP **reanonimiza o identificador de escola a cada edição**
dos microdados. Dos 42.391 ids presentes em 2024 e 2025, apenas **2,0%** estão no mesmo município
nos dois anos.

Cruzar escola entre anos colaria o passado da escola A no aluno da escola B. Todo histórico por
escola foi removido; da escola permanece apenas o que é estrutural e conhecido antes da prova
(porte da turma avaliada). O contexto defasado passou a ser construído no nível de **município e
UF**, cujos códigos IBGE são estáveis. O teste está reproduzido em
`notebooks/01_analise_exploratoria.ipynb`.

### 4.4 Engenharia de atributos

33 atributos no modelo de aluno, organizados em cinco blocos:

- **Desempenho do município em t−1** — taxa de alfabetização, proficiência média e dispersão,
  % em defasagem severa, desigualdade entre escolas, desempenho da pior escola, taxa de presença;
- **Desempenho da UF em t−1** — taxa, proficiência média e dispersão, presença;
- **Estrutura do ano corrente** — porte da escola e do município (matrícula, não resultado);
- **Metas e posição relativa** — meta do ano, meta de 2030, nível INEP, participação, distância
  para a meta, diferença em relação à própria UF, posição percentual dentro do estado;
- **Socioeconômico e territorial** — população (log), PIB per capita, % do PIB em administração
  pública, região, porte do município.

Os atributos derivados são os que traduzem o problema para a linguagem do gestor: `gap_meta_ano`,
`gap_meta_2030`, `esforco_necessario` (quantos pontos percentuais faltam) e `posicao_relativa_uf`.

### 4.5 Pipeline de pré-processamento

```
ColumnTransformer
├── numéricas   → SimpleImputer(mediana, add_indicator=True) → StandardScaler*
└── categóricas → SimpleImputer(moda) → OneHotEncoder(handle_unknown="ignore")
                                      ↓
                            conversão para float32
                                      ↓
                                   modelo
```

\* padronização apenas para modelos lineares; modelos de árvore dispensam.

Três decisões merecem justificativa. A imputação usa **mediana** por robustez a assimetria, com
**indicador de ausência**, porque a falta de histórico não é aleatória — sinaliza município pequeno
ou de participação irregular, e isso é informação. O `handle_unknown="ignore"` é obrigatório aqui:
municípios e categorias que aparecem só em 2025 não existiam em 2024, e sem isso a aplicação
out-of-time quebraria. E a conversão para `float32` corta o consumo de memória pela metade em uma
matriz de dois milhões de linhas.

Manter o pré-processamento **dentro** do objeto do modelo não é organização de código: é o que
impede que a mediana e as categorias do conjunto de teste vazem para o treino durante a validação
cruzada.

### 4.6 Desenho de validação

**Modelo de aluno — out-of-time.** Treino nos alunos de 2024 (contexto de 2023), teste nos alunos de
2025 (contexto de 2024). É exatamente como o modelo seria usado: estimar o risco da coorte que ainda
vai ser avaliada. Um split aleatório dentro do mesmo ano superestimaria o desempenho, porque alunos
do mesmo município apareceriam dos dois lados. A seleção de modelos usou validação cruzada em duas
óticas — estratificada (novos alunos) e **agrupada por município** (municípios nunca vistos).

**Modelo de município — por que o ano-base é 2025.** As metas municipais foram calculadas a partir
da linha de base de 2023: a correlação entre a meta de 2024 e a taxa de 2023 é **0,977**.
Consequência: para o ano de 2024, o "esforço necessário" (meta do ano − taxa do ano anterior) é
quase uma constante — desvio-padrão de 6,0 p.p. e **AUC de 0,505**, ou seja, sem qualquer poder
discriminante. Treinar em 2024 seria treinar em um problema **degenerado**, em que a meta e o
atributo saem da mesma medição. Em 2025 a defasagem é real e a mesma variável passa a discriminar:
desvio-padrão de 14,8 p.p. e **AUC de 0,744**. A modelagem usa a safra de 2025, com predições fora
da amostra por validação cruzada estratificada e um teste de estresse agrupado por UF. O ano de 2024
permanece no relatório como **diagnóstico**, não como treino.

---

## 5. Escolha do algoritmo

Cinco candidatos foram comparados, do trivial ao mais expressivo, sempre com o mesmo
pré-processamento.

**Grão aluno** (amostra de 150 mil alunos, validação cruzada 5-fold):

| Modelo | AUC estratificado | AUC por município |
|---|---|---|
| Baseline (classe majoritária) | 0,5000 | 0,5000 |
| Regressão Logística | 0,6622 | 0,6549 |
| Árvore de Decisão | 0,6606 | 0,6474 |
| Random Forest | 0,6697 | 0,6604 |
| **LightGBM** | **0,6713** | 0,6567 |

**Grão município** (5.338 municípios, validação cruzada 5-fold):

| Modelo | AUC estratificado | AUC por UF |
|---|---|---|
| Baseline (classe majoritária) | 0,5000 | 0,5000 |
| **Baseline analítico — ordenar pelo esforço necessário** | **0,7443** | **0,7406** |
| Árvore de Decisão | 0,7568 | 0,7135 |
| Regressão Logística | 0,7925 | 0,7427 |
| Random Forest | 0,8017 | 0,7449 |
| **LightGBM** | **0,8033** | 0,7340 |

O **baseline analítico** é a régua que importa: ordenar os municípios por "quantos pontos
percentuais faltam para a meta" é uma conta de subtração, sem modelo nenhum. Colocá-lo na tabela é o
que separa um ganho real de um número bonito.

**Escolha: LightGBM** nos dois grãos. Lida nativamente com relações não lineares e interações entre
variáveis territoriais, treina rápido em milhões de linhas, e a regularização (`reg_lambda`,
`min_child_samples`, `subsample`, `colsample_bytree`) entrou explicitamente no espaço de busca para
conter overfitting. Os hiperparâmetros foram otimizados por `RandomizedSearchCV`.

O ganho do LightGBM sobre a regressão logística no grão aluno é pequeno — e isso já é informação: a
relação entre contexto e alfabetização é majoritariamente monotônica, sem grandes interações
escondidas.

---

## 6. Métricas de avaliação

### Modelo de aluno — out-of-time (1.966.605 alunos de 2025)

| Conjunto | AUC | Acurácia | Acurácia balanceada | Precisão | Recall | F1 | Brier |
|---|---|---|---|---|---|---|---|
| Treino 2024 (in-sample) | 0,676 | 0,644 | 0,600 | 0,662 | 0,827 | 0,736 | 0,218 |
| **Teste 2025 (out-of-time)** | **0,639** | 0,660 | 0,563 | 0,697 | 0,859 | 0,770 | 0,214 |
| Teste, limiar calibrado (0,64) | 0,639 | 0,590 | 0,601 | 0,754 | 0,565 | 0,646 | 0,214 |

### Modelo de município — safra 2025 (5.338 municípios)

| Conjunto | AUC | Acurácia | Acurácia balanceada | Precisão | Recall | F1 |
|---|---|---|---|---|---|---|
| In-sample | 0,873 | 0,832 | 0,734 | 0,805 | 0,516 | 0,629 |
| **Fora da amostra (municípios novos)** | **0,806** | 0,792 | 0,683 | 0,694 | 0,441 | 0,539 |
| Fora da amostra, limiar calibrado (0,27) | 0,806 | 0,735 | 0,731 | 0,514 | 0,722 | 0,600 |
| Teste de estresse (estados novos) | 0,720 | 0,720 | 0,621 | 0,490 | 0,400 | 0,441 |

### Limiar de decisão

Fixar o corte em 0,5 é arbitrário, e aqui os erros custam coisas diferentes: **deixar de sinalizar
uma criança ou um município em risco** significa não enviar apoio a quem precisa; **sinalizar a
mais** custa uma vaga adicional em um programa de reforço. Os limiares foram calibrados por
acurácia balanceada. No modelo municipal, o corte em 0,27 leva o recall de 0,44 para **0,72** — de
cada dez municípios que vão ficar abaixo da meta, o modelo sinaliza sete.

---

## 7. Interpretação dos resultados

### O que o modelo de aluno diz — e o que ele não pode dizer

A distância entre treino e teste é pequena (0,676 → 0,639): **o modelo generaliza**. Era o principal
risco a controlar, e o desenho segurou.

Mas o AUC de 0,639 é modesto, e o motivo é estrutural: estamos prevendo um resultado **individual**
usando apenas variáveis de **contexto**. Duas crianças da mesma escola, no mesmo município, recebem
exatamente a mesma predição — o modelo não tem como distingui-las. Os microdados da Avaliação da
Alfabetização não trazem características socioeconômicas do estudante, e toda a variação individual
(trajetória escolar, apoio familiar, frequência, professor) está fora dos dados disponíveis.

Some-se a isso a forma da distribuição: a proficiência é aproximadamente normal e o corte de 743 cai
**perto do centro**. Uma parcela enorme de crianças está a poucos pontos da linha, e pequenas
variações viram mudança de classe.

Isso não invalida o modelo — **redefine o seu uso**. Ele não serve para rotular uma criança
específica, e sim para estimar **risco agregado** e ordenar territórios. O Brier score de 0,214 e a
curva de calibração mostram que as probabilidades são bem calibradas, o que é justamente o que
permite somá-las e estimar *quantas* crianças estão em risco em cada município.

### O que o modelo de município diz

No agregado o ruído individual se cancela, e o desempenho sobe para **AUC 0,806** — um ganho de
**+0,062** sobre o baseline analítico. É a evidência quantitativa da tese central do projeto:
**este é um problema de inteligência territorial, não de rotulagem individual.**

O teste de estresse por UF é onde está a honestidade do trabalho: ao prever estados **inteiros**
nunca vistos, o modelo cai para 0,720 e fica **abaixo** do baseline analítico (0,741). A leitura é
direta — o ganho do modelo é **intra-estadual**. Cada UF tem sua própria política educacional e seu
próprio patamar, e o modelo aprende esses níveis. Para um estado nunca observado, não há o que
transferir além da regra da distância para a meta. Como todos os 5.338 municípios da projeção estão
em estados já observados, a validação estratificada é a que corresponde ao uso real; o teste por UF
delimita até onde o modelo pode ser estendido.

### O que pesa na predição (Feature Importance e SHAP)

No **grão aluno**, o bloco educacional defasado domina: proficiência média do município no ano
anterior (37,6% do ganho), proficiência média da UF (13,4%), meta do ano (7,3%) e taxa de
alfabetização do município (6,2%). As variáveis socioeconômicas aparecem em posição secundária.

No **grão município**, o esforço necessário lidera (22,2%), seguido pela distância para a meta do
ano (20,7%) e pela meta em si (8,8%). O que interessa é o que vem **depois**: a dispersão da
proficiência dentro do município (6,7%) e a taxa de presença na avaliação (5,3%). Municípios com
média aceitável mas alta desigualdade interna carregam risco escondido.

---

## 8. Insights encontrados

**1. O passado educacional do território prevê o futuro da criança.** A correlação de Spearman entre
a taxa municipal de 2024 e a de 2025 é **0,690**. Quem estava mal tende a continuar mal — sem
intervenção, a inércia é a regra.

**2. Renda não é destino.** A correlação entre PIB per capita e taxa de alfabetização entre
municípios é **0,044** — praticamente nula. O Ceará lidera o país com **84,0%**, à frente de estados
muito mais ricos; o Rio Grande do Sul aparece entre os cinco piores, com **53,7%**. Política
educacional local explica mais que orçamento. Para gestão pública, é o achado mais importante deste
projeto: significa que o resultado é **acionável**, não determinado pela economia local.

**3. A desigualdade dentro do município é sinal preditivo.** O desvio-padrão médio da taxa entre
escolas do mesmo município é de **14,6 p.p.**, e **25,6%** dos municípios têm ao menos uma escola
abaixo de 30% de alfabetização. A média municipal esconde bolsões de risco — e é neles que a
intervenção focada rende mais.

**4. Predição individual tem teto baixo; predição territorial funciona.** AUC 0,639 no grão aluno
contra 0,806 no grão município, com os mesmos dados e o mesmo pipeline. O grão certo importa mais
que o algoritmo.

**5. O risco é altamente concentrado.** Estimamos **731 mil crianças em risco** em 2025, e
**228 municípios** — de 5.556 — concentram **metade** delas. Uma política que atinja 4% dos
municípios já cobre metade do problema, o que muda completamente a viabilidade orçamentária de um
programa nacional de reforço.

**6. Um modelo precisa provar que vale mais que uma conta de subtração.** No cenário de estados
novos, não valeu. Reportar isso é parte do trabalho.

---

## 9. Limitações do projeto

**Ausência de variáveis do aluno.** Os microdados não trazem características individuais
(socioeconômicas, de trajetória ou de frequência). É a limitação que impõe o teto do modelo de
aluno, e nenhum ajuste de algoritmo a contorna.

**Série temporal curta.** Três edições da avaliação (2023–2025) e apenas duas safras de metas
observadas. Não há histórico suficiente para atributos de tendência de médio prazo, nem para
validação temporal repetida.

**Identificador de escola não rastreável.** A reanonimização anual do `ID_ESCOLA` impede análise
longitudinal por escola — que seria o grão mais útil para intervenção.

**Mudança de patamar entre edições.** A taxa nacional subiu 6,5 p.p. entre 2024 e 2025 e a
prevalência de municípios abaixo da meta caiu de 46% para 28%. As probabilidades absolutas precisam
ser recalibradas a cada edição; a ordenação é mais estável que o nível.

**Generalização limitada entre estados.** O teste por UF mostra queda para 0,720, abaixo do baseline
analítico. O modelo não deve ser aplicado a uma unidade federativa fora da base de treino sem
revalidação.

**Cobertura das metas.** Municípios sem meta publicada (cerca de 4%) ficam fora da base municipal.

**Correlação não é causalidade.** As importâncias indicam associação, não efeito causal. O modelo
diz *onde* olhar, não *o que* funciona — isso exige desenho experimental ou quase-experimental.

---

## 10. Aplicação prática para políticas públicas

| Uso | Como | Entregável |
|---|---|---|
| Priorizar municípios | Ranking de risco de não atingir a meta de 2026 | `reports/ranking_risco_municipios_2026.csv` |
| Dimensionar o programa | Soma das probabilidades de risco por território | `reports/municipios_criancas_em_risco.csv` |
| Focar dentro do município | Dispersão entre escolas e desempenho da pior escola | atributos do modelo |
| Monitorar trajetória | Distância para a meta do ano, recalculada a cada edição | pipeline reproduzível |

### Projeção para 2026

Aplicando o modelo ao contexto observado em 2025 contra a meta de 2026:

- **1.691 municípios** (31,7% dos 5.338 avaliados) ficam acima do limiar de risco;
- eles cobrem **857.718 crianças** avaliadas em 2025;
- o esforço médio desse grupo é de **9,7 pontos percentuais** para alcançar a meta.

O arquivo de ranking traz, por município, a probabilidade de risco, a taxa atual, a meta e quantos
pontos percentuais faltam — filtrável por UF, região ou porte, pronto para montar uma fila de
atendimento.

## 11. Possíveis evoluções futuras

**Integrar o Censo Escolar** para trazer infraestrutura, formação docente, razão aluno-professor e
jornada — as variáveis que descrevem *o que a escola faz*, e não apenas o que ela obteve.

**Cruzar com o Cadastro Único** para incorporar vulnerabilidade social no nível domiciliar, hoje
ausente da base.

**Modelo longitudinal por escola**, caso o INEP passe a publicar um identificador estável ou
disponibilize chave de ligação sob acordo — é o salto de qualidade mais evidente.

**Clusterização de municípios** por perfil de vulnerabilidade, para desenhar políticas por
arquétipo em vez de por ranking.

**Modelo causal** (diferenças-em-diferenças ou controle sintético) para avaliar o efeito de
programas específicos, respondendo *o que funciona* e não apenas *onde está o risco*.

**Recalibração automática** a cada edição da avaliação, com monitoramento de deriva de distribuição
— o teste out-of-time deste projeto mostra exatamente por que isso é necessário.

**Reintegração ao ambiente da Fase 2**, publicando o escore de risco como uma tabela da camada Gold
no Delta Lake, consumível por dashboards e pelos times de política pública.

---

## Estrutura do repositório

```
tech-challenge-fase3/
├── data/
│   ├── raw/          # microdados do INEP (.zip) e arquivos da Base dos Dados (.csv.gz)
│   ├── external/     # bases do IBGE materializadas (versionadas)
│   ├── interim/      # cache de leitura dos microdados (não versionado)
│   └── processed/    # bases analíticas prontas para modelagem (não versionado)
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_modelagem_aluno.ipynb
│   ├── 03_modelagem_municipio.ipynb
│   └── 04_interpretabilidade_e_insights.ipynb
├── src/
│   ├── config.py                    # caminhos, constantes de negócio, paleta, colunas de leakage
│   ├── eda.py                       # análise exploratória (gera figuras e estatísticas)
│   ├── preprocessing/
│   │   ├── loaders.py               # leitura das fontes brutas
│   │   ├── fetch_externo.py         # coleta das APIs do IBGE
│   │   ├── features.py              # engenharia de atributos e bases analíticas
│   │   ├── pipeline.py              # ColumnTransformer + Pipeline
│   │   └── build_datasets.py        # materialização das bases
│   ├── modeling/
│   │   ├── train_aluno.py           # modelo de aluno (out-of-time)
│   │   └── train_municipio.py       # modelo de município + projeção 2026
│   ├── evaluation/
│   │   ├── metrics.py               # métricas e calibração de limiar
│   │   └── interpretabilidade.py    # Feature Importance, SHAP e perguntas de negócio
│   └── visualization/
│       └── plots.py                 # tema escuro e gráficos do projeto
├── reports/          # métricas, comparações, rankings e insights (JSON/CSV)
├── images/           # figuras geradas pelo pipeline
├── models/           # modelos serializados (.joblib)
├── requirements.txt
└── README.md
```

## Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. baixar os microdados do INEP para data/raw/ (links no portal do INEP)
# 2. coletar as bases externas do IBGE
python -m src.preprocessing.fetch_externo

# 3. construir as bases analíticas
python -m src.preprocessing.build_datasets

# 4. análise exploratória
python -m src.eda

# 5. modelagem
python -m src.modeling.train_aluno
python -m src.modeling.train_municipio

# 6. interpretabilidade e insights
python -m src.evaluation.interpretabilidade
```

Todas as etapas usam `random_state` fixo (`src/config.py::RANDOM_STATE`). A comparação de modelos e
a busca de hiperparâmetros do modelo de aluno são materializadas em `reports/`, de modo que
reexecutar o script reaproveita o que já foi calculado — apagar os arquivos força o recálculo.

## Tecnologias

`pandas` e `pyarrow` para manipulação e persistência colunar; `scikit-learn` para pipeline,
validação e métricas; `LightGBM` como modelo final nos dois grãos; `SHAP` para interpretabilidade;
`matplotlib` para as figuras; `python-pptx` para a apresentação executiva.

## Fase 2

Repositório da fase anterior, que entregou a pipeline de engenharia de dados e a camada Gold
utilizada aqui: [github.com/leandrorcamargo/TechChallenge_2-](https://github.com/leandrorcamargo/TechChallenge_2-)

## Equipe

| Integrante | Contato |
|---|---|
| Isabelle Nicole Santana de Brito | isabelle_nicole@outlook.com |
| Filipe Noberto Justino | justinofilipe03@hotmail.com |
| Leandro Rebes Camargo | leandrorcamargo@hotmail.com |
| Felipe Vieira Sanches | fvieirasanches@gmail.com |
