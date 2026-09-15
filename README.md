# Tech Challenge – Fase 3
## Predição e Inteligência Analítica para Alfabetização no Brasil 🇧🇷

Projeto integrador da Fase 3 da Pós-Tech. Enquanto a **Fase 2** construiu a pipeline de
engenharia de dados (Bronze → Silver → Gold) do **Indicador Criança Alfabetizada**, esta fase
utiliza a camada Gold resultante para desenvolver **modelos supervisionados de Machine
Learning** capazes de transformar dados públicos em inteligência aplicada à decisão.

---

##  Contexto do Problema

A alfabetização na infância é um dos principais indicadores de desenvolvimento educacional e
social do país. O **Compromisso Nacional Criança Alfabetizada** (Decreto nº 11.556/2023)
mobiliza União, estados, Distrito Federal e municípios para garantir que todas as crianças
estejam alfabetizadas até o fim do **2º ano do ensino fundamental**.

A partir da **Pesquisa Alfabetiza Brasil (INEP, 2023)** definiu-se o corte de **743 pontos**
na escala Saeb, a partir do qual uma criança é considerada alfabetizada. O **Indicador Criança
Alfabetizada** expressa o percentual de estudantes que atingem esse patamar, com **meta
nacional de 80% até 2030**.

Conhecer apenas os dados atuais, porém, não basta. Gestores públicos precisam **antecipar
riscos**, identificar regiões vulneráveis e compreender **quais fatores mais influenciam** o
indicador — para agir antes do resultado, não depois dele. É esse o espaço que a Ciência de
Dados ocupa aqui.

---

##  Objetivo Analítico

Desenvolver um **modelo supervisionado** capaz de prever se um aluno será considerado
**alfabetizado ou não alfabetizado**, utilizando variáveis educacionais, territoriais e
socioeconômicas provenientes da camada Gold da Fase 2.

O projeto trabalha em **duas granularidades complementares**, porque a decisão de política
pública acontece nas duas:

| Nível | Pergunta que responde | Uso |
|-------|----------------------|-----|
| **Agregado** (município/rede) | O município/rede vai atingir a meta de 80%? | Planejamento estratégico, alocação de recursos |
| **Individual** (aluno) | Este aluno será alfabetizado? | Intervenção pedagógica, alerta precoce |


---

##  Descrição da Base Utilizada

Os dados vêm da **camada Gold** construída na Fase 2, no Databricks (Unity Catalog + Delta Lake):

| Tabela | Conteúdo | Volume |
|--------|----------|--------|
| `workspace.gold.features_ml` | Features agregadas por município / rede / ano | ~24 mil registros |
| `workspace.default.microdados_alunos_gold` | Microdados de aluno enriquecidos com indicadores municipais | ~3,8 milhões de alunos |
| `workspace.gold.indicadores_municipio` | Indicadores municipais com classificações | — |
| `workspace.gold.metas_vs_resultados_uf` | Metas estaduais × resultados (2024–2030) | — |

### Origem dos dados

Microdados oficiais do **INEP** — Avaliação da Alfabetização / AEEB — integrados na Fase 2 com
as bases de **metas** (nacional, estadual e municipal) da
[Base dos Dados](https://basedosdados.org/).

### Variáveis do aluno disponíveis

A base `TS_ALUNO` do INEP contém 15 colunas. As relevantes para a modelagem:

| Variável | Descrição | Uso no modelo |
|----------|-----------|---------------|
| `IN_ALFABETIZADO` | Alvo: 0 = Não, 1 = Sim (proficiência ≥ 743) | **target** |
| `IN_PRESENCA_LP` | Indicador de presença **na prova** de LP (0/1) | feature |
| `TP_SERIE` | Série (constante: 2º ano) | descartada |
| `CO_CADERNO_LP` | Caderno de prova atribuído ao aluno | feature |
| `TP_DEPENDENCIA` | Rede (federal / estadual / municipal / privada) | feature |
| `ID_ESCOLA`, `CO_MUNICIPIO`, `SG_UF` | Identificação territorial | base das features de contexto |
| `VL_PROFICIENCIA_LP` | Nota na escala Saeb | **excluída — define o alvo** |


---

##  Etapas de Modelagem

```mermaid
flowchart LR
    G["Camada Gold<br/>(Fase 2)"] --> E["EDA"]
    E --> A["Auditoria de<br/>Data Leakage"]
    A --> F["Feature<br/>Engineering"]
    F --> P["Pipeline<br/>Scikit-learn"]
    P --> T["Treino +<br/>Validação"]
    T --> O["Otimização de<br/>Hiperparâmetros"]
    O --> I["Interpretabilidade<br/>(SHAP)"]
    I --> R["Análise de Risco<br/>+ Clustering"]
```

###  Análise Exploratória (EDA)

Distribuição do alvo, balanceamento de classes, valores ausentes, cardinalidade das variáveis
categóricas e distribuição da proficiência em relação ao corte de 743 pontos.

###  Auditoria de Data Leakage

Etapa central do projeto, e a que mais alterou os resultados. Foram identificadas variáveis que
**só existem depois** do resultado que se quer prever:

| Variável removida | Por quê |
|-------------------|---------|
| `proficiencia` | É a nota da prova — define o alvo diretamente pelo corte de 743 |
| `taxa_alfabetizacao` | Calculada a partir dos próprios alunos que se quer prever |
| `media_portugues` | Altamente correlacionada com a proficiência |
| `nivel_0` … `nivel_8` | Distribuição derivada do próprio alvo |
| `preenchimento_caderno` | Correlacionado ao desempenho na prova |

**Impacto medido:** de **99,86% de acurácia** (com vazamento) para **~65%** (sem vazamento). A
queda é o resultado correto — o primeiro número não se sustentaria em produção, porque a
proficiência não existe no momento em que a predição precisaria ser feita.

###  Tratamento do Split Temporal

Investigação da sobreposição de registros entre os anos de treino (2023) e teste (2024). A
função `filtrar_alunos_sem_repeticao` mantém apenas registros cujo `id_aluno` aparece em um
único ano, com o objetivo de impedir que o modelo memorize em vez de generalizar. O split em si
é feito por `split_temporal` (`src/preprocessing/data_loader.py`), que separa treino e teste por
ano de avaliação.

###  Feature Engineering

Implementado em `src/preprocessing/feature_engineering.py`. Criação de variáveis de
**contexto**, já que os microdados praticamente não trazem atributos do próprio aluno:

| Função | Variáveis geradas |
|--------|-------------------|
| `criar_features_escola` | presença média da escola, desvio-padrão, número de alunos |
| `criar_features_municipio_rede` | presença média e porte da rede municipal |
| `criar_features_interacao` | termos quadrático e cúbico da presença; diferença do aluno para a média da escola e do município; faixas de presença e de porte de escola |
| `adicionar_features_historicas` | taxa de alfabetização e média de português do município e da UF em ano anterior |

###  Pipeline Scikit-learn

Todo o pré-processamento é **integrado ao modelo**, garantindo que os parâmetros de
transformação sejam ajustados apenas no conjunto de treino. A construção está centralizada em
`src/modeling/pipeline_builder.py`, reutilizada pelos dois modelos:

```python
from src.modeling.pipeline_builder import build_xgboost_pipeline
from src.modeling.train import train_model, predict

pipeline, num_features, cat_features = build_xgboost_pipeline(X_train, modelo_tipo='agregado')
pipeline = train_model(pipeline, X_train, y_train, cross_validate=True)
y_pred, y_proba = predict(pipeline, X_test)
```

O parâmetro `modelo_tipo` seleciona o conjunto de hiperparâmetros: o modelo **agregado** usa
300 árvores com `max_depth=6`; o **individual**, 50 árvores com `max_depth=4`, `tree_method='hist'`
e regularização mais forte — configuração mais enxuta pelo volume de registros envolvido.

Internamente, `build_preprocessor` monta:

```python
ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ]), num_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])
```

###  Validação

- **Split temporal** — treino em 2023, teste em 2024 (simula predição de um ano futuro)
- **Cross-validation** — `StratifiedKFold` (5 folds) sobre o conjunto de treino
- **`random_state` fixo** em todas as etapas, garantindo replicabilidade

###  Otimização

Duas frentes, em lugares diferentes do projeto:

- **Hiperparâmetros** — `RandomizedSearchCV` sobre `learning_rate`, `max_depth`,
  `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha` e `reg_lambda`,
  explorado no notebook `01`. No código modular, os hiperparâmetros resultantes estão fixados
  em `build_xgboost_pipeline`, separados por tipo de modelo.
- **Limiar de decisão** — `otimizar_threshold` (`src/modeling/train.py`) varre cortes entre
  0,30 e 0,70 buscando o melhor ponto de operação, em vez de assumir o 0,5 padrão. Relevante
  aqui porque a classe positiva é majoritária e o corte padrão não é o ótimo.

---

##  Escolha do Algoritmo

**XGBoost (Extreme Gradient Boosting)** como modelo principal.

| Critério | Justificativa |
|----------|---------------|
| **Desempenho em dados tabulares** | Estado da arte para problemas estruturados como este |
| **Escala** | `tree_method='hist'` processa milhões de registros em CPU |
| **Valores ausentes** | Trata `NaN` nativamente, sem imputação obrigatória |
| **Regularização** | `reg_alpha` (L1), `reg_lambda` (L2), `gamma` e `min_child_weight` controlam overfitting |
| **Interpretabilidade** | `feature_importances_` nativo e compatibilidade direta com SHAP |
| **Desbalanceamento** | `scale_pos_weight` permite ajuste quando necessário |

Complementarmente, **KMeans** é utilizado para agrupar municípios com padrões semelhantes
(pergunta de negócio nº 3).

---

##  Métricas de Avaliação

| Métrica | Por que está aqui |
|---------|-------------------|
| **F1-Score** | Métrica principal. Média **harmônica** entre precisão e recall — e é por ser harmônica que pune o desequilíbrio entre as duas. Um modelo que declara todo mundo alfabetizado tem recall alto e precisão baixa; a média aritmética o perdoaria, a harmônica não. Essencial num problema em que o atalho de chutar a classe majoritária está sempre disponível |
| **Precision** | Dos alunos apontados como alfabetizados, quantos realmente são — evita falso otimismo |
| **Recall** | Dos que realmente são, quantos o modelo encontrou — evita deixar aluno em risco fora do radar |
| **AUC-ROC** | Independe de limiar; mede a capacidade de **ordenar** por risco, que é o uso prático do modelo |
| **Accuracy** | Reportada **por clareza de leitura**, por ser a métrica mais intuitiva na conversa com gestores. Não é usada para selecionar modelo |
| **Matriz de Confusão** | Explicita o custo de cada tipo de erro |


---

##  Interpretação dos Resultados

Evolução documentada ao longo das correções aplicadas:

| # | Modelo | Accuracy | AUC-ROC | Precision | Recall | F1-Score |
|---|--------|----------|---------|-----------|--------|----------|
| 0 | Com data leakage (`proficiencia`) | 99,86% | ~100% | — | — | — |
| 1 | Limpo, sem vazamento de features | 64,72% | 62,09% | 59,97% | 98,21% | 74,47% |
| 2 | + Split temporal corrigido | *preencher* | *preencher* | *preencher* | *preencher* | *preencher* |
| 3 | + Feature Engineering | *preencher* | *preencher* | *preencher* | *preencher* | *preencher* |
| 4 | + Otimização de hiperparâmetros | *preencher* | *preencher* | *preencher* | *preencher* | *preencher* |


**Leitura do modelo nº 1.** Recall de 98,21% com precisão de 59,97% descreve um modelo que
classifica quase todos os alunos como alfabetizados. Ele praticamente não erra ao identificar
quem será alfabetizado, mas quase não distingue quem não será — exatamente o comportamento que
o F1 e a matriz de confusão expõem e que a acurácia isolada esconderia.

**O modelo agregado** (município/rede) alcança desempenho superior ao individual. Não por ser
mais bem construído, mas porque o alvo é diferente: ao agregar centenas de alunos, a variação
individual — imprevisível com os dados disponíveis — se cancela, e resta o efeito do território,
que o histórico consegue capturar.

---

##  Insights Encontrados

**1. O contexto territorial vale mais que o cadastro individual.** Os microdados não trazem
sexo, raça, idade, nível socioeconômico nem frequência anual do aluno. Praticamente toda a
informação preditiva disponível é **onde o aluno estuda** — o que torna as features agregadas de
escola e município o principal ativo do modelo.

**2. Há forte desigualdade entre municípios.** O clustering (KMeans, k=5) separa grupos com
taxas médias que vão de ~36% a ~94% de alfabetização, evidenciando que a meta nacional exige
políticas diferenciadas por perfil, e não uma estratégia única.

**3. Vazamento de dados produz números convincentes e inúteis.** A queda de 99,86% para ~65% foi
o resultado mais importante do projeto: mostra que métrica alta sem auditoria é um risco, não
uma conquista.

---

##  Limitações do Projeto

**Ausência de variáveis do aluno.** O `TS_ALUNO` tem 15 colunas e nenhum atributo individual
(sexo, raça, idade, NSE, frequência ao longo do ano, histórico de reprovação). Dois colegas da
mesma escola são indistinguíveis para o modelo, o que impõe um teto à predição individual
independentemente do algoritmo escolhido.

**Ruído no próprio rótulo.** O corte de 743 pontos cai na região mais densa da distribuição de
proficiência. Alunos próximos à linha podem mudar de classificação por erro de medida do teste,
o que representa incerteza irredutível no alvo.

**`IN_PRESENCA_LP` não está disponível no momento da predição.** Ela só é observada no dia da
aplicação. Um alerta emitido no meio do ano letivo não sabe quem faltará — o que limita o uso da
variável em um cenário real de antecipação.

**Features de contexto escolar exigem defasagem temporal.** Em `criar_features_escola`, a
variável `escola_taxa_alfabetizacao` é calculada como `F.avg('alfabetizado')` agrupando por
`('id_escola', 'ano')` — ou seja, sobre o **mesmo ano** do alvo. O rótulo de cada aluno entra no
cálculo da sua própria feature, o que reintroduz vazamento. O contexto escolar precisa vir de
anos anteriores ao que se está prevendo.

**O join de features históricas não inclui o ano.** Em `adicionar_features_historicas`, o
`df_hist_2023` é unido por `['id_municipio', 'rede']`, sem `ano`. Com isso, linhas de treino
(2023) recebem indicadores do próprio 2023 enquanto linhas de teste (2024) recebem indicadores
defasados — treino e teste passam a significar coisas diferentes, e o ganho não transfere.

**O limiar é otimizado no conjunto de teste.** `otimizar_threshold` recebe `y_test` e escolhe o
corte que maximiza a acurácia nesse mesmo conjunto. Isso não melhora o modelo, apenas o número
reportado. O limiar deveria ser escolhido em um conjunto de validação separado.

**A premissa do filtro de alunos repetidos precisa ser confirmada.** `filtrar_alunos_sem_repeticao`
descarta registros cujo `id_aluno` aparece em mais de um ano, assumindo que se trata da mesma
criança. Vale verificar se o identificador é estável entre edições — se ele for reatribuído a
cada ano, o filtro estará descartando dados válidos sem corrigir vazamento algum.

**Split temporal sobre duas edições.** Treino em 2023 e teste em 2024 oferece apenas um par de
anos. A edição de 2025, já disponível nos microdados, permitiria validação fora do tempo
adicional e mais robusta.

**Fontes externas não incorporadas.** O enunciado sugere enriquecimento com IBGE, Censo Escolar,
FUNDEB, Atlas do Desenvolvimento Humano e Cadastro Único. Esse enriquecimento **não foi
realizado** nesta entrega e permanece como evolução futura.

**Dependência do ambiente Databricks.** Os notebooks leem tabelas do Unity Catalog
(`workspace.gold.*`), o que exige o workspace da Fase 2 provisionado para execução integral.

---

##  Aplicação Prática para Políticas Públicas

| Aplicação | Como o modelo apoia |
|-----------|--------------------|
| **Priorização orçamentária** | Ranqueamento de municípios por probabilidade de não atingir a meta, concentrando recursos onde o risco é maior |
| **Alerta precoce** | Identificação de escolas e alunos em risco antes do fim do ano letivo, viabilizando reforço pedagógico ainda a tempo de alterar o resultado |
| **Mobilização para a avaliação** | Como a ausência conta como não alfabetizado, ações de garantia de comparecimento têm efeito direto e imediato sobre o indicador |
| **Políticas por perfil regional** | Os clusters do KMeans agrupam municípios com padrões semelhantes, permitindo estratégias por grupo em vez de política única |
| **Monitoramento de metas** | Acompanhamento da distância entre resultado projetado e meta pactuada (`gap_meta`), município a município |

**Ciclo operacional sugerido:**

```
Coleta (início do ano) → Predição (junho) → Intervenção (jul–nov) → Reavaliação (out–nov)
```

---

##  Estrutura do Repositório

```
TechChallenge_3/
├── TechChallenge_Fase3/
│   ├── data/
│   │   └── sample_data.py                    # Amostragem das tabelas do Unity Catalog
│   ├── notebooks/
│   │   ├── 01_XGBoost_Alfabetizacao.py       # Notebook original (monolítico, mantido como histórico)
│   │   ├── 02_Modelo_Agregado_Modular.py     # Modelo município/rede usando src/
│   │   └── 03_Modelo_Individual_Modular.py   # Modelo por aluno usando src/
│   ├── src/
│   │   ├── preprocessing/
│   │   │   ├── data_loader.py                # Carga da Gold, target e split temporal
│   │   │   └── feature_engineering.py        # Seleção e criação de features
│   │   ├── modeling/
│   │   │   ├── pipeline_builder.py           # ColumnTransformer + XGBClassifier
│   │   │   └── train.py                      # Treino, CV, predição e threshold
│   │   ├── evaluation/
│   │   │   └── metrics.py                    # Métricas, matriz de confusão, importâncias
│   │   └── visualization/
│   │       └── plots.py                      # Curvas ROC/PR, distribuições, importâncias
│   ├── reports/
│   │   └── ESTRUTURA_PROJETO.md              # Documentação técnica da arquitetura
│   ├── RESUMO_ENTREGA.md                     # Resumo da reestruturação modular
│   ├── requirements.txt                      # Dependências fixadas
│   └── .gitignore
├── [IAST] - Tech Challenge - Fase 3.pdf      # Enunciado do desafio
└── README.md                                 # Este documento
```

A pasta `data/` não versiona arquivos de dados — os dados vivem no Unity Catalog. Ela contém
`sample_data.py`, que extrai amostras estratificadas das tabelas Gold e documenta quais
tabelas o projeto consome.

---

##  Organização do Código

O notebook original (`01_XGBoost_Alfabetizacao.py`) concentrava carga, features, modelagem,
avaliação e gráficos em um único arquivo. O código foi **modularizado em `src/`**, e os
notebooks `02` e `03` passaram a ser roteiros finos que orquestram esses módulos.

| Módulo | Responsabilidade | Principais funções |
|--------|------------------|--------------------|
| `preprocessing/data_loader.py` | Acesso à camada Gold e separação dos conjuntos | `load_features_ml`, `load_microdados_alunos`, `create_target_variable`, `split_temporal` |
| `preprocessing/feature_engineering.py` | Seleção e construção de variáveis | `select_features_modelo_agregado`, `criar_features_escola`, `criar_features_municipio_rede`, `criar_features_interacao`, `adicionar_features_historicas` |
| `modeling/pipeline_builder.py` | Montagem do pipeline Scikit-learn | `build_preprocessor`, `build_xgboost_pipeline`, `get_feature_names` |
| `modeling/train.py` | Treino, validação cruzada e limiar | `train_model`, `predict`, `otimizar_threshold` |
| `evaluation/metrics.py` | Cálculo e exibição de métricas | `calculate_metrics`, `print_confusion_matrix`, `get_feature_importance` |
| `visualization/plots.py` | Gráficos analíticos | `plot_confusion_matrix`, `plot_roc_curve`, `plot_precision_recall_curve`, `plot_feature_importance` |

**Por que modularizar.** O enunciado pede pipeline reproduzível e código-fonte organizado. Além
disso, a separação permite testar cada etapa isoladamente, reaproveitar a mesma construção de
pipeline nos dois modelos (agregado e individual) e revisar mudanças em diff — algo inviável
num notebook de ~2.000 linhas.

### Notebooks

| Notebook | Modelo | O que faz |
|----------|--------|-----------|
| `01_XGBoost_Alfabetizacao.py` | Agregado + individual | Versão original e monolítica. Mantida como registro do percurso analítico, incluindo o diagnóstico de data leakage |
| `02_Modelo_Agregado_Modular.py` | Município / rede | Carga → split temporal → pipeline → treino → avaliação → importâncias |
| `03_Modelo_Individual_Modular.py` | Aluno | Carga sem leakage → correção de split → features históricas → amostragem → treino → avaliação → otimização de limiar |

---

##  Como Executar

**Pré-requisitos**
- Workspace **Databricks** com as tabelas da Fase 2 disponíveis no Unity Catalog
- Python 3.8+

**Dependências**
```bash
pip install -r TechChallenge_Fase3/requirements.txt
```

**Execução dos notebooks modulares (recomendado)**
1. Importar a pasta `TechChallenge_Fase3/` no workspace do Databricks, preservando a árvore
   de diretórios — os notebooks adicionam a raiz do projeto ao `sys.path` para importar `src/`
2. Anexar a um cluster com acesso ao catálogo `workspace`
3. Executar `02_Modelo_Agregado_Modular` e, em seguida, `03_Modelo_Individual_Modular`

**Execução do notebook original**
1. Importar `01_XGBoost_Alfabetizacao.py` no workspace
2. Executar as células em ordem (`Run All`)

**Amostragem local dos dados**
```bash
python TechChallenge_Fase3/data/sample_data.py
```

---

##  Equipe

| Integrante | Contato |
|------------|---------|
| Isabelle Nicole Santana de Brito | isabelle_nicole@outlook.com |
| Filipe Noberto Justino | justinofilipe03@hotmail.com |
| Leandro Rebes Camargo | leandrorcamargo@hotmail.com |
| Felipe Vieira Sanches | fvieirasanches@gmail.com |

---

##  Vídeo Executivo

> _Link a ser adicionado (apresentação executiva de até 5 minutos)._

---
