# Tech Challenge - Fase 3
## Predição e Inteligência Analítica para Alfabetização no Brasil 🇧🇷📊

---

## 🎯 Objetivo

Desenvolver um **modelo supervisionado** capaz de prever se um aluno será considerado **alfabetizado ou não alfabetizado**, utilizando variáveis educacionais, territoriais e socioeconômicas.

Este projeto é a continuação do **TechChallenge - Fase 2**, onde construímos a pipeline de engenharia de dados (Bronze → Silver → Gold). Agora, utilizamos os dados tratados da **camada Gold** para desenvolver modelos de Machine Learning.

---

## 📁 Estrutura do Projeto

```
TechChallenge_Fase3/
├── notebooks/                    # Notebooks Jupyter
│   └── 01_XGBoost_Alfabetizacao.ipynb   # Modelo XGBoost completo
├── src/                          # Código modularizado (futuro)
├── reports/                      # Relatórios e visualizações
└── README.md                     # Este arquivo
```

---

## 📦 Dataset

### Fonte de Dados
Utilizamos os dados da **camada Silver** do projeto TechChallenge_2:
- **Tabela**: `workspace.silver.ts_aluno`
- **Registros**: ~6 milhões de alunos (2023-2025)
- **Variável Target**: `alfabetizado` (1 = alfabetizado, 0 = não alfabetizado)

### Features Utilizadas

**Numéricas:**
- `proficiencia` - Nota do aluno (escala 0-1000)
- `presenca` - Se o aluno estava presente (0/1)
- `preenchimento` - Se preencheu o teste (0/1)
- `peso_aluno` - Peso amostral
- `serie` - Série escolar (1 ou 2)
- `ano` - Ano da avaliação (2023, 2024, 2025)

**Categóricas:**
- `codigo_uf` - Código do estado
- `dependencia` - Tipo de rede (federal, estadual, municipal, privada)
- `sigla_uf` - Sigla do estado

---

## 🤖 Modelo: XGBoost (Gradient Boosting)

### Por que XGBoost?

1. ✅ **Performance Superior**: Estado da arte em problemas de classificação tabular
2. ✅ **Interpretabilidade**: Feature Importance nativo + compatibilidade com SHAP
3. ✅ **Robustez**: Trata valores faltantes e outliers internamente
4. ✅ **Eficiência**: Otimizado para datasets grandes (6M+ registros)
5. ✅ **Regularização**: Previne overfitting com L1/L2 e controle de profundidade

### Hiperparâmetros Utilizados

```python
{
    'n_estimators': 300,         # Número de árvores
    'learning_rate': 0.05,       # Taxa de aprendizado
    'max_depth': 6,              # Profundidade máxima
    'min_child_weight': 3,       # Peso mínimo em folhas
    'subsample': 0.8,            # Fração de amostras por árvore
    'colsample_bytree': 0.8,     # Fração de features por árvore
    'gamma': 0.1,                # Regularização
    'reg_alpha': 0.1,            # L1 regularization
    'reg_lambda': 1.0            # L2 regularization
}
```

---

## 🛠️ Pipeline de Machine Learning

### 1️⃣ Carregamento e Exploração
- Carga dos dados da camada Silver
- Análise exploratória (EDA)
- Distribuição do target (balanceamento)

### 2️⃣ Feature Engineering
- Seleção de features relevantes
- Identificação de tipos (numéricas vs. categóricas)
- Verificação de valores ausentes

### 3️⃣ Preprocessamento
**Features Numéricas:**
- Imputação de valores faltantes (mediana)
- Padronização (StandardScaler)

**Features Categóricas:**
- Imputação de valores faltantes (constante 'desconhecido')
- One-Hot Encoding

### 4️⃣ Split Temporal
- **Treino**: 2023 + 2024 (dados passados)
- **Teste**: 2025 (dados recentes - simula predição futura)
- Previne data leakage!

### 5️⃣ Modelagem
- Pipeline Scikit-learn (preprocessamento + modelo)
- Treinamento do XGBoost
- Proteção contra data leakage (fit apenas no treino)

### 6️⃣ Avaliação
- **Métricas**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Matriz de Confusão**: Análise de erros (FP, FN, TP, TN)
- **Overfitting Check**: Comparação treino vs. teste

### 7️⃣ Interpretabilidade
- **Feature Importance**: Identifica features mais relevantes
- **Insights Estratégicos**: Fatores que impactam alfabetização

---

## 📊 Métricas de Avaliação

### Métricas Principais
- **Accuracy**: Taxa de acerto geral
- **Precision**: Proporção de positivos corretos
- **Recall**: Proporção de positivos capturados
- **F1-Score**: Média harmônica de Precision e Recall
- **AUC-ROC**: Área sob a curva ROC (discriminação)

### Por que essas métricas?

Para o contexto de **alfabetização infantil**:
- **Recall é crítico**: Identificar alunos em risco é mais importante que evitar falsos alarmes
- **F1-Score**: Balanceia Precision e Recall
- **AUC-ROC**: Avaliação geral da capacidade discriminativa

---

## 💡 Insights Estratégicos

### Fatores que Mais Impactam Alfabetização

1. **Proficiência** 🎯
   - A nota do aluno é o fator mais crítico
   - **Ação**: Intervenções pedagógicas focadas

2. **Localização (UF)** 🗺️
   - Disparidades regionais significativas
   - **Ação**: Políticas públicas adaptadas por região

3. **Dependência Administrativa** 🏫
   - Tipo de rede influencia alfabetização
   - **Ação**: Compartilhar boas práticas entre redes

4. **Evolução Temporal** 📈
   - Tendências ao longo dos anos
   - **Ação**: Monitoramento contínuo

---

## 🚀 Como Executar

### Pré-requisitos
- Databricks Workspace
- Acesso à tabela `workspace.silver.ts_aluno`
- Python 3.8+
- Bibliotecas: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

### Passo a Passo

1. **Abrir o notebook**:
   ```
   notebooks/01_XGBoost_Alfabetizacao.ipynb
   ```

2. **Executar as células sequencialmente**:
   - Célula 1: Importações
   - Célula 2: Carregamento dos dados
   - Célula 3: EDA
   - Célula 4: Feature Engineering
   - Célula 5: Split Temporal
   - Célula 6: Preprocessamento
   - Célula 7: Modelagem XGBoost
   - Célula 8: Avaliação
   - Célula 9: Interpretabilidade
   - Célula 10: Conclusões

3. **Analisar os resultados**:
   - Métricas de performance
   - Matriz de confusão
   - Feature Importance

---

## 📝 Próximos Passos

### Curto Prazo
- [ ] Otimização de hiperparâmetros (GridSearch/Optuna)
- [ ] Implementar SHAP Values para interpretabilidade avançada
- [ ] Feature Engineering: adicionar contexto socioeconômico

### Médio Prazo
- [ ] Comparar com LightGBM e Random Forest
- [ ] Ensemble de modelos (Voting Classifier)
- [ ] Análise de municípios em risco

### Longo Prazo
- [ ] Dashboard interativo (Streamlit/Dash)
- [ ] API REST para predições em tempo real
- [ ] Clusterização de regiões similares

---

## 📚 Referências

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Tech Challenge - Fase 2](../TechChallenge_2/README.md)
- [Indicador Criança Alfabetizada - INEP](https://www.gov.br/inep/pt-br)
- [Base dos Dados](https://basedosdados.org/)

---

## 👥 Equipe

| Integrante | Contato |
|------------|---------|
| Isabelle Nicole Santana de Brito | isabelle_nicole@outlook.com |
| Filipe Noberto Justino | justinofilipe03@hotmail.com |
| Leandro Rebes Camargo | leandrorcamargo@hotmail.com |
| Felipe Vieira Sanches | fvieirasanches@gmail.com |

---

## 🎓 Licença

Este projeto faz parte do Tech Challenge da Pós-Tech FIAP.

---

**🏆 TechChallenge - Fase 3 | 2026**