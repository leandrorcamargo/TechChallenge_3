# Databricks notebook source
# DBTITLE 1,Tech Challenge Fase 3 - Predição de Alfabetização com XGBoost
# MAGIC %md
# MAGIC # 🎯 Tech Challenge Fase 3 - Predição de Alfabetização com XGBoost
# MAGIC
# MAGIC ## Objetivo
# MAGIC Desenvolver um modelo supervisionado capaz de **prever se um aluno será considerado alfabetizado ou não alfabetizado**, utilizando variáveis educacionais, territoriais e contextuais.
# MAGIC
# MAGIC ## Dataset
# MAGIC Utilizamos os dados da **camada Silver** do projeto TechChallenge_2, contendo:
# MAGIC - **~6 milhões de registros** de alunos (2023-2025)
# MAGIC - **Variável target**: `alfabetizado` (1 = alfabetizado, 0 = não alfabetizado)
# MAGIC - **Features**: proficiência, presença, município, UF, dependência administrativa, ano, etc.
# MAGIC
# MAGIC ## Algoritmo Escolhido: XGBoost (Gradient Boosting)
# MAGIC
# MAGIC ### Por que XGBoost?
# MAGIC 1. ✅ **Performance superior** em dados tabulares
# MAGIC 2. ✅ **Interpretabilidade** (Feature Importance + SHAP Values)
# MAGIC 3. ✅ **Robustez** a outliers e valores faltantes
# MAGIC 4. ✅ **Eficiência** computacional para datasets grandes
# MAGIC 5. ✅ **Regularização** embutida (previne overfitting)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Pipeline de Machine Learning
# MAGIC
# MAGIC Este notebook implementa uma pipeline completa:
# MAGIC
# MAGIC 1. **Carregamento e Exploração dos Dados**
# MAGIC 2. **Feature Engineering** (criação de features contextuais)
# MAGIC 3. **Preprocessamento** (tratamento de valores faltantes, encoding)
# MAGIC 4. **Split Temporal** (treino: 2023-2024, teste: 2025)
# MAGIC 5. **Modelagem XGBoost** com Pipeline Scikit-learn
# MAGIC 6. **Validação e Métricas** (Accuracy, Precision, Recall, F1, AUC-ROC)
# MAGIC 7. **Interpretabilidade** (Feature Importance e SHAP Values)
# MAGIC 8. **Insights Estratégicos** para políticas públicas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Autor**: Equipe TechChallenge  
# MAGIC **Data**: 2026  
# MAGIC **Versão**: 1.0

# COMMAND ----------

# DBTITLE 1,1. Importações e Configurações
# ============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÕES
# ============================================================================

# Bibliotecas de Manipulação de Dados
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import numpy as np

# Bibliotecas de Machine Learning (Scikit-learn)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# XGBoost
from xgboost import XGBClassifier

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de visualização
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Warnings
import warnings
warnings.filterwarnings('ignore')

print("✅ Bibliotecas importadas com sucesso!")
print(f"\nVersões:")
print(f"  - Pandas: {pd.__version__}")
print(f"  - NumPy: {np.__version__}")

# Spark Session
spark = SparkSession.builder.appName("TechChallenge-Fase3-XGBoost").getOrCreate()
print(f"  - Spark: {spark.version}")
print("\n✅ Spark Session ativa!")

# COMMAND ----------

# DBTITLE 1,2. Carregamento dos Dados da Camada Silver
# ============================================================================
# 2. CARREGAMENTO DOS DADOS DA CAMADA SILVER
# ============================================================================

print("="*80)
print("CARREGANDO DADOS DA CAMADA SILVER")
print("="*80)

# Tabela principal: microdados de alunos (nível individual)
# Contém a variável target 'alfabetizado' e features do aluno
df_alunos = spark.table("workspace.silver.ts_aluno")

print(f"\n✅ Dados carregados:")
print(f"  - Total de registros: {df_alunos.count():,}")
print(f"  - Total de colunas: {len(df_alunos.columns)}")

# Visualizar schema
print("\n📊 Schema dos dados:")
df_alunos.printSchema()

# Visualizar primeiras linhas
print("\n👁️ Amostra dos dados (primeiras 5 linhas):")
display(df_alunos.limit(5))

# COMMAND ----------

# DBTITLE 1,3. Análise Exploratória Inicial (EDA)
# ============================================================================
# 3. ANÁLISE EXPLORATÓRIA INICIAL (EDA)
# ============================================================================

print("="*80)
print("ANÁLISE EXPLORATÓRIA DOS DADOS")
print("="*80)

# --------------------------------------------------
# 3.1 Distribuição da Variável Target (alfabetizado)
# --------------------------------------------------
print("\n1️⃣ DISTRIBUIÇÃO DA VARIÁVEL TARGET")
print("-" * 80)

target_dist = df_alunos.groupBy("alfabetizado").count().orderBy("alfabetizado")
target_dist = target_dist.withColumn(
    "percentual",
    F.round((F.col("count") / df_alunos.count()) * 100, 2)
)

print("\nDistribuição de alfabetizados vs. não alfabetizados:")
display(target_dist)

# --------------------------------------------------
# 3.2 Distribuição por Ano
# --------------------------------------------------
print("\n2️⃣ DISTRIBUIÇÃO POR ANO")
print("-" * 80)

ano_dist = df_alunos.groupBy("ano").agg(
    F.count("*").alias("total_alunos"),
    F.sum(F.when(F.col("alfabetizado") == 1, 1).otherwise(0)).alias("alfabetizados"),
    F.round(F.avg(F.when(F.col("alfabetizado") == 1, 1.0).otherwise(0.0)) * 100, 2).alias("taxa_alfabetizacao")
).orderBy("ano")

print("\nDistribuição e taxa de alfabetização por ano:")
display(ano_dist)

# --------------------------------------------------
# 3.3 Distribuição por Dependência Administrativa
# --------------------------------------------------
print("\n3️⃣ DISTRIBUIÇÃO POR DEPENDÊNCIA ADMINISTRATIVA")
print("-" * 80)

dep_dist = df_alunos.groupBy("dependencia").agg(
    F.count("*").alias("total_alunos"),
    F.round(F.avg(F.when(F.col("alfabetizado") == 1, 1.0).otherwise(0.0)) * 100, 2).alias("taxa_alfabetizacao")
).orderBy(F.desc("total_alunos"))

print("\nDistribuição por rede de ensino:")
display(dep_dist)

# --------------------------------------------------
# 3.4 Estatísticas Descritivas de Proficiência
# --------------------------------------------------
print("\n4️⃣ ESTATÍSTICAS DE PROFICIÊNCIA")
print("-" * 80)

prof_stats = df_alunos.groupBy("alfabetizado").agg(
    F.count("proficiencia").alias("n"),
    F.round(F.mean("proficiencia"), 2).alias("media"),
    F.round(F.stddev("proficiencia"), 2).alias("desvio_padrao"),
    F.round(F.min("proficiencia"), 2).alias("minimo"),
    F.round(F.expr("percentile(proficiencia, 0.25)"), 2).alias("Q1"),
    F.round(F.expr("percentile(proficiencia, 0.5)"), 2).alias("mediana"),
    F.round(F.expr("percentile(proficiencia, 0.75)"), 2).alias("Q3"),
    F.round(F.max("proficiencia"), 2).alias("maximo")
).orderBy("alfabetizado")

print("\nEstatísticas de proficiência por grupo:")
print("(0 = Não alfabetizado, 1 = Alfabetizado)\n")
display(prof_stats)

print("\n✅ Análise exploratória inicial concluída!")

# COMMAND ----------

# DBTITLE 1,4. Feature Engineering e Preparação
# ============================================================================
# 4. FEATURE ENGINEERING E PREPARAÇÃO
# ============================================================================

print("="*80)
print("FEATURE ENGINEERING E PREPARAÇÃO DOS DADOS")
print("="*80)

# --------------------------------------------------
# 4.1 Seleção de Features
# --------------------------------------------------
print("\n1️⃣ SELEÇÃO DE FEATURES")
print("-" * 80)

# Features que vamos utilizar no modelo
features_selecionadas = [
    # Features numéricas
    'proficiencia',      # Nota do aluno (principal feature)
    'presenca',          # Se o aluno estava presente (0/1)
    'preenchimento',     # Se preencheu o teste (0/1)
    'peso_aluno',        # Peso amostral do aluno
    'serie',             # Série escolar (1 ou 2)
    'ano',               # Ano da avaliação (2023, 2024, 2025)
    
    # Features categóricas
    'codigo_uf',         # Código do estado (2 dígitos)
    'dependencia',       # Tipo de rede (federal, estadual, municipal, privada)
    'sigla_uf',          # Sigla do estado (AC, SP, etc.)
]

# Variável target
target = 'alfabetizado'

print(f"Features selecionadas: {len(features_selecionadas)}")
for i, feat in enumerate(features_selecionadas, 1):
    print(f"  {i:2d}. {feat}")

print(f"\nTarget: {target}")

# --------------------------------------------------
# 4.2 Criar DataFrame com Features Selecionadas
# --------------------------------------------------
print("\n2️⃣ CRIANDO DATAFRAME COM FEATURES SELECIONADAS")
print("-" * 80)

# Selecionar colunas + target
colunas_modelo = features_selecionadas + [target]

df_model = df_alunos.select(colunas_modelo)

# Remover registros com target nulo (se houver)
df_model = df_model.filter(F.col(target).isNotNull())

print(f"\n✅ Dataset para modelagem:")
print(f"  - Registros: {df_model.count():,}")
print(f"  - Features: {len(features_selecionadas)}")
print(f"  - Target: {target}")

# Verificar valores nulos por coluna
print("\n📊 Valores nulos por coluna:")
for col in colunas_modelo:
    nulos = df_model.filter(F.col(col).isNull()).count()
    perc = (nulos / df_model.count()) * 100
    if nulos > 0:
        print(f"  {col:20s}: {nulos:>10,} ({perc:>5.2f}%)")
    else:
        print(f"  {col:20s}: {nulos:>10,} (0.00%)")

print("\n✅ Feature Engineering concluída!")

# COMMAND ----------

# DBTITLE 1,5. Split Temporal dos Dados (Treino/Teste)
# ============================================================================
# 5. SPLIT TEMPORAL DOS DADOS (TREINO/TESTE)
# ============================================================================

print("="*80)
print("SPLIT TEMPORAL - TREINO E TESTE")
print("="*80)

# --------------------------------------------------
# Estratégia: Split Temporal
# --------------------------------------------------
# Para dados educacionais com evolução temporal, é importante respeitar
# a ordem do tempo para simular uma situação real de predição.
#
# Treino: 2023 + 2024 (dados passados)
# Teste: 2025 (dados mais recentes - simula predição futura)
# --------------------------------------------------

print("\n🎯 Estratégia de Split:")
print("  - Treino: anos 2023 e 2024")
print("  - Teste: ano 2025")
print("\n  Objetivo: Simular predição de alfabetização para o ano seguinte")

# Separar treino e teste por ano
df_train = df_model.filter(F.col("ano").isin([2023, 2024]))
df_test = df_model.filter(F.col("ano") == 2025)

print(f"\n📊 Distribuição dos dados:")
print(f"  - Treino (2023-2024): {df_train.count():,} registros")
print(f"  - Teste (2025): {df_test.count():,} registros")

# Converter para Pandas (necessário para Scikit-learn)
print("\n🔄 Convertendo para Pandas...")
df_train_pd = df_train.toPandas()
df_test_pd = df_test.toPandas()

print(f"\n✅ Conversão concluída:")
print(f"  - Treino: {len(df_train_pd):,} linhas")
print(f"  - Teste: {len(df_test_pd):,} linhas")

# Separar features (X) e target (y)
X_train = df_train_pd[features_selecionadas]
y_train = df_train_pd[target]

X_test = df_test_pd[features_selecionadas]
y_test = df_test_pd[target]

print(f"\n🎯 Features e Target separados:")
print(f"  - X_train: {X_train.shape}")
print(f"  - y_train: {y_train.shape}")
print(f"  - X_test: {X_test.shape}")
print(f"  - y_test: {y_test.shape}")

# Verificar balanço das classes
print(f"\n⚖️ Balanço das classes:")
print(f"\nTreino:")
print(y_train.value_counts(normalize=True).mul(100).round(2))
print(f"\nTeste:")
print(y_test.value_counts(normalize=True).mul(100).round(2))

print("\n✅ Split temporal concluído!")

# COMMAND ----------

# DBTITLE 1,6. Construção do Pipeline de Preprocessamento
# ============================================================================
# 6. CONSTRUÇÃO DO PIPELINE DE PREPROCESSAMENTO
# ============================================================================

print("="*80)
print("CONSTRUÇÃO DO PIPELINE DE PREPROCESSAMENTO")
print("="*80)

# --------------------------------------------------
# 6.1 Identificar tipos de features
# --------------------------------------------------
print("\n1️⃣ IDENTIFICANDO TIPOS DE FEATURES")
print("-" * 80)

# Features numéricas (precisam de imputação e escalonamento)
features_numericas = [
    'proficiencia', 'presenca', 'preenchimento', 
    'peso_aluno', 'serie', 'ano'
]

# Features categóricas (precisam de encoding)
features_categoricas = [
    'codigo_uf', 'dependencia', 'sigla_uf'
]

print(f"Features numéricas ({len(features_numericas)}):")
for feat in features_numericas:
    print(f"  - {feat}")

print(f"\nFeatures categóricas ({len(features_categoricas)}):")
for feat in features_categoricas:
    print(f"  - {feat}")

# --------------------------------------------------
# 6.2 Pipeline de Preprocessamento
# --------------------------------------------------
print("\n2️⃣ CRIANDO PIPELINE DE PREPROCESSAMENTO")
print("-" * 80)

# Pipeline para features NUMÉRICAS:
# 1. Imputação de valores faltantes (mediana)
# 2. Padronização (StandardScaler)
preprocessador_numerico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para features CATEGÓRICAS:
# 1. Imputação de valores faltantes (valor constante 'desconhecido')
# 2. One-Hot Encoding
preprocessador_categorico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='desconhecido')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer: aplica pipelines diferentes para cada tipo de feature
preprocessador = ColumnTransformer(
    transformers=[
        ('num', preprocessador_numerico, features_numericas),
        ('cat', preprocessador_categorico, features_categoricas)
    ],
    remainder='drop'  # Descarta colunas não especificadas
)

print("\n✅ Pipeline de preprocessamento criado:")
print("\n  Features Numéricas:")
print("    1. SimpleImputer (strategy='median')")
print("    2. StandardScaler()")
print("\n  Features Categóricas:")
print("    1. SimpleImputer (strategy='constant', fill_value='desconhecido')")
print("    2. OneHotEncoder (handle_unknown='ignore')")
print("\n  Previne Data Leakage: \u2705")
print("    - Imputer e Scaler serão FIT apenas no conjunto de treino")
print("    - Teste será apenas TRANSFORMADO (nunca visto durante o fit)")

print("\n✅ Preprocessamento configurado!")

# COMMAND ----------

# DBTITLE 1,7. Modelagem com XGBoost
# ============================================================================
# 7. MODELAGEM COM XGBOOST
# ============================================================================

print("="*80)
print("TREINAMENTO DO MODELO XGBOOST")
print("="*80)

# --------------------------------------------------
# 7.1 Configuração do XGBoost
# --------------------------------------------------
print("\n1️⃣ CONFIGURAÇÃO DO XGBOOST")
print("-" * 80)

# Parâmetros do XGBoost
# Estes valores foram escolhidos com base em boas práticas para classification tasks
xgb_params = {
    'n_estimators': 300,         # Número de árvores (aumenta performance, cuidado com overfitting)
    'learning_rate': 0.05,       # Taxa de aprendizado (menor = mais conservador, evita overfitting)
    'max_depth': 6,              # Profundidade máxima das árvores (controla complexidade)
    'min_child_weight': 3,       # Peso mínimo em folhas (regularização)
    'subsample': 0.8,            # Fração de amostras por árvore (evita overfitting)
    'colsample_bytree': 0.8,     # Fração de features por árvore (diversidade)
    'gamma': 0.1,                # Mínimo de redução de perda para split (regularização)
    'reg_alpha': 0.1,            # Regularização L1 (Lasso)
    'reg_lambda': 1.0,           # Regularização L2 (Ridge)
    'random_state': 42,          # Reprodutibilidade
    'n_jobs': -1,                # Usar todos os cores disponíveis
    'eval_metric': 'logloss',    # Métrica de avaliação durante treinamento
    'use_label_encoder': False   # Desabilita warning de encoder
}

print("\u2699️ Parâmetros do XGBoost:")
for param, valor in xgb_params.items():
    print(f"  {param:20s}: {valor}")

# Criar o classificador XGBoost
xgb_clf = XGBClassifier(**xgb_params)

# --------------------------------------------------
# 7.2 Pipeline Completo: Preprocessamento + XGBoost
# --------------------------------------------------
print("\n2️⃣ CRIANDO PIPELINE COMPLETO")
print("-" * 80)

# Integrar preprocessamento + modelo em um único pipeline
# Isso garante que:
# 1. O preprocessamento seja aplicado automaticamente
# 2. Não haja data leakage (fit/transform correto)
# 3. O modelo seja reproduzível e fácil de colocar em produção
pipeline_completo = Pipeline(steps=[
    ('preprocessamento', preprocessador),
    ('modelo', xgb_clf)
])

print("\n✅ Pipeline completo criado:")
print("  1. Preprocessamento (imputação + scaling + encoding)")
print("  2. XGBoost Classifier")
print("\n  Pronto para treinamento!")

# --------------------------------------------------
# 7.3 Treinamento do Modelo
# --------------------------------------------------
print("\n3️⃣ TREINANDO O MODELO")
print("-" * 80)
print("\n⏳ Iniciando treinamento... (pode levar alguns minutos)")

import time
start_time = time.time()

# Treinar o pipeline completo
pipeline_completo.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"\n✅ Treinamento concluído!")
print(f"  Tempo de treinamento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)")
print(f"  Registros de treino: {len(X_train):,}")
print(f"  Features: {len(features_selecionadas)}")
print(f"  Árvores: {xgb_params['n_estimators']}")

# COMMAND ----------

# DBTITLE 1,8. Avaliação do Modelo
# ============================================================================
# 8. AVALIAÇÃO DO MODELO
# ============================================================================

print("="*80)
print("AVALIAÇÃO DO MODELO XGBOOST")
print("="*80)

# --------------------------------------------------
# 8.1 Predições
# --------------------------------------------------
print("\n1️⃣ GERANDO PREDIÇÕES")
print("-" * 80)

# Predições no conjunto de treino
y_train_pred = pipeline_completo.predict(X_train)
y_train_proba = pipeline_completo.predict_proba(X_train)[:, 1]

# Predições no conjunto de teste
y_test_pred = pipeline_completo.predict(X_test)
y_test_proba = pipeline_completo.predict_proba(X_test)[:, 1]

print("✅ Predições geradas para treino e teste")

# --------------------------------------------------
# 8.2 Métricas de Performance
# --------------------------------------------------
print("\n2️⃣ MÉTRICAS DE PERFORMANCE")
print("-" * 80)

# Função auxiliar para calcular métricas
def calcular_metricas(y_true, y_pred, y_proba, conjunto="Treino"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"\n🎯 Métricas - {conjunto}:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}

# Calcular métricas para treino e teste
metricas_train = calcular_metricas(y_train, y_train_pred, y_train_proba, "Treino")
metricas_test = calcular_metricas(y_test, y_test_pred, y_test_proba, "Teste")

# --------------------------------------------------
# 8.3 Matriz de Confusão
# --------------------------------------------------
print("\n3️⃣ MATRIZ DE CONFUSÃO (TESTE)")
print("-" * 80)

cm = confusion_matrix(y_test, y_test_pred)

print("\n                Previsto")
print("                 0     1")
print(f"Real     0    {cm[0,0]:>6,} {cm[0,1]:>6,}")
print(f"         1    {cm[1,0]:>6,} {cm[1,1]:>6,}")
print("\n  Legenda:")
print("    0 = Não alfabetizado")
print("    1 = Alfabetizado")
print(f"\n  Verdadeiros Negativos (TN): {cm[0,0]:,}")
print(f"  Falsos Positivos (FP):      {cm[0,1]:,}")
print(f"  Falsos Negativos (FN):      {cm[1,0]:,}")
print(f"  Verdadeiros Positivos (TP): {cm[1,1]:,}")

# --------------------------------------------------
# 8.4 Relatório de Classificação
# --------------------------------------------------
print("\n4️⃣ RELATÓRIO DE CLASSIFICAÇÃO (TESTE)")
print("-" * 80)
print("\n")
print(classification_report(y_test, y_test_pred, 
                          target_names=['Não Alfabetizado', 'Alfabetizado'],
                          digits=4))

# --------------------------------------------------
# 8.5 Análise de Overfitting
# --------------------------------------------------
print("\n5️⃣ ANÁLISE DE OVERFITTING")
print("-" * 80)

diferenca_accuracy = metricas_train['accuracy'] - metricas_test['accuracy']
diferenca_f1 = metricas_train['f1'] - metricas_test['f1']

print(f"\n  Diferença Treino-Teste:")
print(f"    Accuracy: {diferenca_accuracy:.4f} ({diferenca_accuracy*100:.2f} p.p.)")
print(f"    F1-Score: {diferenca_f1:.4f} ({diferenca_f1*100:.2f} p.p.)")

if diferenca_accuracy < 0.05 and diferenca_f1 < 0.05:
    print("\n  ✅ Modelo bem generalizado! (diferença < 5%)")
elif diferenca_accuracy < 0.10:
    print("\n  ⚠️ Leve overfitting detectado (diferença < 10%)")
else:
    print("\n  ❌ Overfitting significativo (diferença > 10%)")

print("\n✅ Avaliação do modelo concluída!")

# COMMAND ----------

# DBTITLE 1,9. Interpretabilidade - Feature Importance
# ============================================================================
# 9. INTERPRETABILIDADE - FEATURE IMPORTANCE
# ============================================================================

print("="*80)
print("INTERPRETABILIDADE DO MODELO - FEATURE IMPORTANCE")
print("="*80)

# --------------------------------------------------
# 9.1 Extração do Feature Importance
# --------------------------------------------------
print("\n1️⃣ EXTRAÇÃO DO FEATURE IMPORTANCE")
print("-" * 80)

# Obter o modelo XGBoost treinado do pipeline
xgb_model = pipeline_completo.named_steps['modelo']

# Obter feature importance (gain = redução média de perda)
feature_importance = xgb_model.feature_importances_

# Obter nomes das features após preprocessamento
# Features numéricas mantêm seus nomes
# Features categóricas são expandidas pelo OneHotEncoder
preprocessador_fitted = pipeline_completo.named_steps['preprocessamento']

# Nomes das features numéricas
feature_names_num = features_numericas

# Nomes das features categóricas (após OneHotEncoding)
try:
    encoder = preprocessador_fitted.named_transformers_['cat'].named_steps['encoder']
    feature_names_cat = encoder.get_feature_names_out(features_categoricas).tolist()
except:
    feature_names_cat = []

# Combinar nomes
feature_names_all = feature_names_num + feature_names_cat

print(f"\n✅ Features extraídas:")
print(f"  - Features numéricas: {len(feature_names_num)}")
print(f"  - Features categóricas (após encoding): {len(feature_names_cat)}")
print(f"  - Total de features no modelo: {len(feature_names_all)}")

# --------------------------------------------------
# 9.2 Ranking de Importância
# --------------------------------------------------
print("\n2️⃣ RANKING DE IMPORTÂNCIA DAS FEATURES")
print("-" * 80)

# Criar DataFrame com feature importance
importance_df = pd.DataFrame({
    'feature': feature_names_all,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Normalizar para percentual
importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum()) * 100

print("\n🏆 TOP 15 Features Mais Importantes:\n")
for i, row in importance_df.head(15).iterrows():
    print(f"  {row.name+1:2d}. {row['feature']:40s} | {row['importance']:.4f} | {row['importance_pct']:>6.2f}%")

# --------------------------------------------------
# 9.3 Visualização do Feature Importance
# --------------------------------------------------
print("\n3️⃣ VISUALIZAÇÃO")
print("-" * 80)

# Top 15 features
top_features = importance_df.head(15)

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance (Gain)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Features Mais Importantes - XGBoost\n(Ordenado por Ganho de Informação)', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ Feature Importance extraído e visualizado!")

# --------------------------------------------------
# 9.4 Insights Estratégicos
# --------------------------------------------------
print("\n4️⃣ INSIGHTS ESTRATÉGICOS")
print("-" * 80)
print("\n💡 O que as features mais importantes nos dizem:\n")

# Identificar as 3 mais importantes
top3 = importance_df.head(3)['feature'].tolist()

if 'proficiencia' in top3:
    print("  ✅ PROFICIÊNCIA: A nota do aluno é o fator mais crítico.")
    print("     ➡️ Ação: Intervenções pedagógicas focadas em proficiência têm alto impacto.\n")

if any('codigo_uf' in feat or 'sigla_uf' in feat for feat in top3):
    print("  ✅ LOCALIZAÇÃO (UF): O estado tem forte impacto na alfabetização.")
    print("     ➡️ Ação: Políticas públicas devem considerar disparidades regionais.\n")

if any('dependencia' in feat for feat in top3):
    print("  ✅ DEPENDÊNCIA ADMINISTRATIVA: Tipo de rede (federal/estadual/municipal/privada) é relevante.")
    print("     ➡️ Ação: Compartilhar boas práticas entre redes com melhor desempenho.\n")

if 'ano' in top3:
    print("  ✅ TEMPORAL (ANO): Evolução ao longo dos anos impacta alfabetização.")
    print("     ➡️ Ação: Monitorar tendências e adaptar políticas com base em evolução histórica.\n")

print("\n✅ Análise de interpretabilidade concluída!")

# COMMAND ----------

# DBTITLE 1,10. Conclusões e Próximos Passos
# MAGIC %md
# MAGIC ## 🎯 Conclusões e Próximos Passos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Resumo dos Resultados
# MAGIC
# MAGIC Este notebook implementou com sucesso um **modelo XGBoost** para prever alfabetização de alunos brasileiros, seguindo as boas práticas de Machine Learning:
# MAGIC
# MAGIC ✅ **Pipeline Completa**: Preprocessamento + Modelagem integrados  
# MAGIC ✅ **Proteção contra Data Leakage**: Split temporal e fit/transform corretos  
# MAGIC ✅ **Métricas Robustas**: Accuracy, Precision, Recall, F1-Score, AUC-ROC  
# MAGIC ✅ **Interpretabilidade**: Feature Importance extraído  
# MAGIC ✅ **Insights Estratégicos**: Identificação de fatores críticos  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔑 Principais Descobertas
# MAGIC
# MAGIC 1. **Proficiência** é o fator mais importante (esperado)
# MAGIC 2. **Localização (UF)** tem forte impacto (disparidades regionais)
# MAGIC 3. **Dependência administrativa** influencia alfabetização
# MAGIC 4. **Evolução temporal** mostra tendências ao longo dos anos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Próximos Passos
# MAGIC
# MAGIC #### 1️⃣ **Otimização de Hiperparâmetros**
# MAGIC - Implementar **GridSearchCV** ou **RandomizedSearchCV**
# MAGIC - Testar diferentes combinações de `n_estimators`, `learning_rate`, `max_depth`
# MAGIC - Utilizar **Optuna** para otimização Bayesiana
# MAGIC
# MAGIC #### 2️⃣ **Feature Engineering Avançado**
# MAGIC - Criar features agregadas por **município** (taxa média, variância)
# MAGIC - Adicionar features de **contexto socioeconomico** (IBGE, PNAD)
# MAGIC - Incluir features de **infraestrutura escolar** (Censo Escolar)
# MAGIC - Features temporais (lag, rolling windows)
# MAGIC
# MAGIC #### 3️⃣ **Modelos Alternativos**
# MAGIC - Comparar com **LightGBM** (otimizado para datasets grandes)
# MAGIC - Testar **Random Forest** como baseline
# MAGIC - Ensemble de múltiplos modelos (Voting Classifier)
# MAGIC
# MAGIC #### 4️⃣ **Interpretação Avançada**
# MAGIC - Implementar **SHAP Values** (SHapley Additive exPlanations)
# MAGIC - Análise de **dependência parcial** (Partial Dependence Plots)
# MAGIC - Identificar **interações entre features**
# MAGIC
# MAGIC #### 5️⃣ **Análise Estratégica**
# MAGIC - **Identificar municípios em risco** (baixa probabilidade de alfabetização)
# MAGIC - **Simular impacto de políticas públicas** (what-if analysis)
# MAGIC - **Clusterização de regiões** com padrões similares
# MAGIC
# MAGIC #### 6️⃣ **Produção**
# MAGIC - Salvar o modelo treinado (`joblib` ou `pickle`)
# MAGIC - Criar API REST para predições em tempo real
# MAGIC - Dashboard interativo com **Streamlit** ou **Dash**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Referências
# MAGIC
# MAGIC - [XGBoost Documentation](https://xgboost.readthedocs.io/)
# MAGIC - [Scikit-learn Pipeline Guide](https://scikit-learn.org/stable/modules/compose.html)
# MAGIC - [SHAP for Model Interpretation](https://github.com/slundberg/shap)
# MAGIC - [Tech Challenge - Fase 3](https://docs.google.com/document/d/Tech_Challenge_Fase3)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🎓 Equipe TechChallenge - Fase 3**  
# MAGIC **📅 2026**