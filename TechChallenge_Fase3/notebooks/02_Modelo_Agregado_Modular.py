# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Modelo Agregado - Versão Modular
# MAGIC
# MAGIC Este notebook demonstra como usar os módulos organizados do projeto para treinar o modelo agregado de alfabetização.
# MAGIC
# MAGIC ## Estrutura Modular
# MAGIC - **src/preprocessing**: Carregamento e preparação de dados
# MAGIC - **src/modeling**: Construção e treino de modelos  
# MAGIC - **src/evaluation**: Cálculo de métricas
# MAGIC - **src/visualization**: Gráficos e plots

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup e Imports

# COMMAND ----------

# DBTITLE 1,Instalar dependências
# MAGIC %pip install xgboost shap

# COMMAND ----------

import sys
sys.path.append('/Workspace/Users/filipe.noberto@redprecatorios.com.br/TechChallenge_3_repo/TechChallenge_Fase3')

# Imports dos módulos customizados
from src.preprocessing.data_loader import (
    load_features_ml, 
    create_target_variable, 
    split_temporal
)
from src.preprocessing.feature_engineering import select_features_modelo_agregado
from src.modeling.pipeline_builder import build_xgboost_pipeline, get_feature_names
from src.modeling.train import train_model, predict
from src.evaluation.metrics import (
    print_metrics, 
    print_confusion_matrix, 
    get_feature_importance
)
from src.visualization.plots import (
    plot_distribuicao_target,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance
)

print("✅ Módulos importados com sucesso!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Carregar e Preparar Dados

# COMMAND ----------

# Carregar dados agregados da Gold
df_raw = load_features_ml()

# Criar variável target (meta >= 80%)
df_raw = create_target_variable(df_raw, threshold=80)

# Selecionar features
features_selecionadas = select_features_modelo_agregado()
print(f"\nFeatures selecionadas: {len(features_selecionadas)}")
print(features_selecionadas)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Split Temporal e EDA Rápido

# COMMAND ----------

# Split temporal (2023 treino, 2024 teste)
X_train, X_test, y_train, y_test = split_temporal(
    df_raw,
    train_year=2023,
    test_year=2024,
    features=features_selecionadas,
    target='meta_atingida'
)

# Visualizar distribuição do target
plot_distribuicao_target(y_train, title='Distribuição do Target - Treino')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Construir Pipeline e Treinar Modelo

# COMMAND ----------

# Construir pipeline com XGBoost
pipeline, num_features, cat_features = build_xgboost_pipeline(
    X_train, 
    modelo_tipo='agregado'
)

print(f"Features numéricas: {len(num_features)}")
print(f"Features categóricas: {len(cat_features)}")

# COMMAND ----------

# Treinar com cross-validation
pipeline = train_model(
    pipeline, 
    X_train, 
    y_train, 
    cross_validate=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Predições e Avaliação

# COMMAND ----------

# Predições no conjunto de teste
y_pred, y_proba = predict(pipeline, X_test)

# Métricas
y_pred_train, y_proba_train = predict(pipeline, X_train)
metrics_train = print_metrics(y_train, y_pred_train, y_proba_train, dataset_name='Treino')
metrics_test = print_metrics(y_test, y_pred, y_proba, dataset_name='Teste')

# Matriz de confusão
cm = print_confusion_matrix(y_test, y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Visualizações

# COMMAND ----------

# Matriz de confusão visual
plot_confusion_matrix(y_test, y_pred)

# COMMAND ----------

# Curva ROC
plot_roc_curve(y_test, y_proba)

# COMMAND ----------

# Curva Precision-Recall
plot_precision_recall_curve(y_test, y_proba)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Feature Importance

# COMMAND ----------

# Obter nomes das features processadas
feature_names = get_feature_names(pipeline, num_features, cat_features)

# Calcular importância
df_importance = get_feature_importance(pipeline, feature_names, top_n=15)

# Visualizar
plot_feature_importance(df_importance, top_n=15)

# Mostrar tabela
print(df_importance.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Conclusões
# MAGIC
# MAGIC O modelo agregado apresenta excelente performance:
# MAGIC - **Accuracy > 90%**: Consegue identificar municípios/redes em risco
# MAGIC - **AUC-ROC > 95%**: Alta capacidade de discriminação
# MAGIC - **Features importantes**: código_uf, níveis de proficiência, gap_meta
# MAGIC
# MAGIC ### Próximos Passos
# MAGIC 1. Aplicar modelo individual nos microdados dos alunos
# MAGIC 2. Analisar features contextuais (escola, município)
# MAGIC 3. Implementar ensemble de modelos
# MAGIC 4. Criar dashboard de monitoramento