"""Módulo para carregar dados da camada Gold."""
from pyspark.sql import SparkSession, functions as F
import pandas as pd


def get_spark():
    """Obtém ou cria sessão Spark."""
    return SparkSession.builder.getOrCreate()


def load_features_ml():
    """Carrega dados agregados (município/rede) da Gold."""
    spark = get_spark()
    df = spark.table('workspace.gold.features_ml')
    print(f"Registros: {df.count():,} | Colunas: {len(df.columns)}")
    return df


def load_microdados_alunos():
    """Carrega microdados de alunos da Gold."""
    spark = get_spark()
    df = spark.table('workspace.default.microdados_alunos_gold')
    print(f"Registros: {df.count():,} | Colunas: {len(df.columns)}")
    return df


def load_indicadores_municipio():
    """Carrega indicadores municipais da Gold."""
    spark = get_spark()
    return spark.table('workspace.gold.indicadores_municipio')


def load_metas_uf():
    """Carrega metas estaduais da Gold."""
    spark = get_spark()
    return spark.table('workspace.gold.metas_vs_resultados_uf')


def create_target_variable(df, threshold=80):
    """Cria variável target binária baseada na taxa de alfabetização."""
    return df.withColumn(
        'meta_atingida', 
        F.when(F.col('taxa_alfabetizacao') >= threshold, 1).otherwise(0)
    )


def split_temporal(df, train_year=2023, test_year=2024, features=None, target='meta_atingida'):
    """Realiza split temporal dos dados."""
    df_train = df.filter(F.col('ano') == train_year).toPandas()
    df_test = df.filter(F.col('ano') == test_year).toPandas()
    
    if features is None:
        features = [c for c in df_train.columns if c != target]
    
    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]
    
    print(f"Treino: {len(X_train):,} | Teste: {len(X_test):,}")
    return X_train, X_test, y_train, y_test
