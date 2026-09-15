"""
Script para amostrar dados das bases do projeto.

Como não temos arquivos locais e sim tabelas em banco de dados Unity Catalog,
este script extrai amostras representativas das tabelas Gold para análise exploratória.
"""

from pyspark.sql import SparkSession
import pandas as pd


def get_spark():
    """Inicializa sessão Spark."""
    return SparkSession.builder.getOrCreate()


def sample_features_ml(fraction=0.1, seed=42):
    """
    Amostra da tabela workspace.gold.features_ml (modelo agregado).
    
    Args:
        fraction: Fração de amostragem (default 10%)
        seed: Seed para reprodutibilidade
    
    Returns:
        DataFrame pandas com amostra
    """
    spark = get_spark()
    df = spark.table('workspace.gold.features_ml')
    
    print(f"📊 Tabela: workspace.gold.features_ml")
    print(f"   Registros totais: {df.count():,}")
    print(f"   Colunas: {len(df.columns)}")
    
    df_sample = df.sample(fraction=fraction, seed=seed)
    df_pd = df_sample.toPandas()
    
    print(f"   Amostra: {len(df_pd):,} registros ({fraction*100}%)\n")
    
    return df_pd


def sample_microdados_alunos(fraction=0.01, seed=42):
    """
    Amostra da tabela workspace.default.microdados_alunos_gold (modelo individual).
    
    Args:
        fraction: Fração de amostragem (default 1%)
        seed: Seed para reprodutibilidade
    
    Returns:
        DataFrame pandas com amostra
    """
    spark = get_spark()
    df = spark.table('workspace.default.microdados_alunos_gold')
    
    print(f"📊 Tabela: workspace.default.microdados_alunos_gold")
    print(f"   Registros totais: {df.count():,}")
    print(f"   Colunas: {len(df.columns)}")
    
    # Amostra estratificada por alfabetizado
    df_sample = df.sampleBy('alfabetizado', fractions={0: fraction, 1: fraction}, seed=seed)
    df_pd = df_sample.toPandas()
    
    print(f"   Amostra estratificada: {len(df_pd):,} registros ({fraction*100}%)\n")
    
    return df_pd


def sample_indicadores_municipio(fraction=0.2, seed=42):
    """
    Amostra da tabela workspace.gold.indicadores_municipio.
    
    Args:
        fraction: Fração de amostragem (default 20%)
        seed: Seed para reprodutibilidade
    
    Returns:
        DataFrame pandas com amostra
    """
    spark = get_spark()
    df = spark.table('workspace.gold.indicadores_municipio')
    
    print(f"📊 Tabela: workspace.gold.indicadores_municipio")
    print(f"   Registros totais: {df.count():,}")
    print(f"   Colunas: {len(df.columns)}")
    
    df_sample = df.sample(fraction=fraction, seed=seed)
    df_pd = df_sample.toPandas()
    
    print(f"   Amostra: {len(df_pd):,} registros ({fraction*100}%)\n")
    
    return df_pd


def sample_metas_uf():
    """
    Carrega todas as metas por UF (tabela pequena).
    
    Returns:
        DataFrame pandas
    """
    spark = get_spark()
    df = spark.table('workspace.gold.metas_vs_resultados_uf')
    
    print(f"📊 Tabela: workspace.gold.metas_vs_resultados_uf")
    print(f"   Registros totais: {df.count():,}")
    print(f"   Colunas: {len(df.columns)}")
    
    df_pd = df.toPandas()
    
    print(f"   Carregando todos os registros (tabela pequena)\n")
    
    return df_pd


def export_samples_to_csv(output_dir='./'):
    """
    Exporta amostras de todas as tabelas para CSV.
    
    Args:
        output_dir: Diretório de saída para os CSVs
    """
    print("="*70)
    print("EXPORTANDO AMOSTRAS DAS TABELAS GOLD")
    print("="*70 + "\n")
    
    # Features ML (agregado)
    df_features = sample_features_ml(fraction=0.1)
    csv_path = f"{output_dir}/sample_features_ml.csv"
    df_features.to_csv(csv_path, index=False)
    print(f"✅ Exportado: {csv_path}")
    
    # Microdados alunos
    df_alunos = sample_microdados_alunos(fraction=0.01)
    csv_path = f"{output_dir}/sample_microdados_alunos.csv"
    df_alunos.to_csv(csv_path, index=False)
    print(f"✅ Exportado: {csv_path}")
    
    # Indicadores municipais
    df_indicadores = sample_indicadores_municipio(fraction=0.2)
    csv_path = f"{output_dir}/sample_indicadores_municipio.csv"
    df_indicadores.to_csv(csv_path, index=False)
    print(f"✅ Exportado: {csv_path}")
    
    # Metas UF
    df_metas = sample_metas_uf()
    csv_path = f"{output_dir}/sample_metas_uf.csv"
    df_metas.to_csv(csv_path, index=False)
    print(f"✅ Exportado: {csv_path}")
    
    print(f"\n{'='*70}")
    print("AMOSTRAS EXPORTADAS COM SUCESSO!")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Executar amostragem
    export_samples_to_csv(output_dir='/Workspace/Users/filipe.noberto@redprecatorios.com.br/TechChallenge_3_repo/TechChallenge_Fase3/data')
